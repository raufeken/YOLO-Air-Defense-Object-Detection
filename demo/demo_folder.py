"""
Bu dosya bir klasordeki fotograflar uzerinde YOLO ile hava araci tespiti yapar.
Calistirma:
    python demo_folder.py --model weights/best.pt --source test_images --conf 0.40
Model dosyasi:
    Egitilmis model weights/best.pt yolunda olmalidir.
"""

from pathlib import Path
import argparse
import time

import cv2
import torch
from ultralytics import YOLO

from config import DEFAULT_CONF, FOLDER_OUTPUT_DIR, IMAGE_EXTENSIONS, default_model_path, ensure_demo_dirs
from terminal_effects import type_print as print


def resolve_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def draw_detections(frame, results, class_names):
    if results.boxes is None:
        return frame

    for box in results.boxes:
        # xyxy: kutunun sol ust ve sag alt noktalaridir
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        # conf: modelin bu tahmine verdigi guven skorudur
        conf = float(box.conf[0])
        # cls_id: tahmin edilen sinif numarasidir
        cls_id = int(box.cls[0])

        cls_name = class_names.get(cls_id, str(cls_id))
        label = f"{cls_name} {conf:.2f}"

        # Kutuyu ve sinif etiketini ciziyoruz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        label_y = max(y1 - 8, text_h + 8)
        cv2.rectangle(frame, (x1, label_y - text_h - 6), (x1 + text_w + 6, label_y + 4), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return frame


def draw_fps(frame, fps):
    text = f"FPS: {fps:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    margin = 10
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x = max(frame.shape[1] - text_w - margin, margin)
    y = max(frame.shape[0] - margin, text_h + margin)
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Klasordeki fotograflarda YOLO tespiti")
    parser.add_argument("--model", default=str(default_model_path()), help="Egitilmis .pt model yolu")
    parser.add_argument("--source", required=True, help="Test gorsellerinin oldugu klasor")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--output", default=str(FOLDER_OUTPUT_DIR), help="Cikti klasoru")
    args = parser.parse_args()

    ensure_demo_dirs()
    model_path = Path(args.model)
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadi. Kontrol edilen yol: {model_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Klasor bulunamadi. Kontrol edilen yol: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Bu yol klasor degil: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f"Bu klasorde gorsel bulunamadi: {source_dir}")
        return

    # Egitilmis YOLO modelini yukluyoruz
    device = resolve_device()
    print(f"Model su cihazda calisacak: {device}")
    model = YOLO(str(model_path))

    total_time = 0.0
    processed = 0

    for image_path in image_paths:
        # Fotografi okuyoruz
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Okunamadi, atlandi: {image_path}")
            continue

        # Model tahmin yapiyor
        start = time.time()
        result = model.predict(frame, conf=args.conf, device=device, verbose=False)[0]
        elapsed = time.time() - start
        total_time += elapsed
        processed += 1
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # Sonuclari goruntunun uzerine ciziyoruz
        annotated = draw_detections(frame, result, result.names)
        draw_fps(annotated, fps)

        save_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
        cv2.imwrite(str(save_path), annotated)
        print(f"Kaydedildi: {save_path}")

    avg_fps = processed / total_time if total_time > 0 else 0.0
    print(f"Islenen gorsel sayisi: {processed}")
    print(f"Ortalama FPS: {avg_fps:.2f}")
    print(f"Cikti klasoru: {output_dir}")


if __name__ == "__main__":
    main()
