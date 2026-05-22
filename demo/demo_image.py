"""
Bu dosya tek bir fotograf uzerinde YOLO ile hava araci tespiti yapar.
Calistirma:
    python demo_image.py --model weights/best.pt --source test_images/ornek.jpg --conf 0.25
Model dosyasi:
    Egitilmis model weights/best.pt yolunda olmalidir.
"""

from pathlib import Path
import argparse
import time

import cv2
import torch
from ultralytics import YOLO

from config import DEFAULT_CONF, IMAGE_OUTPUT_DIR, default_model_path, ensure_demo_dirs
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
    parser = argparse.ArgumentParser(description="Tek fotografta YOLO tespiti")
    parser.add_argument("--model", default=str(default_model_path()), help="Egitilmis .pt model yolu")
    parser.add_argument("--source", required=True, help="Islenecek fotograf yolu")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--output", default=str(IMAGE_OUTPUT_DIR), help="Cikti klasoru")
    args = parser.parse_args()

    ensure_demo_dirs()
    model_path = Path(args.model)
    source_path = Path(args.source)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadi. Kontrol edilen yol: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Fotograf bulunamadi. Kontrol edilen yol: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Egitilmis YOLO modelini yukluyoruz
    device = resolve_device()
    print(f"Model su cihazda calisacak: {device}")
    model = YOLO(str(model_path))

    # Fotografi okuyoruz
    frame = cv2.imread(str(source_path))
    if frame is None:
        raise ValueError(f"Fotograf okunamadi. Kontrol edilen yol: {source_path}")

    # Model tahmin yapiyor
    start = time.time()
    result = model.predict(frame, conf=args.conf, device=device, verbose=False)[0]
    elapsed = time.time() - start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    # Sonuclari goruntunun uzerine ciziyoruz
    annotated = draw_detections(frame, result, result.names)
    draw_fps(annotated, fps)

    save_path = output_dir / f"{source_path.stem}_detected{source_path.suffix}"
    cv2.imwrite(str(save_path), annotated)

    print("Tespit tamamlandi.")
    print(f"Cikti: {save_path}")


if __name__ == "__main__":
    main()
