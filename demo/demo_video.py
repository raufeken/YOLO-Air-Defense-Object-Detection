"""
Bu dosya bir video uzerinde YOLO ile hava araci tespiti yapar.
Calistirma:
    python demo_video.py --model weights/best.pt --source videos/test.mp4 --conf 0.25
Model dosyasi:
    Egitilmis model weights/best.pt yolunda olmalidir.
"""

from pathlib import Path
import argparse
import time

import cv2
import torch
from ultralytics import YOLO

from config import DEFAULT_CONF, VIDEO_OUTPUT_DIR, default_model_path, ensure_demo_dirs
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
    parser = argparse.ArgumentParser(description="Video uzerinde YOLO tespiti")
    parser.add_argument("--model", default=str(default_model_path()), help="Egitilmis .pt model yolu")
    parser.add_argument("--source", required=True, help="Islenecek video yolu")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--output", default=str(VIDEO_OUTPUT_DIR), help="Cikti klasoru")
    args = parser.parse_args()

    ensure_demo_dirs()
    model_path = Path(args.model)
    video_path = Path(args.source)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadi. Kontrol edilen yol: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadi. Kontrol edilen yol: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Egitilmis YOLO modelini yukluyoruz
    device = resolve_device()
    print(f"Model su cihazda calisacak: {device}")
    model = YOLO(str(model_path))

    # Videoyu okuyoruz
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi. Kontrol edilen yol: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    output_path = output_dir / f"{video_path.stem}_detected.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), input_fps, (width, height))

    frame_count = 0
    total_inference_time = 0.0

    print("Video isleniyor...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Model tahmin yapiyor
        start = time.time()
        result = model.predict(frame, conf=args.conf, device=device, verbose=False)[0]
        elapsed = time.time() - start
        total_inference_time += elapsed
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # Sonuclari goruntunun uzerine ciziyoruz
        annotated = draw_detections(frame, result, result.names)
        draw_fps(annotated, fps)

        writer.write(annotated)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Islenen frame: {frame_count}")

    cap.release()
    writer.release()

    avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0.0
    print("Video isleme tamamlandi.")
    print(f"Toplam frame: {frame_count}")
    print(f"Ortalama inference FPS: {avg_fps:.2f}")
    print(f"Cikti videosu: {output_path}")


if __name__ == "__main__":
    main()
