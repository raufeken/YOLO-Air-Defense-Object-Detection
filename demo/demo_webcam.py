"""
Bu dosya webcam goruntusu uzerinde YOLO ile canli hava araci tespiti yapar.
Calistirma:
    python demo_webcam.py --model weights/best.pt --conf 0.25
Model dosyasi:
    Egitilmis model weights/best.pt yolunda olmalidir.
"""

from pathlib import Path
import argparse
import time

import cv2
import torch
from ultralytics import YOLO

from config import DEFAULT_CONF, default_model_path, ensure_demo_dirs
from terminal_effects import type_print as print


def resolve_device(device_arg):
    if device_arg != "auto":
        return device_arg
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
    parser = argparse.ArgumentParser(description="Webcam ile canli YOLO tespiti")
    parser.add_argument("--model", default=str(default_model_path()), help="Egitilmis .pt model yolu")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0 veya 0")
    parser.add_argument("--half", action="store_true", help="CUDA kullanirken FP16 ile calistirir")
    parser.add_argument("--camera", type=int, default=0, help="Kamera index degeri")
    args = parser.parse_args()

    ensure_demo_dirs()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadi. Kontrol edilen yol: {model_path}")

    device = resolve_device(args.device)
    if device == "cpu":
        print("CUDA bulunamadi veya CPU secildi. Model CPU uzerinde calisacak.")
    else:
        print(f"Model su cihazda calisacak: {device}")

    # Egitilmis YOLO modelini yukluyoruz
    model = YOLO(str(model_path))

    # Webcam goruntusunu aciyoruz
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Webcam acilamadi. Farkli kamera icin --camera 1 denenebilir.")

    prev_time = time.time()
    print("Webcam basladi. Cikmak icin q tusuna bas.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kameradan goruntu alinamadi.")
            break

        # Model tahmin yapiyor
        result = model.predict(frame, conf=args.conf, device=device, half=args.half and device != "cpu", verbose=False)[0]

        # Sonuclari goruntunun uzerine ciziyoruz
        annotated = draw_detections(frame, result, result.names)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        draw_fps(annotated, fps)

        cv2.imshow("YOLO Live Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
