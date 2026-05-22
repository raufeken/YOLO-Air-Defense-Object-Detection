from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "best.pt"
DATA_YAML = ROOT / "dataset_info" / "data.yaml"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    model = YOLO(str(MODEL_PATH))
    model.val(data=str(DATA_YAML), split="test", imgsz=640, conf=0.40)


if __name__ == "__main__":
    main()
