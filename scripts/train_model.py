from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "dataset_info" / "data.yaml"


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=20,
        imgsz=640,
        batch=-1,
        lr0=0.001,
        patience=8,
        workers=4,
        project=str(ROOT / "training_runs"),
        name="yolo11n_air_defense",
    )


if __name__ == "__main__":
    main()
