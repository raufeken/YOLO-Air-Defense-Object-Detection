from __future__ import annotations

from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_DIR.parent

DEFAULT_MODEL = PACKAGE_ROOT / "model" / "best.pt"
DEFAULT_CONF = 0.40

TEST_IMAGES_DIR = CODE_DIR / "test_images"
TEST_VIDEOS_DIR = CODE_DIR / "test_videos"

IMAGE_OUTPUT_DIR = CODE_DIR / "image_detected"
FOLDER_OUTPUT_DIR = IMAGE_OUTPUT_DIR
VIDEO_OUTPUT_DIR = CODE_DIR / "video_detected"
WEBCAM_OUTPUT_DIR = VIDEO_OUTPUT_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def ensure_demo_dirs() -> None:
    for path in (
        TEST_IMAGES_DIR,
        TEST_VIDEOS_DIR,
        IMAGE_OUTPUT_DIR,
        VIDEO_OUTPUT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def default_model_path() -> Path:
    return DEFAULT_MODEL
