from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config import (
    CODE_DIR,
    DEFAULT_CONF,
    FOLDER_OUTPUT_DIR,
    IMAGE_EXTENSIONS,
    IMAGE_OUTPUT_DIR,
    TEST_IMAGES_DIR,
    TEST_VIDEOS_DIR,
    VIDEO_EXTENSIONS,
    VIDEO_OUTPUT_DIR,
    default_model_path,
    ensure_demo_dirs,
)
from terminal_effects import type_input, type_print


def print_header() -> None:
    ensure_demo_dirs()
    model_path = default_model_path()

    print()
    print("=" * 72)
    print("YOLO11 Hava Savunma Demo Menusu")
    print("=" * 72)
    print(f"Model: {model_path}")
    print(f"Model durumu: {'bulundu' if model_path.exists() else 'bulunamadi'}")
    print(f"Resim giris klasoru: {TEST_IMAGES_DIR}")
    print(f"Video giris klasoru: {TEST_VIDEOS_DIR}")
    print(f"Resim ciktilari: {IMAGE_OUTPUT_DIR}")
    print(f"Video ciktilari: {VIDEO_OUTPUT_DIR}")
    print("=" * 72)
    print()


def list_files(folder: Path, extensions: set[str]) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def ask_conf(default: float = DEFAULT_CONF) -> float:
    raw = type_input(f"Confidence threshold [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        type_print("Gecersiz deger, varsayilan kullaniliyor.")
        return default
    return value


def run_script(script_name: str, args: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(CODE_DIR / ".ultralytics_config"))
    env.setdefault("ULTRALYTICS_SETTINGS_DIR", str(CODE_DIR / ".ultralytics_config"))

    command = [sys.executable, str(CODE_DIR / script_name), *args]
    type_print()
    type_print("Calisan komut:")
    type_print(" ".join(f'"{part}"' if " " in part else part for part in command))
    type_print()
    subprocess.run(command, cwd=str(CODE_DIR), env=env, check=False)


def choose_file(files: list[Path], title: str) -> Path | None:
    if not files:
        return None

    type_print(title)
    for idx, path in enumerate(files, start=1):
        type_print(f"{idx}. {path.name}")

    raw = type_input("Secim numarasi veya dosya yolu: ").strip()
    if not raw:
        return files[0]

    if raw.isdigit():
        index = int(raw)
        if 1 <= index <= len(files):
            return files[index - 1]
        type_print("Gecersiz secim.")
        return None

    custom_path = Path(raw)
    if custom_path.exists():
        return custom_path
    type_print(f"Dosya bulunamadi: {custom_path}")
    return None


def run_single_image() -> None:
    files = list_files(TEST_IMAGES_DIR, IMAGE_EXTENSIONS)
    if not files:
        type_print(f"{TEST_IMAGES_DIR} icinde resim yok.")
        raw = type_input("Istersen tek resim yolu gir: ").strip()
        if not raw:
            return
        image_path = Path(raw)
    else:
        image_path = choose_file(files, "Islenecek resmi sec:")
        if image_path is None:
            return

    conf = ask_conf()
    run_script(
        "demo_image.py",
        [
            "--model",
            str(default_model_path()),
            "--source",
            str(image_path),
            "--conf",
            str(conf),
            "--output",
            str(IMAGE_OUTPUT_DIR),
        ],
    )


def run_image_folder() -> None:
    files = list_files(TEST_IMAGES_DIR, IMAGE_EXTENSIONS)
    if not files:
        type_print(f"{TEST_IMAGES_DIR} icinde resim yok. Resimleri buraya atip tekrar dene.")
        return

    conf = ask_conf()
    run_script(
        "demo_folder.py",
        [
            "--model",
            str(default_model_path()),
            "--source",
            str(TEST_IMAGES_DIR),
            "--conf",
            str(conf),
            "--output",
            str(FOLDER_OUTPUT_DIR),
        ],
    )


def run_single_video() -> None:
    files = list_files(TEST_VIDEOS_DIR, VIDEO_EXTENSIONS)
    if not files:
        type_print(f"{TEST_VIDEOS_DIR} icinde video yok.")
        raw = type_input("Istersen tek video yolu gir: ").strip()
        if not raw:
            return
        video_path = Path(raw)
    else:
        video_path = choose_file(files, "Islenecek videoyu sec:")
        if video_path is None:
            return

    conf = ask_conf()
    run_script(
        "demo_video.py",
        [
            "--model",
            str(default_model_path()),
            "--source",
            str(video_path),
            "--conf",
            str(conf),
            "--output",
            str(VIDEO_OUTPUT_DIR),
        ],
    )


def run_all_videos() -> None:
    files = list_files(TEST_VIDEOS_DIR, VIDEO_EXTENSIONS)
    if not files:
        type_print(f"{TEST_VIDEOS_DIR} icinde video yok. Videolari buraya atip tekrar dene.")
        return

    conf = ask_conf()
    for video_path in files:
        run_script(
            "demo_video.py",
            [
                "--model",
                str(default_model_path()),
                "--source",
                str(video_path),
                "--conf",
                str(conf),
                "--output",
                str(VIDEO_OUTPUT_DIR),
            ],
        )


def run_webcam() -> None:
    conf = ask_conf()
    raw_camera = type_input("Kamera index [0]: ").strip()
    camera = raw_camera if raw_camera else "0"
    run_script(
        "demo_webcam.py",
        [
            "--model",
            str(default_model_path()),
            "--conf",
            str(conf),
            "--camera",
            camera,
        ],
    )


def show_folders() -> None:
    ensure_demo_dirs()
    type_print()
    type_print("Dosya atilacak klasorler:")
    type_print(f"- Resimler: {TEST_IMAGES_DIR}")
    type_print(f"- Videolar: {TEST_VIDEOS_DIR}")
    type_print()
    type_print("Sonuc klasorleri:")
    type_print(f"- Resimler: {IMAGE_OUTPUT_DIR}")
    type_print(f"- Videolar: {VIDEO_OUTPUT_DIR}")
    type_print()


def main() -> None:
    actions = {
        "1": run_single_image,
        "2": run_image_folder,
        "3": run_single_video,
        "4": run_all_videos,
        "5": run_webcam,
        "6": show_folders,
    }

    while True:
        print_header()
        print("1. Tek resim isle")
        print("2. test_images klasorundeki tum resimleri isle")
        print("3. Tek video isle")
        print("4. test_videos klasorundeki tum videolari isle")
        print("5. Webcam ile canli tespit")
        print("6. Klasor yollarini goster")
        print("0. Cikis")
        print()

        choice = input("Seciminiz: ").strip()
        if choice == "0":
            type_print("Cikis yapildi.")
            break

        action = actions.get(choice)
        if action is None:
            type_print("Gecersiz secim.")
            input("Devam etmek icin Enter...")
            continue

        action()
        input("Menuye donmek icin Enter...")


if __name__ == "__main__":
    main()
