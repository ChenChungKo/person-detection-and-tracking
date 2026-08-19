"""Build a Windows onedir folder for launch_detect_grid.py (PyInstaller)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_NAME = "detect_grid_launcher"
DIST_DIR = ROOT / "dist" / DIST_NAME


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    py = Path(sys.executable)
    subprocess.check_call([str(py), "-m", "pip", "install", "pyinstaller"], cwd=str(ROOT))
    cmd = [
        str(py),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        f"--name={DIST_NAME}",
        "--collect-all=ultralytics",
        "--collect-all=torchreid",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=app_paths",
        "--hidden-import=detect_grid",
        "--hidden-import=grid_occupancy",
        "--hidden-import=latest_frame",
        "--hidden-import=stable_id",
        "--hidden-import=reid_encoder",
        str(ROOT / "launch_detect_grid.py"),
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))

    _copy_tree(ROOT / "calibration", DIST_DIR / "calibration")
    _copy_tree(ROOT / "trackers", DIST_DIR / "trackers")
    (DIST_DIR / "test").mkdir(parents=True, exist_ok=True)
    for name in ("test4.mp4", "test.mp4", "static_frame.jpg"):
        src = ROOT / "test" / name
        if src.exists():
            shutil.copy2(src, DIST_DIR / "test" / name)
    models_src = ROOT / "models" / "osnet"
    if models_src.exists():
        _copy_tree(models_src, DIST_DIR / "models" / "osnet")
    for pt in ROOT.glob("yolo26*.pt"):
        shutil.copy2(pt, DIST_DIR / pt.name)

    readme = DIST_DIR / "使用說明.txt"
    readme.write_text(
        "雙擊 detect_grid_launcher.exe 啟動。\n"
        "請把此資料夾整包一起帶走（不要只複製 exe）。\n"
        "YOLO 權重 yolo26s.pt / yolo26s-pose.pt 需與此 exe 同一層。\n"
        "校正檔在 calibration\\，追蹤設定在 trackers\\。\n",
        encoding="utf-8",
    )
    print(f"完成：{DIST_DIR / (DIST_NAME + '.exe')}")


if __name__ == "__main__":
    main()
