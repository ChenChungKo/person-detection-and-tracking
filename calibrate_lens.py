"""Camera lens (intrinsics + distortion) calibration using a small handheld chessboard.

This is a DIFFERENT calibration from the floor Homography: here the chessboard
is held close to the camera (filling a good chunk of the frame) and photographed
at many different angles/positions, NOT laid flat on the distant floor.

Two stages:

1) Capture frames (live preview, press 's' to save when a board is detected):
   python calibrate_lens.py capture --source "rtsp://user:pass@ip:554/stream1"
   python calibrate_lens.py capture --source 0

2) Compute intrinsics from the saved frames:
   python calibrate_lens.py calibrate

Defaults match a 7x6-inner-corner, 25mm-square chessboard (chessboard_7x6_25mm).
Override with --cols/--rows/--square-cm if you used a different board.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_FRAMES_DIR = Path(__file__).resolve().parent / "calibration" / "lens_frames"
DEFAULT_OUT = Path(__file__).resolve().parent / "calibration" / "camera_intrinsics.json"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


def open_capture(source: str) -> cv2.VideoCapture | None:
    if source.lower().startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        return cap if cap.isOpened() else None
    path = Path(source)
    if path.exists() and path.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        return cap if cap.isOpened() else None
    return None


def resize_for_preview(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame.copy(), 1.0
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA), scale


FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)


def find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags=FIND_FLAGS)
    if not ok:
        return None
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return corners


def cmd_capture(args: argparse.Namespace) -> None:
    pattern_size = (args.cols, args.rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    n = len(existing)

    cap = open_capture(args.source)
    if cap is None:
        raise SystemExit(f"無法開啟來源：{args.source}")

    win = "Lens Calibration Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"棋盤格內角點：{pattern_size[0]}x{pattern_size[1]}，方格 {args.square_cm:g}cm")
    print("把板子拿到鏡頭前，填滿畫面一部分，多角度、多距離、多位置（含四個角落）。")
    print("偵測到棋盤格時畫面會出現彩色角點；按 s 存檔，q 結束。")
    print(f"目前已有 {n} 張，建議累積到 15~20 張再進入 calibrate 階段。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("讀取失敗或結束。")
                break

            view, scale = resize_for_preview(frame, args.max_width)
            gray_preview = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            found_preview, corners_preview = cv2.findChessboardCorners(
                gray_preview, pattern_size, flags=FIND_FLAGS
            )
            canvas = view.copy()
            if found_preview:
                cv2.drawChessboardCorners(canvas, pattern_size, corners_preview, found_preview)
            status = "FOUND (press s to save)" if found_preview else "not found"
            cv2.putText(
                canvas,
                f"{status}  saved={n}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if found_preview else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners_full = find_corners(gray_full, pattern_size)
                if corners_full is None:
                    print("  存檔失敗：這一幀在全解析度下偵測不到棋盤格，調整角度/距離再試。")
                    continue
                n += 1
                path = out_dir / f"frame_{n:03d}.jpg"
                imwrite_unicode(path, frame)
                print(f"  已存：{path}（第 {n} 張）")
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"共 {n} 張，存放於 {out_dir}")
    print("接著執行：python calibrate_lens.py calibrate")


def cmd_calibrate(args: argparse.Namespace) -> None:
    pattern_size = (args.cols, args.rows)
    frames_dir = Path(args.frames_dir)
    paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if len(paths) < 5:
        raise SystemExit(f"照片太少（{len(paths)} 張），建議至少 10~15 張：{frames_dir}")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * args.square_cm

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    used_files: list[str] = []
    image_size: tuple[int, int] | None = None

    for path in paths:
        img = imread_unicode(path)
        if img is None:
            print(f"  跳過（讀取失敗）：{path.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        corners = find_corners(gray, pattern_size)
        if corners is None:
            print(f"  跳過（偵測不到棋盤格）：{path.name}")
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        used_files.append(path.name)
        print(f"  可用：{path.name}")

    if len(objpoints) < 5:
        raise SystemExit(f"成功偵測到棋盤格的照片太少（{len(objpoints)} 張），至少需要 5 張以上，建議 10+ 張。")

    assert image_size is not None
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    per_view_errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(err))

    mean_err_px = float(np.mean(per_view_errors))
    print()
    print(f"用了 {len(objpoints)} / {len(paths)} 張照片")
    print(f"整體 RMS 重投影誤差：{ret:.4f} px")
    print(f"平均逐張重投影誤差：{mean_err_px:.4f} px（越小越好，一般 <0.5px 算不錯）")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coeffs (k1,k2,p1,p2,k3,...):")
    print(dist_coeffs.ravel())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_size_wh": [int(image_size[0]), int(image_size[1])],
        "pattern_inner_corners": [pattern_size[0], pattern_size[1]],
        "square_cm": args.square_cm,
        "num_images_used": len(objpoints),
        "num_images_total": len(paths),
        "used_files": used_files,
        "rms_reproj_error_px": float(ret),
        "mean_per_view_reproj_error_px": mean_err_px,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "note": "cv2.calibrateCamera intrinsics; use cv2.undistort(frame, camera_matrix, dist_coeffs) before Homography for best floor-plane accuracy.",
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已儲存：{out_path}")

    if args.preview:
        sample = imread_unicode(paths[0])
        if sample is not None:
            undist = cv2.undistort(sample, camera_matrix, dist_coeffs)
            side_by_side = np.hstack([sample, undist])
            preview_path = out_path.with_name("undistort_preview.jpg")
            imwrite_unicode(preview_path, side_by_side)
            print(f"已存對照圖（左=原圖，右=校正後）：{preview_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera lens intrinsics calibration")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("capture", help="live-capture chessboard frames near the camera")
    pc.add_argument("--source", required=True, help="rtsp:// URL, video file, or webcam index")
    pc.add_argument("--out-dir", default=str(DEFAULT_FRAMES_DIR))
    pc.add_argument("--cols", type=int, default=7, help="inner corners across")
    pc.add_argument("--rows", type=int, default=6, help="inner corners down")
    pc.add_argument("--square-cm", type=float, default=2.5, help="physical square edge length (cm)")
    pc.add_argument("--max-width", type=int, default=1280)
    pc.set_defaults(func=cmd_capture)

    pcal = sub.add_parser("calibrate", help="compute intrinsics from saved frames")
    pcal.add_argument("--frames-dir", default=str(DEFAULT_FRAMES_DIR))
    pcal.add_argument("--cols", type=int, default=7, help="inner corners across")
    pcal.add_argument("--rows", type=int, default=6, help="inner corners down")
    pcal.add_argument("--square-cm", type=float, default=2.5, help="physical square edge length (cm)")
    pcal.add_argument("--out", default=str(DEFAULT_OUT))
    pcal.add_argument("--preview", action="store_true", help="also save an undistort before/after JPEG")
    pcal.set_defaults(func=cmd_calibrate)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
