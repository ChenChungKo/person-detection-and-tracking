"""Floor Homography from a large printed chessboard on the ground.

Unlike calibrate_lens.py (small board held near the camera), this expects the
board laid flat on the floor and fully visible in one camera frame.

Board defaults match make_chessboard_board.py:
  5x4 squares, 20cm each -> inner corners 4x3

Workflow:
  1) Capture one frame from RTSP (or use an existing image):
       python calibrate_chessboard_floor.py capture --source "rtsp://..."
  2) Detect corners + compute Homography (enter where the board sits in cm):
       python calibrate_chessboard_floor.py calibrate --image calibration/chessboard_floor/capture.jpg
       python calibrate_chessboard_floor.py calibrate --image ... --origin-x 200 --origin-y 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DIR = Path(__file__).resolve().parent / "calibration" / "chessboard_floor"
DEFAULT_CAPTURE = DEFAULT_DIR / "capture.jpg"
DEFAULT_OUT = Path(__file__).resolve().parent / "calibration" / "homography.json"
DEFAULT_INTRINSICS = Path(__file__).resolve().parent / "calibration" / "camera_intrinsics.json"

FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FILTER_QUADS
)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if path.exists():
        cap = cv2.VideoCapture(str(path))
        return cap if cap.isOpened() else None
    return None


def resize_for_preview(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame.copy(), 1.0
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA), scale


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    k = np.array(payload["camera_matrix"], dtype=np.float64)
    d = np.array(payload["dist_coeffs"], dtype=np.float64)
    return k, d


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def find_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Try classic + SB detectors, both orientations, with CLAHE."""
    candidates = [pattern_size]
    if pattern_size[0] != pattern_size[1]:
        candidates.append((pattern_size[1], pattern_size[0]))

    images = [gray, enhance_gray(gray)]
    classic_flags = [
        FIND_FLAGS,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    ]

    for img in images:
        for pattern in candidates:
            for flags in classic_flags:
                ok, corners = cv2.findChessboardCorners(img, pattern, flags=flags)
                if ok:
                    cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
                    return corners, pattern
            if hasattr(cv2, "findChessboardCornersSB"):
                ok, corners = cv2.findChessboardCornersSB(
                    img,
                    pattern,
                    flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
                )
                if ok and corners is not None and len(corners) == pattern[0] * pattern[1]:
                    return corners.astype(np.float32), pattern
    return None


def board_world_points(
    cols: int,
    rows: int,
    square_cm: float,
    origin_xy: tuple[float, float],
    col_axis: str,
    col_dir: str,
    row_axis: str,
    row_dir: str,
) -> np.ndarray:
    """World (X,Y) cm for each inner corner, OpenCV order: row-major over (col, row).

    ``col_axis`` / ``row_axis`` say which floor axis each OpenCV board direction maps to
    (needed because the board may be rotated 90° relative to the room axes).
    """
    if {col_axis, row_axis} != {"x", "y"}:
        raise ValueError("col_axis/row_axis 必須一個是 x、一個是 y")
    ox, oy = origin_xy
    col_sign = 1.0 if col_dir == "increase" else -1.0
    row_sign = 1.0 if row_dir == "increase" else -1.0
    pts = []
    for r in range(rows):
        for c in range(cols):
            dx = 0.0
            dy = 0.0
            if col_axis == "x":
                dx += col_sign * c * square_cm
            else:
                dy += col_sign * c * square_cm
            if row_axis == "x":
                dx += row_sign * r * square_cm
            else:
                dy += row_sign * r * square_cm
            pts.append((ox + dx, oy + dy))
    return np.array(pts, dtype=np.float64)


def make_bird_eye(
    image: np.ndarray,
    h_mat: np.ndarray,
    world_points: np.ndarray,
    px_per_cm: float = 2.0,
    margin_cm: float = 30.0,
) -> np.ndarray:
    xs = world_points[:, 0]
    ys = world_points[:, 1]
    min_x, max_x = float(xs.min()) - margin_cm, float(xs.max()) + margin_cm
    min_y, max_y = float(ys.min()) - margin_cm, float(ys.max()) + margin_cm
    out_w = max(int(np.ceil((max_x - min_x) * px_per_cm)), 100)
    out_h = max(int(np.ceil((max_y - min_y) * px_per_cm)), 100)
    t = np.array(
        [
            [px_per_cm, 0, -min_x * px_per_cm],
            [0, px_per_cm, -min_y * px_per_cm],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return cv2.warpPerspective(image, t @ h_mat, (out_w, out_h))


def cmd_capture(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = open_capture(args.source)
    if cap is None:
        raise SystemExit(f"無法開啟來源：{args.source}")

    win = "Chessboard Floor Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    pattern = (args.cols, args.rows)
    print(f"目標內角點：{pattern[0]}x{pattern[1]}（也會試反向 {pattern[1]}x{pattern[0]}）")
    print(f"方格 {args.square_cm:g}cm。偵測在「全解析度」上跑，不要只看預覽縮圖。")
    print("按 s = 偵測成功才存；按 f = 強制存目前畫面（方便離線除錯）；q = 離開。")
    print("若一直 not found：減少反光、讓拼縫貼緊／黑膠帶蓋縫、確認 5x4 格完整入鏡。")

    last_corners: np.ndarray | None = None
    last_pattern: tuple[int, int] | None = None
    detect_every = max(1, args.detect_every)
    frame_i = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame_i += 1
            if frame_i % detect_every == 0:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found = find_corners(gray_full, pattern)
                if found is not None:
                    last_corners, last_pattern = found
                else:
                    last_corners, last_pattern = None, None

            view, scale = resize_for_preview(frame, args.max_width)
            canvas = view.copy()
            if last_corners is not None and last_pattern is not None:
                corners_view = (last_corners.reshape(-1, 1, 2) * scale).astype(np.float32)
                cv2.drawChessboardCorners(canvas, last_pattern, corners_view, True)
                tip = f"FOUND {last_pattern[0]}x{last_pattern[1]} - press s"
                color = (0, 255, 0)
            else:
                tip = "not found - press f to force-save"
                color = (0, 0, 255)
            cv2.putText(canvas, tip, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found = find_corners(gray_full, pattern)
                if found is None:
                    print("  仍偵測不到。可按 f 先強制存圖，或改善拼縫／反光後再試。")
                    continue
                corners_full, used_pattern = found
                imwrite_unicode(out, frame)
                dbg = frame.copy()
                cv2.drawChessboardCorners(dbg, used_pattern, corners_full, True)
                imwrite_unicode(out.with_name("capture_corners.jpg"), dbg)
                print(f"已存：{out}")
                print(f"角點預覽：{out.with_name('capture_corners.jpg')}（pattern={used_pattern}）")
                break
            if key == ord("f"):
                imwrite_unicode(out, frame)
                print(f"已強制存圖（未確認角點）：{out}")
                print("之後可離線試：python calibrate_chessboard_floor.py calibrate --image ...")
                break
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def cmd_calibrate(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    image = imread_unicode(image_path)
    if image is None:
        raise SystemExit(f"無法讀取影像：{image_path}")

    pattern = (args.cols, args.rows)
    work = image
    intr = load_intrinsics(Path(args.intrinsics)) if args.intrinsics else None
    if intr is not None:
        k, d = intr
        work = cv2.undistort(image, k, d)
        print(f"已套用鏡頭校正：{args.intrinsics}")

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    found = find_corners(gray, pattern)
    if found is None:
        raise SystemExit(
            f"偵測不到 {pattern[0]}x{pattern[1]} 內角點。"
            "常見原因：拼縫把格子切斷、反光過強、外圍白邊不足、板子不完整入鏡。"
        )
    corners, used_pattern = found
    args.cols, args.rows = used_pattern

    dbg = work.copy()
    cv2.drawChessboardCorners(dbg, used_pattern, corners, True)
    # Mark corner 0 (board local origin used below)
    p0 = corners.reshape(-1, 2)[0]
    cv2.circle(dbg, (int(p0[0]), int(p0[1])), 18, (0, 0, 255), 3)
    cv2.putText(
        dbg,
        "origin corner #0",
        (int(p0[0]) + 20, int(p0[1]) - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    debug_path = Path(args.out).parent / "chessboard_floor" / "detected_corners.jpg"
    imwrite_unicode(debug_path, dbg)
    print(f"角點偵測成功（{used_pattern[0]}x{used_pattern[1]} = {used_pattern[0] * used_pattern[1]} 點），預覽：{debug_path}")
    print("紅色圓圈＝角點 #0（當作棋盤局部原點）。請確認它對應你量測的那一角。")

    if args.origin_x is None or args.origin_y is None:
        print()
        print("請輸入角點 #0 在地板座標系的位置（cm，相對虛擬 TL=(0,0)）：")
        try:
            ox = float(input("  origin X_cm: ").strip())
            oy = float(input("  origin Y_cm: ").strip())
        except ValueError as exc:
            raise SystemExit("座標格式錯誤") from exc
    else:
        ox, oy = float(args.origin_x), float(args.origin_y)

    x_sign_note = f"col→{args.col_axis}({args.col_dir}), row→{args.row_axis}({args.row_dir})"
    world = board_world_points(
        args.cols,
        args.rows,
        args.square_cm,
        (ox, oy),
        args.col_axis,
        args.col_dir,
        args.row_axis,
        args.row_dir,
    )
    print(f"軸向對應：{x_sign_note}")
    img_pts = corners.reshape(-1, 2).astype(np.float64)

    h_mat, _ = cv2.findHomography(img_pts, world, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if h_mat is None:
        raise SystemExit("Homography 計算失敗")

    projected = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
    errs = np.linalg.norm(projected - world, axis=1)
    mean_err = float(errs.mean())
    max_err = float(errs.max())
    print(f"平均重投影誤差：{mean_err:.2f} cm（最大 {max_err:.2f} cm）")

    bird = make_bird_eye(work, h_mat, world)
    bird_path = Path(args.out).parent / "chessboard_floor" / "bird_eye_preview.jpg"
    imwrite_unicode(bird_path, bird)
    print(f"鳥瞰預覽：{bird_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        "image_size_wh": [int(work.shape[1]), int(work.shape[0])],
        "tile_cm": args.square_cm,
        "units": "cm",
        "method": "chessboard_floor",
        "pattern_inner_corners": [args.cols, args.rows],
        "square_cm": args.square_cm,
        "origin_corner_world_xy": [ox, oy],
        "col_axis": args.col_axis,
        "col_dir": args.col_dir,
        "row_axis": args.row_axis,
        "row_dir": args.row_dir,
        "roi": {
            "origin": "top_left_virtual",
            "note": "World coords relative to virtual TL=(0,0); chessboard anchors only cover the board area.",
        },
        "image_points_xy": img_pts.tolist(),
        "world_points_xy": world.tolist(),
        "homography": h_mat.tolist(),
        "mean_reproj_error_cm": mean_err,
        "max_reproj_error_cm": max_err,
        "undistorted_with": str(args.intrinsics) if intr is not None else None,
        "note": "Maps image pixels to floor plane (X_cm, Y_cm) from large floor chessboard.",
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已儲存：{out_path}")
    print("若要先比較舊結果，可改 --out 存成另一個檔名，不要直接覆蓋。")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Floor Homography from large chessboard")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("capture", help="grab a frame from RTSP / video / webcam")
    pc.add_argument("--source", required=True)
    pc.add_argument("--out", default=str(DEFAULT_CAPTURE))
    pc.add_argument("--cols", type=int, default=4, help="inner corners across")
    pc.add_argument("--rows", type=int, default=3, help="inner corners down")
    pc.add_argument("--square-cm", type=float, default=20.0)
    pc.add_argument("--max-width", type=int, default=1280)
    pc.add_argument(
        "--detect-every",
        type=int,
        default=5,
        help="run full-res detection every N frames (lower = smoother but slower)",
    )
    pc.set_defaults(func=cmd_capture)

    pcal = sub.add_parser("calibrate", help="detect corners and compute Homography")
    pcal.add_argument("--image", default=str(DEFAULT_CAPTURE))
    pcal.add_argument("--cols", type=int, default=4)
    pcal.add_argument("--rows", type=int, default=3)
    pcal.add_argument("--square-cm", type=float, default=20.0)
    pcal.add_argument("--origin-x", type=float, default=None, help="world X_cm of corner #0")
    pcal.add_argument("--origin-y", type=float, default=None, help="world Y_cm of corner #0")
    pcal.add_argument(
        "--col-axis",
        choices=["x", "y"],
        default="y",
        help="floor axis along OpenCV board columns (corner 0→1); default y for this room CCTV",
    )
    pcal.add_argument(
        "--col-dir",
        choices=["increase", "decrease"],
        default="decrease",
        help="does that floor axis increase or decrease from corner 0→1?",
    )
    pcal.add_argument(
        "--row-axis",
        choices=["x", "y"],
        default="x",
        help="floor axis along OpenCV board rows (corner 0→next-row); default x",
    )
    pcal.add_argument(
        "--row-dir",
        choices=["increase", "decrease"],
        default="increase",
        help="does that floor axis increase or decrease along board rows?",
    )
    pcal.add_argument(
        "--intrinsics",
        default="",
        help="optional camera_intrinsics.json; if present, undistort before Homography",
    )
    pcal.add_argument("--out", default=str(DEFAULT_OUT))
    pcal.set_defaults(func=cmd_calibrate)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "calibrate" and not args.intrinsics:
        # auto-use lens file if it already exists
        if DEFAULT_INTRINSICS.exists():
            args.intrinsics = str(DEFAULT_INTRINSICS)
        else:
            args.intrinsics = ""
    args.func(args)


if __name__ == "__main__":
    main()
