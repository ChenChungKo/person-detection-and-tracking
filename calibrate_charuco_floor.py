"""ChArUco floor Homography. Does not replace calibration/homography.json.

Opens a window on the latest capture. The marked green point is the detected
corner nearest the camera and on the image-left side. Enter that point's
floor coordinates in centimetres. The short side runs to +X; the long side
runs away from the camera (Y decreases).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "calibration" / "charuco_floor"
OUT_JSON = ROOT / "calibration" / "homography_charuco.json"
OLD_JSON = ROOT / "calibration" / "homography_v2_chessboard.json"
TAPE_JSON = ROOT / "calibration" / "homography_error_report.json"
RTSP = "rtsp://oriongo:123456789@192.168.0.200:554/stream1"
CAPTURE = OUT_DIR / "capture.jpg"

SQUARE_CM = 20.0
BOARD_W = 100.0
BOARD_H = 80.0


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


def grab_rtsp() -> np.ndarray:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame = None
    for _ in range(20):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
    cap.release()
    if frame is None:
        raise SystemExit("無法讀取 RTSP")
    imwrite_unicode(CAPTURE, frame)
    return frame


def detect(frame: np.ndarray):
    """Return image points, board-cm points, the four outer paper corners in pixels, and the BL index."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 4), SQUARE_CM / 100.0, 14.0 / 100.0, dictionary)
    corners3d = board.getChessboardCorners().reshape(-1, 3)
    params = cv2.aruco.CharucoParameters()
    params.checkMarkers = False
    params.minMarkers = 1
    cc, ids, _mc, _mids = cv2.aruco.CharucoDetector(board, params).detectBoard(frame)
    n_c = 0 if ids is None else len(ids)
    if ids is None or n_c < 6:
        raise SystemExit(f"角點太少（{n_c}/12）。先把紙攤平，確認整塊入鏡。")

    ids = ids.reshape(-1)
    img_pts = cc.reshape(-1, 2).astype(np.float64)
    board_cm = np.array([corners3d[int(i), :2] * 100.0 for i in ids], dtype=np.float64)
    aff, inliers = cv2.estimateAffine2D(
        board_cm.reshape(-1, 1, 2),
        img_pts.reshape(-1, 1, 2),
        ransacReprojThreshold=15,
    )
    if aff is None or inliers is None:
        raise SystemExit("無法對齊板子座標")
    pred = (aff[:, :2] @ board_cm.T).T + aff[:, 2]
    keep = np.linalg.norm(pred - img_pts, axis=1) < 40
    if int(keep.sum()) >= 6:
        img_pts = img_pts[keep]
        board_cm = board_cm[keep]
        aff, _ = cv2.estimateAffine2D(board_cm.reshape(-1, 1, 2), img_pts.reshape(-1, 1, 2))
    if aff is None:
        raise SystemExit("無法對齊板子座標")

    def board_to_px(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return (aff[:, :2] @ pts.T).T + aff[:, 2]

    expected = corners3d[:, :2] * 100.0
    have = {(int(round(float(p[0]))), int(round(float(p[1])))) for p in board_cm}
    missing = [p for p in expected if (int(round(float(p[0]))), int(round(float(p[1])))) not in have]
    if missing:
        missing_cm = np.asarray(missing, dtype=np.float64)
        img_pts = np.vstack([img_pts, board_to_px(missing_cm)])
        board_cm = np.vstack([board_cm, missing_cm])

    outer_b = np.array([[0, 0], [BOARD_W, 0], [BOARD_W, BOARD_H], [0, BOARD_H]], dtype=np.float64)
    outer_px = board_to_px(outer_b)
    near = np.argsort(-outer_px[:, 1])[:2]
    bl = int(near[np.argmin(outer_px[near, 0])])
    br = int(near[np.argmax(outer_px[near, 0])])
    far = [i for i in range(4) if i not in (bl, br)]
    away = int(min(far, key=lambda i: outer_px[i, 0]))
    green = int(np.argmin(np.linalg.norm(board_cm - outer_b[bl], axis=1)))
    return img_pts, board_cm, outer_b, bl, br, away, green


def solve(img_pts: np.ndarray, board_cm: np.ndarray, outer_b: np.ndarray, bl: int, br: int, away: int, green: int, origin_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rb = outer_b[br] - outer_b[bl]
    ab = outer_b[away] - outer_b[bl]
    near_cm = float(np.linalg.norm(rb))
    away_cm = float(np.linalg.norm(ab))
    basis = np.column_stack([rb, ab])
    world_axes = np.array([[near_cm, 0.0], [0.0, -away_cm]], dtype=np.float64).T
    mapping = world_axes @ np.linalg.inv(basis)
    world = (mapping @ (board_cm - board_cm[green]).T).T + np.array(origin_xy)
    h_mat, _ = cv2.findHomography(img_pts, world, method=0)
    if h_mat is None:
        raise SystemExit("Homography 失敗")
    projected = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
    err = np.linalg.norm(projected - world, axis=1)
    return h_mat, world, err


def tape_report(h_mat: np.ndarray) -> str:
    if not OLD_JSON.exists() or not TAPE_JSON.exists():
        return ""
    old = np.array(json.loads(OLD_JSON.read_text(encoding="utf-8"))["homography"], dtype=np.float64)
    lines = ["捲尺點    舊棋盤    這次"]
    old_errs = []
    new_errs = []
    for s in json.loads(TAPE_JSON.read_text(encoding="utf-8"))["samples"]:
        ip = np.array(s["image_xy"], dtype=np.float64).reshape(1, 1, 2)
        truth = np.array(s["truth_world_xy"], dtype=np.float64)
        pred_old = cv2.perspectiveTransform(ip, old).reshape(2)
        pred_new = cv2.perspectiveTransform(ip, h_mat).reshape(2)
        eo = float(np.linalg.norm(pred_old - truth))
        en = float(np.linalg.norm(pred_new - truth))
        old_errs.append(eo)
        new_errs.append(en)
        lines.append(f"{truth[0]:.0f}, {truth[1]:.0f}    {eo:.1f} cm    {en:.1f} cm")
    lines.append(f"平均    {np.mean(old_errs):.1f} cm    {np.mean(new_errs):.1f} cm")
    text = "\n".join(lines)
    print(text)
    return text


def save_result(frame: np.ndarray, h_mat: np.ndarray, img_pts: np.ndarray, world: np.ndarray, err: np.ndarray, origin_xy: tuple[float, float], tape_text: str) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_path": str(CAPTURE),
        "image_size_wh": [int(frame.shape[1]), int(frame.shape[0])],
        "units": "cm",
        "method": "charuco_floor",
        "dictionary": "DICT_4X4_50",
        "squares_xy": [5, 4],
        "square_cm": SQUARE_CM,
        "marker_cm": 14.0,
        "bottom_left_green_world_xy": [origin_xy[0], origin_xy[1]],
        "axis": "near edge +X, away from camera -Y",
        "homography": h_mat.tolist(),
        "image_points_xy": img_pts.tolist(),
        "world_points_xy": world.tolist(),
        "mean_reproj_error_cm": float(err.mean()),
        "max_reproj_error_cm": float(err.max()),
        "tape_report": tape_text,
        "note": "Not the live homography. The entered value is the bottom-left green corner, not the paper edge.",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已存：{OUT_JSON}")


def _step_index(values: np.ndarray, zero_at_high: bool) -> tuple[list[int], int]:
    levels = sorted({int(round(float(v))) for v in values})
    if zero_at_high:
        levels = levels[::-1]
    rank = {level: i for i, level in enumerate(levels)}
    return [rank[int(round(float(v)))] for v in values], len(levels)


def display_numbers(img_pts: np.ndarray, board_cm: np.ndarray, green: int) -> list[int]:
    """Number like the old chessboard figure.

    0 is the inner corner nearest the camera and to the left. Numbers then
    increase away from the camera, and the next column to the right continues
    the sequence.
    """
    del green  # origin is chosen from the image, then matched to board axes
    corr = []
    for axis in (0, 1):
        corr.append(float(np.corrcoef(board_cm[:, axis], img_pts[:, 1])[0, 1]))
    col_axis = 0 if abs(corr[0]) >= abs(corr[1]) else 1
    row_axis = 1 - col_axis
    # Larger image y is closer to the camera, so that end of the column is 0.
    col_steps, n_col = _step_index(board_cm[:, col_axis], zero_at_high=corr[col_axis] > 0)
    row_corr = float(np.corrcoef(board_cm[:, row_axis], img_pts[:, 0])[0, 1])
    row_steps, _n_row = _step_index(board_cm[:, row_axis], zero_at_high=row_corr < 0)
    return [c + r * n_col for c, r in zip(col_steps, row_steps)]


def draw_numbered(frame: np.ndarray, img_pts: np.ndarray, numbers: list[int]) -> np.ndarray:
    """Draw 0–11 on the full camera view, same layout as the old chessboard preview."""
    canvas = frame.copy()
    s = frame.shape[1] / 2880.0
    radius = max(8, int(12 * s))
    thick = max(2, int(round(2 * s)))
    font = 0.62 * s
    font_thick = max(2, int(round(2 * s)))
    placed = {n: (int(round(img_pts[i, 0])), int(round(img_pts[i, 1]))) for i, n in enumerate(numbers)}
    center = img_pts.mean(axis=0)
    for col in range(3):
        chain = [placed[col * 4 + k] for k in range(4) if col * 4 + k in placed]
        if len(chain) >= 2:
            cv2.polylines(canvas, [np.array(chain, np.int32)], False, (0, 140, 255), thick, cv2.LINE_AA)
    for n, c in placed.items():
        away = np.array(c, dtype=np.float64) - center
        norm = float(np.linalg.norm(away))
        away = away / norm if norm > 1 else np.array([1.0, -1.0])
        label = (int(c[0] + away[0] * radius * 2.4), int(c[1] + away[1] * radius * 2.4))
        if n == 0:
            cv2.circle(canvas, c, int(radius * 1.6), (0, 0, 255), thick + 1, cv2.LINE_AA)
            cv2.putText(canvas, "0", (c[0] + radius, c[1] - radius), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 255), font_thick, cv2.LINE_AA)
            cv2.putText(canvas, "ORIGIN #0", (c[0] + radius + 4, c[1] + int(18 * s)), cv2.FONT_HERSHEY_SIMPLEX, font * 1.15, (0, 0, 255), font_thick, cv2.LINE_AA)
        else:
            cv2.circle(canvas, c, radius, (0, 255, 255), thick, cv2.LINE_AA)
            cv2.putText(canvas, str(n), label, cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 255), font_thick, cv2.LINE_AA)
    return canvas


def ask_origin(preview_bgr: np.ndarray) -> tuple[float, float] | None:
    import tkinter as tk
    from tkinter import messagebox

    from PIL import Image, ImageTk

    rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    scale = min(1.0, 1400 / rgb.shape[1], 820 / rgb.shape[0])
    shown = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    root = tk.Tk()
    root.title("ChArUco 交點編號")
    root.configure(bg="white")
    try:
        root.tk.call("tk", "scaling", 1.4)
    except tk.TclError:
        pass

    photo = ImageTk.PhotoImage(Image.fromarray(shown))
    tk.Label(root, image=photo, bg="white").pack(padx=12, pady=(12, 4))
    tk.Label(
        root,
        text="紅圈 0 是最靠近鏡頭、靠左的交點。請輸入這個點的實際座標（cm）。",
        bg="white",
        fg="#222",
        font=("Microsoft JhengHei UI", 12),
    ).pack(anchor="w", padx=16)

    form = tk.Frame(root, bg="white")
    form.pack(fill="x", padx=16, pady=10)
    x_var = tk.StringVar()
    y_var = tk.StringVar()
    result: dict[str, tuple[float, float] | None] = {"xy": None}

    def field(row: int, name: str, var: tk.StringVar) -> None:
        tk.Label(form, text=name, bg="white", font=("Microsoft JhengHei UI", 12)).grid(row=row, column=0, sticky="w", pady=4)
        tk.Entry(form, textvariable=var, width=12, font=("Microsoft JhengHei UI", 14)).grid(row=row, column=1, sticky="w", padx=8)

    field(0, "X", x_var)
    field(1, "Y", y_var)

    def accept(_event=None) -> None:
        try:
            result["xy"] = (float(x_var.get().strip()), float(y_var.get().strip()))
        except ValueError:
            messagebox.showerror("座標", "X 和 Y 請填數字，單位是公分。", parent=root)
            return
        root.destroy()

    def cancel() -> None:
        root.destroy()

    buttons = tk.Frame(root, bg="white")
    buttons.pack(pady=(0, 14))
    tk.Button(buttons, text="計算並儲存", command=accept, font=("Microsoft JhengHei UI", 12), width=12).pack(side="left", padx=6)
    tk.Button(buttons, text="取消", command=cancel, font=("Microsoft JhengHei UI", 12), width=8).pack(side="left", padx=6)
    root.bind("<Return>", accept)
    root.mainloop()
    return result["xy"]


def main() -> None:
    refresh = "--capture" in sys.argv
    if refresh or not CAPTURE.exists():
        frame = grab_rtsp()
    else:
        frame = imread_unicode(CAPTURE)
        if frame is None:
            raise SystemExit(f"無法讀取 {CAPTURE}")
    print(f"畫面 {frame.shape[1]}x{frame.shape[0]}  {CAPTURE}")

    img_pts, board_cm, outer_b, bl, br, away, green = detect(frame)
    print(f"角點 {len(img_pts)}/12，左下綠點像素 {img_pts[green].round(1).tolist()}")

    numbers = display_numbers(img_pts, board_cm, green)
    print("編號:", " ".join(f"{n}" for n in numbers))
    preview = draw_numbered(frame, img_pts, numbers)
    imwrite_unicode(OUT_DIR / "detected_corners.jpg", preview)

    origin = ask_origin(preview)
    if origin is None:
        print("已取消")
        return

    h_mat, world, err = solve(img_pts, board_cm, outer_b, bl, br, away, green, origin)
    print(f"左下綠點設為 ({origin[0]:g}, {origin[1]:g})")
    print(f"板子上重投影：平均 {err.mean():.2f} cm，最大 {err.max():.2f} cm")
    report = tape_report(h_mat)
    save_result(frame, h_mat, img_pts, world, err, origin, report)


if __name__ == "__main__":
    main()
