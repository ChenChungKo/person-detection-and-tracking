"""One-shot floor Homography from the printed ChArUco board.

Does not replace calibration/homography.json.
Bottom-left paper corner (near the camera, image left) is the tape point
(170, 405) cm. Short side runs to +X, long side runs to decreasing Y.
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

SQUARE_CM = 20.0
BOARD_W = 100.0
BOARD_H = 80.0
ORIGIN_XY = (170.0, 405.0)


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
    return frame


def main() -> None:
    frame = grab_rtsp()
    imwrite_unicode(OUT_DIR / "capture.jpg", frame)
    print(f"畫面 {frame.shape[1]}x{frame.shape[0]}")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 4), SQUARE_CM / 100.0, 14.0 / 100.0, dictionary)
    corners3d = board.getChessboardCorners().reshape(-1, 3)
    params = cv2.aruco.CharucoParameters()
    params.checkMarkers = False
    params.minMarkers = 1
    detector = cv2.aruco.CharucoDetector(board, params)
    cc, ids, _mc, mids = detector.detectBoard(frame)
    n_m = 0 if mids is None else len(mids)
    n_c = 0 if ids is None else len(ids)
    print(f"標記 {n_m}/10，角點 {n_c}/12")
    if ids is None or n_c < 6:
        raise SystemExit("角點太少，先把紙攤平、確認整塊入鏡")

    ids = ids.reshape(-1)
    img_pts = cc.reshape(-1, 2).astype(np.float64)
    board_cm = np.array([corners3d[int(i), :2] * 100.0 for i in ids], dtype=np.float64)

    aff, _ = cv2.estimateAffine2D(board_cm.reshape(-1, 1, 2), img_pts.reshape(-1, 1, 2))
    if aff is None:
        raise SystemExit("無法對齊板子座標")

    def board_to_px(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return (aff[:, :2] @ pts.T).T + aff[:, 2]

    outer_b = np.array([[0, 0], [BOARD_W, 0], [BOARD_W, BOARD_H], [0, BOARD_H]], dtype=np.float64)
    outer_px = board_to_px(outer_b)
    near = np.argsort(-outer_px[:, 1])[:2]
    bl = int(near[np.argmin(outer_px[near, 0])])
    br = int(near[np.argmax(outer_px[near, 0])])
    far = [i for i in range(4) if i not in (bl, br)]
    away = int(far[np.argmin(outer_px[np.array(far), 0])])
    print(f"左下紙角 board_cm {outer_b[bl].tolist()}  px {outer_px[bl].round(1).tolist()}")
    print(f"右側紙角 board_cm {outer_b[br].tolist()}  遠離鏡頭左側 {outer_b[away].tolist()}")

    # Axis-aligned: +X along the near edge, -Y away from the camera.
    rb = outer_b[br] - outer_b[bl]
    ab = outer_b[away] - outer_b[bl]
    near_cm = float(np.linalg.norm(rb))
    away_cm = float(np.linalg.norm(ab))
    print(f"近端邊長 {near_cm:.0f} cm，縱深邊長 {away_cm:.0f} cm")
    basis = np.column_stack([rb, ab])
    world_axes = np.array([[near_cm, 0.0], [0.0, -away_cm]], dtype=np.float64).T
    # world_delta = M @ board_delta, M @ rb = (near_cm, 0), M @ ab = (0, -away_cm)
    mapping = world_axes @ np.linalg.inv(basis)
    delta = board_cm - outer_b[bl]
    world = (mapping @ delta.T).T + np.array(ORIGIN_XY)

    h_mat, _ = cv2.findHomography(img_pts, world, method=0)
    if h_mat is None:
        raise SystemExit("Homography 失敗")
    projected = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
    err = np.linalg.norm(projected - world, axis=1)
    print(f"板子上重投影：平均 {err.mean():.2f} cm，最大 {err.max():.2f} cm")

    old = np.array(json.loads(OLD_JSON.read_text(encoding="utf-8"))["homography"], dtype=np.float64)
    tape = json.loads(TAPE_JSON.read_text(encoding="utf-8"))["samples"]
    print("捲尺點  舊棋盤誤差  這次 ChArUco")
    old_errs = []
    new_errs = []
    for s in tape:
        ip = np.array(s["image_xy"], dtype=np.float64).reshape(1, 1, 2)
        truth = np.array(s["truth_world_xy"], dtype=np.float64)
        pred_old = cv2.perspectiveTransform(ip, old).reshape(2)
        pred_new = cv2.perspectiveTransform(ip, h_mat).reshape(2)
        eo = float(np.linalg.norm(pred_old - truth))
        en = float(np.linalg.norm(pred_new - truth))
        old_errs.append(eo)
        new_errs.append(en)
        print(
            f"  真值 {truth[0]:.0f},{truth[1]:.0f}  "
            f"舊 {eo:.1f} cm  新 {en:.1f} cm  "
            f"新預測 {pred_new[0]:.1f},{pred_new[1]:.1f}"
        )
    print(f"四點平均：舊 {np.mean(old_errs):.1f} cm，新 {np.mean(new_errs):.1f} cm")

    dbg = frame.copy()
    try:
        cv2.aruco.drawDetectedCornersCharuco(dbg, cc, ids.reshape(-1, 1))
    except cv2.error:
        for pt in img_pts:
            cv2.circle(dbg, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)
    pbl = tuple(outer_px[bl].astype(int))
    cv2.circle(dbg, pbl, 16, (0, 0, 255), 3)
    cv2.putText(dbg, "BL 170,405", (pbl[0] + 12, pbl[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    imwrite_unicode(OUT_DIR / "detected_corners.jpg", dbg)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_path": str(OUT_DIR / "capture.jpg"),
        "image_size_wh": [int(frame.shape[1]), int(frame.shape[0])],
        "units": "cm",
        "method": "charuco_floor",
        "dictionary": "DICT_4X4_50",
        "squares_xy": [5, 4],
        "square_cm": SQUARE_CM,
        "marker_cm": 14.0,
        "bottom_left_world_xy": list(ORIGIN_XY),
        "axis": "near edge +X, away from camera -Y",
        "homography": h_mat.tolist(),
        "image_points_xy": img_pts.tolist(),
        "world_points_xy": world.tolist(),
        "charuco_ids": ids.astype(int).tolist(),
        "mean_reproj_error_cm": float(err.mean()),
        "max_reproj_error_cm": float(err.max()),
        "tape_mean_error_cm": float(np.mean(new_errs)),
        "note": "Not the live homography. Bottom-left paper corner fixed at the tape point.",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已存：{OUT_JSON}")


if __name__ == "__main__":
    main()
