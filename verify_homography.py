"""Verify Homography: click image -> floor (X, Y) cm, optionally measure real error.

Usage:
  # Just see predicted world coords
  python verify_homography.py
  python verify_homography.py --calib calibration/homography_v2_chessboard.json

  # Click points, then enter tape-measured ground truth to get real error
  python verify_homography.py --measure-error
  python verify_homography.py --measure-error --image calibration/chessboard_floor/capture.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"
DEFAULT_FALLBACK_IMAGE = Path(__file__).resolve().parent / "test" / "static_frame.jpg"
DEFAULT_ERROR_OUT = Path(__file__).resolve().parent / "calibration" / "homography_error_report.json"


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_for_preview(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame.copy(), 1.0
    scale = max_width / float(w)
    view = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return view, scale


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify Homography by clicking / measure real error")
    p.add_argument("--calib", default=str(DEFAULT_CALIB), help="homography.json path")
    p.add_argument("--image", default="", help="Override image path (optional)")
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument(
        "--measure-error",
        action="store_true",
        help="after each click, ask for tape-measured X Y and report error (cm)",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_ERROR_OUT),
        help="save error report JSON when using --measure-error (empty to skip)",
    )
    return p.parse_args()


def ask_truth() -> tuple[float, float] | None:
    raw = input("  真實世界座標 X Y（cm，Enter 跳過此點，q 結束量測）: ").strip()
    if not raw:
        return None
    if raw.lower() in {"q", "quit"}:
        raise KeyboardInterrupt
    parts = raw.replace(",", " ").split()
    if len(parts) != 2:
        print("  格式錯誤，請輸入兩個數字，例如：215 360")
        return ask_truth()
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        print("  數字格式錯誤，請再試一次。")
        return ask_truth()


def main() -> None:
    args = parse_args()
    calib_path = Path(args.calib)
    if not calib_path.exists():
        raise SystemExit(f"找不到校正檔：{calib_path}")

    payload = json.loads(calib_path.read_text(encoding="utf-8"))
    h_mat = np.array(payload["homography"], dtype=np.float64)
    roi = payload.get("roi") or {}

    if args.image:
        image_path = Path(args.image)
    else:
        image_path = Path(payload.get("image_path", ""))
        if not image_path.exists():
            # Prefer chessboard capture if present (same scene as v2).
            chess = Path(__file__).resolve().parent / "calibration" / "chessboard_floor" / "capture.jpg"
            if chess.exists():
                image_path = chess
            elif DEFAULT_FALLBACK_IMAGE.exists():
                image_path = DEFAULT_FALLBACK_IMAGE
            else:
                raise SystemExit(f"找不到影像：{payload.get('image_path')}")

    image = imread_unicode(image_path)
    if image is None:
        raise SystemExit(f"無法讀取影像：{image_path}")

    view, scale = resize_for_preview(image, args.max_width)
    # (image_xy, pred_world, truth_world|None, err_cm|None)
    samples: list[dict] = []
    pending: dict | None = None

    win = "Verify Homography" + (" [measure error]" if args.measure_error else "")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param) -> None:  # noqa: ARG001
        nonlocal pending
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if pending is not None:
            print("  請先在終端機輸入上一個點的真實座標（或按 Enter 跳過）。")
            return
        full_x = x / scale
        full_y = y / scale
        pts = np.array([[[full_x, full_y]]], dtype=np.float64)
        world = cv2.perspectiveTransform(pts, h_mat)[0, 0]
        wx, wy = float(world[0]), float(world[1])
        note = ""
        if roi:
            xmin = float(roi.get("x_min_cm", 0))
            ymin = float(roi.get("y_min_cm", 0))
            xmax = float(roi.get("x_max_cm", 0))
            ymax = float(roi.get("y_max_cm", 0))
            if xmax > xmin and ymax > ymin and not (xmin <= wx <= xmax and ymin <= wy <= ymax):
                note = "  [超出有效 ROI]"

        sample = {
            "image_xy": [full_x, full_y],
            "pred_world_xy": [wx, wy],
            "truth_world_xy": None,
            "error_cm": None,
            "error_xy_cm": None,
        }
        if args.measure_error:
            pending = sample
            print(f"點擊 image=({full_x:.1f}, {full_y:.1f}) -> 預測 world=({wx:.1f}, {wy:.1f}) cm{note}")
        else:
            samples.append(sample)
            print(f"image=({full_x:.1f}, {full_y:.1f}) -> world=({wx:.1f}, {wy:.1f}) cm{note}")

    cv2.setMouseCallback(win, on_mouse)
    print(f"校正檔：{calib_path}")
    print(f"影像：{image_path}")
    print("請只點地板（磁磚角／已知標點）。牆上/螢幕上的點會算出不合理座標。")
    if roi:
        print(
            f"有效 ROI：X {roi.get('x_min_cm')}~{roi.get('x_max_cm')} cm, "
            f"Y {roi.get('y_min_cm')}~{roi.get('y_max_cm')} cm"
        )
    if args.measure_error:
        print("模式：點擊量測誤差")
        print("  1) 在畫面上點一個你量過的地板點")
        print("  2) 在終端機輸入卷尺量到的 X Y（cm）")
        print("  3) 程式會顯示 |預測−真實| 誤差；u 撤銷上一筆；q 結束並彙總")
    else:
        print("左鍵點地板查看平面座標；加 --measure-error 可量真實誤差；q 離開。")

    try:
        while True:
            canvas = view.copy()
            for s in samples[-30:]:
                ix, iy = s["image_xy"]
                wx, wy = s["pred_world_xy"]
                px, py = int(round(ix * scale)), int(round(iy * scale))
                color = (0, 255, 0) if s["error_cm"] is not None else (255, 0, 0)
                cv2.circle(canvas, (px, py), 7, color, -1, lineType=cv2.LINE_AA)
                if s["error_cm"] is not None:
                    label = f"e={s['error_cm']:.0f}cm"
                else:
                    label = f"({wx:.0f},{wy:.0f})"
                cv2.putText(
                    canvas,
                    label,
                    (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            tip = "click known floor point  [u]ndo  [q]uit" if args.measure_error else "click floor  [q]uit"
            if pending is not None:
                tip = "enter truth X Y in terminal..."
            cv2.putText(canvas, tip, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            measured = [s for s in samples if s["error_cm"] is not None]
            if measured:
                mean_e = float(np.mean([s["error_cm"] for s in measured]))
                max_e = float(np.max([s["error_cm"] for s in measured]))
                cv2.putText(
                    canvas,
                    f"n={len(measured)}  mean={mean_e:.1f}cm  max={max_e:.1f}cm",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(win, canvas)

            if pending is not None and args.measure_error:
                cv2.waitKey(1)
                try:
                    truth = ask_truth()
                except KeyboardInterrupt:
                    pending = None
                    break
                if truth is None:
                    print("  已跳過此點。")
                    pending = None
                else:
                    tx, ty = truth
                    pxw, pyw = pending["pred_world_xy"]
                    dx, dy = pxw - tx, pyw - ty
                    err = float(np.hypot(dx, dy))
                    pending["truth_world_xy"] = [tx, ty]
                    pending["error_cm"] = err
                    pending["error_xy_cm"] = [dx, dy]
                    samples.append(pending)
                    print(
                        f"  真實=({tx:.1f},{ty:.1f})  預測=({pxw:.1f},{pyw:.1f})  "
                        f"誤差={err:.1f} cm  (dX={dx:+.1f}, dY={dy:+.1f})"
                    )
                    pending = None
                continue

            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and samples and pending is None:
                removed = samples.pop()
                print(f"撤銷上一點：pred={removed['pred_world_xy']} err={removed['error_cm']}")
    finally:
        cv2.destroyAllWindows()

    measured = [s for s in samples if s["error_cm"] is not None]
    if measured:
        errs = np.array([s["error_cm"] for s in measured], dtype=np.float64)
        print()
        print(f"量測點數：{len(measured)}")
        print(f"平均誤差：{float(errs.mean()):.2f} cm")
        print(f"最大誤差：{float(errs.max()):.2f} cm")
        print(f"中位數：{float(np.median(errs)):.2f} cm")
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "calib": str(calib_path),
                "image": str(image_path),
                "num_points": len(measured),
                "mean_error_cm": float(errs.mean()),
                "max_error_cm": float(errs.max()),
                "median_error_cm": float(np.median(errs)),
                "samples": measured,
                "note": "Real localization error = |H(click) - tape-measured world|. "
                "This is NOT the chessboard fit residual (mean_reproj_error_cm).",
            }
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已存報告：{out_path}")
    elif args.measure_error:
        print("沒有完成任何誤差量測點。")


if __name__ == "__main__":
    main()
