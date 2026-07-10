"""Test real-time RTSP capture with OpenCV.

Usage (in VS Code terminal with .venv active):
  python test_rtsp.py "rtsp://user:pass@192.168.0.200:554/stream1"

Press q to quit.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV RTSP live preview")
    parser.add_argument(
        "url",
        nargs="?",
        default=os.getenv("RTSP_URL", ""),
        help="RTSP URL, or set env RTSP_URL",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Only grab frames and print stats (no window)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="With --no-preview, stop after N frames (0 = run until Ctrl+C)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Preview max width (full frame is still captured; default 1280)",
    )
    return parser.parse_args()


def open_capture(url: str) -> cv2.VideoCapture:
    # Prefer TCP; UDP often drops frames on Wi-Fi / Tapo cameras.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def resize_for_preview(frame, max_width: int):
    """Scale down for display only; keeps full FOV (not a crop)."""
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame
    scale = max_width / float(w)
    new_size = (max_width, int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    if not args.url:
        raise SystemExit(
            "請提供 RTSP URL，例如：\n"
            '  python test_rtsp.py "rtsp://user:pass@192.168.0.200:554/stream1"\n'
            "或先設定環境變數 RTSP_URL。"
        )

    cap = open_capture(args.url)
    if not cap.isOpened():
        raise SystemExit("無法開啟 RTSP 串流，請確認網址、帳密與攝影機是否在線。")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"已連線：{width}x{height}")
    if not args.no_preview:
        print(f"預覽會縮放到寬度 <= {args.max_width}（完整畫面，非裁切）")
        print("按 q 結束")
        cv2.namedWindow("RTSP OpenCV", cv2.WINDOW_NORMAL)
    else:
        print("無預覽模式，Ctrl+C 結束")

    frame_count = 0
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("讀取失敗，嘗試重連…")
                cap.release()
                time.sleep(1)
                cap = open_capture(args.url)
                if not cap.isOpened():
                    raise SystemExit("重連失敗。")
                continue

            frame_count += 1
            elapsed = time.perf_counter() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            if args.no_preview:
                if frame_count == 1:
                    print(f"第一幀 shape={frame.shape}")
                if frame_count % 30 == 0:
                    print(f"frames={frame_count}  fps~{fps:.1f}")
                if args.frames and frame_count >= args.frames:
                    print(f"完成：抓取 {frame_count} 幀，平均 fps~{fps:.1f}")
                    break
            else:
                view = resize_for_preview(frame, args.max_width)
                cv2.putText(
                    view,
                    f"FPS {fps:.1f} | src {width}x{height}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("RTSP OpenCV", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
