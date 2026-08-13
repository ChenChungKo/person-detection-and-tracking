"""YOLO26 + BoT-SORT short tracks, then Stable-ID (OSNet-AIN gallery).

BoT-SORT keeps boxes on people but mints a new raw ID after every break.
Stable-ID remaps those raw IDs to session-stable numbers (ID1, ID2, …).

  python track_preview.py --source "rtsp://user:pass@ip:554/stream1"
  python track_preview.py --stride 1
  python track_preview.py --no-stable-id
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import cv2

from grid_occupancy import id_bgr_color

ROOT = Path(__file__).resolve().parent
DEFAULT_TRACKER = ROOT / "trackers" / "botsort.yaml"
DEFAULT_GALLERY = ROOT / "test" / "reid_gallery"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BoT-SORT short tracks + Stable-ID long IDs"
    )
    p.add_argument(
        "--source",
        default="rtsp://oriongo:123456789@192.168.0.200:554/stream1",
        help="RTSP URL or video path",
    )
    p.add_argument("--model", default="yolo26s.pt", help="YOLO weights")
    p.add_argument(
        "--tracker",
        default=str(DEFAULT_TRACKER),
        help="Ultralytics tracker yaml (default: trackers/botsort.yaml)",
    )
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="submit YOLO.track every N display frames (official persist wants 1)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-width", type=int, default=1280, help="preview window width")
    p.add_argument(
        "--all-classes",
        action="store_true",
        help="track every class (implies --no-stable-id)",
    )
    p.add_argument(
        "--no-stable-id",
        action="store_true",
        help="show BoT-SORT raw IDs only (no OSNet rematch)",
    )
    p.add_argument(
        "--show-raw",
        action="store_true",
        help="also print BoT-SORT raw ID next to the stable ID",
    )
    p.add_argument(
        "--reid-model",
        default="osnet_ain",
        help="Stable-ID appearance model (osnet_ain / yolo26n-reid.onnx)",
    )
    p.add_argument(
        "--no-reid",
        action="store_true",
        help="HSV clothing histogram instead of OSNet",
    )
    p.add_argument("--appear-thresh", type=float, default=0.34)
    p.add_argument("--min-hits", type=int, default=16)
    p.add_argument("--id-coast", type=int, default=8)
    p.add_argument("--id-sticky", type=int, default=100)
    p.add_argument(
        "--calib",
        default=str(ROOT / "calibration" / "homography.json"),
        help="homography for floor rematch (optional)",
    )
    p.add_argument(
        "--gallery-dir",
        default=str(DEFAULT_GALLERY),
        help="appearance crop dump (cleared each run)",
    )
    p.add_argument(
        "--no-gallery-dump",
        action="store_true",
        help="do not save appearance gallery crops",
    )
    return p.parse_args()


def resize_preview(img, max_width: int):
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0
    scale = max_width / float(w)
    out = cv2.resize(
        img, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA
    )
    return out, scale


def _put_id_label(vis, x1: int, y1: int, text: str, bg: tuple[int, int, int]) -> None:
    """Same camera-box label as detect_grid: just ID1, black text on color fill."""
    fs = max(0.75, vis.shape[1] / 1280.0)
    scale = 0.9 * fs
    thickness = max(2, int(round(2 * fs)))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 6
    ly = y1 - 12
    if ly < int(40 * fs):
        ly = y1 + int(36 * fs)
    rx1 = max(0, x1 - pad)
    ry1 = max(0, ly - th - pad)
    rx2 = min(vis.shape[1] - 1, x1 + tw + pad)
    ry2 = min(vis.shape[0] - 1, ly + baseline + pad)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), bg, -1)
    cv2.putText(
        vis,
        text,
        (x1, ly),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def person_dets_from_result(result, conf_thres: float) -> list[dict]:
    """All YOLO person boxes (no detect_grid geometry gate — seated people stay)."""
    out: list[dict] = []
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return out
    has_ids = boxes.id is not None
    for i in range(len(boxes)):
        if int(boxes.cls[i].item()) != 0:
            continue
        conf = float(boxes.conf[i].item())
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        tid = None
        if has_ids:
            raw = float(boxes.id[i].item())
            if raw == raw:  # not NaN
                tid = int(raw)
        out.append(
            {
                "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                "foot": (0.5 * (x1 + x2), float(y2)),
                "conf": conf,
                "track_id": tid,
            }
        )
    return out


def draw_id_boxes(frame, dets: list[dict], max_width: int):
    vis, scale = resize_preview(frame, max_width)
    thick = max(2, int(round(2 * max(0.75, vis.shape[1] / 1280.0))))
    for det in dets:
        tid = det.get("track_id")
        if tid is None:
            continue
        sid = int(tid)
        x1, y1, x2, y2 = det["xyxy"]
        sx1, sy1, sx2, sy2 = [int(round(v * scale)) for v in (x1, y1, x2, y2)]
        color = id_bgr_color(sid)
        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), color, thick)
        _put_id_label(vis, sx1, sy1, f"ID{sid}", color)
    return vis


def _add_world(dets: list[dict], h_mat) -> None:
    if h_mat is None:
        return
    from detect_grid import image_to_world

    for det in dets:
        fx, fy = det["foot"]
        det["world"] = image_to_world(h_mat, fx, fy)


def _format_ids(ids) -> str:
    ids = list(ids)
    return ",".join(f"ID{i}" for i in ids) if ids else "—"


def _format_map(stable_dets: list[dict]) -> str:
    parts = []
    for d in sorted(stable_dets, key=lambda x: int(x.get("track_id") or 0)):
        sid = d.get("track_id")
        raw = d.get("raw_track_id")
        if sid is None:
            continue
        if raw is not None:
            parts.append(f"r{int(raw)}→ID{int(sid)}")
        else:
            parts.append(f"ID{int(sid)}")
    return " ".join(parts) if parts else "—"


def _load_mapper(args) -> tuple[object, object | None]:
    from reid_encoder import PersonReIDEncoder, resolve_reid_model
    from stable_id import StableIdMapper

    encoder = None
    if not args.no_reid:
        model_name = resolve_reid_model(args.reid_model)
        print(f"載入 Re-ID：{model_name} …")
        encoder = PersonReIDEncoder(model_name)
        print(
            f"Re-ID 就緒：{encoder.model_name} "
            f"[backend={encoder.backend}]（Stable-ID 外貌圖庫）。"
        )
    mapper = StableIdMapper(
        appear_thresh=args.appear_thresh,
        fps=20.0,
        min_hits=args.min_hits,
        encoder=encoder,
        coast_frames=max(int(args.id_coast), int(args.stride)),
        sticky_frames=args.id_sticky,
        max_prototypes=0,
        gallery_dir=None if args.no_gallery_dump else args.gallery_dir,
    )
    return mapper, encoder


def _maybe_log_ids(
    mapper,
    dets: list[dict],
    raw_n: int,
    frame_idx: int,
    last_id_key: tuple[int, ...] | None,
    last_map: str | None,
) -> tuple[tuple[int, ...] | None, str | None]:
    on_screen = tuple(
        sorted(int(d["track_id"]) for d in dets if d.get("track_id") is not None)
    )
    sticky = max(8, int(20 * 1.0))
    session_alive = tuple(
        sorted(
            sid
            for sid, meta in mapper._stable.items()
            if frame_idx - int(meta["frame"]) <= sticky
        )
    )
    map_s = _format_map(dets)
    if on_screen != last_id_key or map_s != last_map:
        n_pending = max(0, raw_n - len(dets))
        print(
            f"[ID] f{frame_idx:05d}  on {_format_ids(on_screen)}  "
            f"alive {_format_ids(session_alive)}  "
            f"yolo={raw_n} id={len(on_screen)} pending={n_pending}  {map_s}",
            flush=True,
        )
        return on_screen, map_s
    return last_id_key, last_map


class DetectWorker:
    """Run YOLO + Stable-ID off the display thread so the preview does not hitch."""

    def __init__(
        self,
        model,
        track_kw: dict,
        conf: float,
        use_stable: bool,
        mapper,
        h_mat,
    ) -> None:
        self._model = model
        self._track_kw = track_kw
        self._conf = conf
        self._use_stable = use_stable
        self._mapper = mapper
        self._h_mat = h_mat
        self._lock = threading.Lock()
        self._pending: tuple[object, int] | None = None
        self._dets: list[dict] = []
        self._detect_idx = 0
        self._id_key: tuple[int, ...] | None = None
        self._id_map: str | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="yolo-track", daemon=True)
        self._thread.start()

    def submit(self, frame, frame_idx: int) -> None:
        with self._lock:
            if self._pending is not None:
                return
            self._pending = (frame.copy(), frame_idx)

    def snapshot(self) -> tuple[list[dict], int]:
        with self._lock:
            return [dict(d) for d in self._dets], self._detect_idx

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None:
                time.sleep(0.003)
                continue
            frame, frame_idx = job
            result = self._model.track(frame, **self._track_kw)[0]
            raw_dets = person_dets_from_result(result, self._conf)
            if self._use_stable:
                _add_world(raw_dets, self._h_mat)
                dets = self._mapper.apply(raw_dets, frame_idx, frame=frame)
                self._id_key, self._id_map = _maybe_log_ids(
                    self._mapper,
                    dets,
                    len(raw_dets),
                    frame_idx,
                    self._id_key,
                    self._id_map,
                )
            else:
                dets = raw_dets
            with self._lock:
                self._dets = dets
                self._detect_idx = frame_idx



def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride 必須 >= 1")
    is_rtsp = str(args.source).startswith("rtsp://")
    if is_rtsp:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    use_stable = not args.no_stable_id and not args.all_classes
    if args.all_classes and not args.no_stable_id:
        print("提示：--all-classes 只顯示 BoT-SORT raw ID（Stable-ID 只處理 person）。")

    from ultralytics import YOLO

    from detect_grid import DetectionCoaster, load_homography, open_capture

    h_mat = None
    calib = Path(args.calib)
    if use_stable and calib.exists():
        h_mat = load_homography(calib)
        print(f"Homography：{calib.name}（地板距離協助 ID 接回）")
    elif use_stable:
        print(f"找不到 {calib}，Stable-ID 只靠外貌、不用地板距離。")

    mapper = None
    if use_stable:
        mapper, _enc = _load_mapper(args)
        if not args.no_gallery_dump:
            print(f"外貌圖庫：{args.gallery_dir}（每次重跑清空）")
        print(
            f"ID 穩定層：YOLO+BoT-SORT raw ID → OSNet 圖庫接回；"
            f"appear≥{args.appear_thresh:.2f}，新 ID 需連續 {args.min_hits} 次對不上才發號。"
        )
        print("log：on=這一幀畫面上的 ID；alive=最近 1 秒內還在的 ID（閃一下不算消失）。")

    model = YOLO(args.model)
    track_kw = dict(
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=False,
    )
    if not args.all_classes:
        track_kw["classes"] = [0]

    print(
        f"短期追蹤：{args.model} + {Path(args.tracker).name}"
        f"{'' if args.all_classes else '，只抓 person'}"
        f"{' + Stable-ID' if use_stable else '（raw ID）'}"
        f"，stride={args.stride}"
    )
    if args.stride > 1:
        if is_rtsp:
            print(
                f"跳幀：約每 {args.stride / 20.0:.2f}s 跑一次 YOLO／BoT-SORT"
                f"（{20 / args.stride:.0f} 次/秒，相機當 20fps）；"
                "中間畫面沿用並預測框（要每幀辨識請加 --stride 1）。"
            )
        else:
            print(
                f"跳幀：每 {args.stride} 幀才跑 YOLO／BoT-SORT，"
                "中間幀沿用並預測框位置（要每幀辨識請加 --stride 1）。"
            )
    if is_rtsp:
        print("RTSP：只吃最新幀（避免 Waiting for stream / 積幀讓 BoT-SORT 斷 ID）。")
    print("畫面與 YOLO 分開執行緒：預覽照 fps 更新，辨識不再卡住視窗。")
    print("視窗出現後按 q 結束。")
    win = "BoT-SORT + Stable-ID" if use_stable else "Ultralytics track"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cap = open_capture(args.source)
    if cap is None:
        raise SystemExit(f"無法開啟來源：{args.source}")

    from latest_frame import LatestFrameCapture

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 1e-3:
        video_fps = 20.0
    if not is_rtsp:
        print(f"本機影片：依 {video_fps:.0f} fps 即時播放（推論慢會丟幀，畫面不卡）。")

    reader = LatestFrameCapture(cap) if is_rtsp else None
    if reader is not None:
        for _ in range(50):
            ok, frame = reader.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        else:
            reader.release()
            raise SystemExit("RTSP 連線後未收到畫面。")

    worker = DetectWorker(
        model,
        track_kw,
        args.conf,
        use_stable,
        mapper,
        h_mat,
    )
    coaster = DetectionCoaster()
    last_obs_idx = -1
    frame_idx = 0
    last_detect_t = -1.0
    t_play0 = time.perf_counter()
    try:
        while True:
            if reader is not None:
                ok, frame = reader.read()
                if not ok or frame is None:
                    if not reader.is_alive():
                        print("讀取結束或失敗。")
                        break
                    time.sleep(0.01)
                    continue
                frame_idx += 1
            else:
                target = int((time.perf_counter() - t_play0) * video_fps) + 1
                while frame_idx + 1 < target:
                    if not cap.grab():
                        ok, frame = False, None
                        break
                    frame_idx += 1
                else:
                    ok, frame = cap.read()
                    if ok:
                        frame_idx += 1
                if not ok or frame is None:
                    print("讀取結束或失敗。")
                    break
            if is_rtsp and args.stride > 1:
                now = time.perf_counter()
                run_detect = last_detect_t < 0 or (now - last_detect_t) >= (
                    args.stride / 20.0
                )
            else:
                run_detect = frame_idx == 1 or (frame_idx - 1) % args.stride == 0
            if run_detect:
                last_detect_t = time.perf_counter()
                worker.submit(frame, frame_idx)
            dets, det_idx = worker.snapshot()
            if det_idx != last_obs_idx:
                coaster.observe(dets, det_idx)
                last_obs_idx = det_idx
            draw_dets = coaster.extrapolate(dets, frame_idx) if dets else dets
            vis = draw_id_boxes(frame, draw_dets, args.max_width)
            cv2.imshow(win, vis)
            if is_rtsp:
                wait_ms = 1
            else:
                ahead = frame_idx / video_fps - (time.perf_counter() - t_play0)
                wait_ms = max(1, int(ahead * 1000)) if ahead > 0.005 else 1
            if (cv2.waitKey(wait_ms) & 0xFF) == ord("q"):
                break
    finally:
        worker.stop()
        if reader is not None:
            reader.release()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
