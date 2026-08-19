"""Settings page for detect_grid.py: pick file or RTSP, tweak flags, then Run."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from app_paths import app_root

ROOT = app_root()
DEFAULT_VIDEO = ROOT / "test" / "test4.mp4"
DEFAULT_RTSP = "rtsp://oriongo:123456789@192.168.0.200:554/stream1"
DEFAULT_ERROR_COMP = ROOT / "calibration" / "homography_error_report.json"


def _fit_contain(bgr: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    """Resize keeping aspect ratio so the whole image fits in the box."""
    h, w = bgr.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _blit_center(dst: np.ndarray, src: np.ndarray) -> None:
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]
    x0 = max(0, (dw - sw) // 2)
    y0 = max(0, (dh - sh) // 2)
    sh = min(sh, dh - y0)
    sw = min(sw, dw - x0)
    dst[y0 : y0 + sh, x0 : x0 + sw] = src[:sh, :sw]


class _GuiWriter(io.TextIOBase):
    def __init__(self, emit) -> None:
        self._emit = emit

    def write(self, s: str) -> int:
        if s:
            self._emit(s)
        return len(s)

    def flush(self) -> None:
        return None


class Launcher(tk.Tk):
    def __init__(
        self,
        record_path: str | None = None,
        auto_run: bool = False,
        max_seconds: float = 0,
    ) -> None:
        super().__init__()
        self.title("detect_grid 啟動")
        self.minsize(900, 720)
        self._stop_ev = threading.Event()
        self._pause_ev = threading.Event()
        self._worker: threading.Thread | None = None
        self._vis: np.ndarray | None = None
        self._grid: np.ndarray | None = None
        self._paint_scheduled = False
        self._photo_grid: ImageTk.PhotoImage | None = None
        self._quit_after_run = False

        self.source_kind = tk.StringVar(value="file")
        self.video_path = tk.StringVar(value=str(DEFAULT_VIDEO))
        self.rtsp_url = tk.StringVar(value=DEFAULT_RTSP)
        self.ref = tk.StringVar(value="pose")
        self.reid_model = tk.StringVar(value="osnet_ain")
        self.conf = tk.DoubleVar(value=0.45)
        self.stride = tk.IntVar(value=5)
        self.cell_hold = tk.IntVar(value=2)
        self.min_hits = tk.IntVar(value=16)
        self.out_margin = tk.DoubleVar(value=45.0)
        self.skeleton = tk.BooleanVar(value=True)
        self.track = tk.BooleanVar(value=True)
        self.error_comp = tk.BooleanVar(value=DEFAULT_ERROR_COMP.exists())
        self.floor_grid = tk.BooleanVar(value=True)
        self.quiet = tk.BooleanVar(value=True)
        self.review_dump = tk.BooleanVar(value=False)
        self.cmd_preview = tk.StringVar(value="")
        self.status = tk.StringVar(value="")

        self._build()
        self._bind_preview()
        self._on_kind()
        self._refresh_cmd()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._record_path = record_path
        self._record_writer: cv2.VideoWriter | None = None
        self._record_size: tuple[int, int] | None = None
        if record_path or auto_run:
            self.geometry("1400x900")
        if record_path:
            self.after(300, self._record_tick)
        if auto_run:
            self.after(900, self._run)
        if max_seconds > 0:
            self.after(int(900 + max_seconds * 1000), self._stop)

    def _grab_window_bgr(self) -> np.ndarray | None:
        from PIL import ImageGrab

        self.update_idletasks()
        x = int(self.winfo_rootx())
        y = int(self.winfo_rooty())
        w = int(self.winfo_width())
        h = int(self.winfo_height())
        if w < 32 or h < 32:
            return None
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        rgb = np.array(img)
        if rgb.size == 0:
            return None
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _record_tick(self) -> None:
        if not self._record_path:
            return
        frame = self._grab_window_bgr()
        if frame is not None:
            h, w = frame.shape[:2]
            if self._record_writer is None:
                path = Path(self._record_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._record_size = (w, h)
                self._record_writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10.0,
                    (w, h),
                )
            elif (w, h) != self._record_size:
                frame = cv2.resize(frame, self._record_size)
            if self._record_writer is not None:
                self._record_writer.write(frame)
        self.after(100, self._record_tick)

    def _stop_recording(self) -> None:
        if self._record_writer is not None:
            self._record_writer.release()
            self._record_writer = None
            self._record_path = ""

    def _set_hint(self, text: str | None) -> None:
        self.hint_var.set(text or "")

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 4}
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="來源", font=("", 11, "bold")).grid(row=0, column=0, sticky="w")
        kind = ttk.Frame(frm)
        kind.grid(row=1, column=0, columnspan=3, sticky="w", **pad)
        ttk.Radiobutton(
            kind, text="本機影片", variable=self.source_kind, value="file", command=self._on_kind
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Radiobutton(
            kind, text="RTSP", variable=self.source_kind, value="rtsp", command=self._on_kind
        ).pack(side=tk.LEFT)

        self.video_row = ttk.Frame(frm)
        self.video_row.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(self.video_row, text="影片").pack(side=tk.LEFT)
        ttk.Entry(self.video_row, textvariable=self.video_path, width=62).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(self.video_row, text="瀏覽…", command=self._browse).pack(side=tk.LEFT)

        self.rtsp_row = ttk.Frame(frm)
        self.rtsp_row.grid(row=3, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(self.rtsp_row, text="網址").pack(side=tk.LEFT)
        ttk.Entry(self.rtsp_row, textvariable=self.rtsp_url, width=70).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )

        opts = ttk.Frame(frm)
        opts.grid(row=4, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(opts, text="腳點").pack(side=tk.LEFT)
        ttk.Combobox(
            opts, textvariable=self.ref, values=("pose", "auto", "foot", "head_drop"), width=12, state="readonly"
        ).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Checkbutton(opts, text="骨架", variable=self.skeleton).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(opts, text="追蹤 / ID", variable=self.track).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(opts, text="四點補償", variable=self.error_comp).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(opts, text="A/B/C/O", variable=self.floor_grid).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(opts, text="少印 log", variable=self.quiet).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(opts, text="審查裁圖", variable=self.review_dump).pack(side=tk.LEFT, padx=6)

        reid = ttk.Frame(frm)
        reid.grid(row=5, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(reid, text="Re-ID").pack(side=tk.LEFT)
        ttk.Combobox(
            reid,
            textvariable=self.reid_model,
            values=("osnet_ain", "osnet", "none"),
            width=14,
            state="readonly",
        ).pack(side=tk.LEFT, padx=6)

        sliders = ttk.LabelFrame(frm, text="參數", padding=8)
        sliders.grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        frm.columnconfigure(0, weight=1)
        sliders.columnconfigure(1, weight=1)
        self._slider(sliders, 0, "conf", self.conf, 0.10, 0.90, 0.01)
        self._slider(sliders, 1, "stride", self.stride, 1, 15, 1)
        self._slider(sliders, 2, "cell-hold", self.cell_hold, 1, 8, 1)
        self._slider(sliders, 3, "min-hits", self.min_hits, 1, 40, 1)
        self._slider(sliders, 4, "out-margin (cm)", self.out_margin, 0, 90, 1)

        ttk.Label(frm, text="將執行").grid(row=7, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.cmd_preview, state="readonly").grid(
            row=8, column=0, columnspan=3, sticky="ew", **pad
        )

        btns = ttk.Frame(frm)
        btns.grid(row=9, column=0, columnspan=3, sticky="w", pady=8)
        self.run_btn = ttk.Button(btns, text="Run", command=self._run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn = ttk.Button(btns, text="停止", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.pause_btn = ttk.Button(btns, text="暫停", command=self._toggle_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT)
        ttk.Label(btns, text="本機影片可暫停；按停止結束").pack(side=tk.LEFT, padx=16)

        ttk.Label(frm, textvariable=self.status, foreground="#444").grid(
            row=10, column=0, columnspan=3, sticky="w", **pad
        )

        self.hint_var = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.hint_var, foreground="#555").grid(
            row=11, column=0, columnspan=3, sticky="w", **pad
        )

        views = ttk.Panedwindow(frm, orient=tk.HORIZONTAL)
        views.grid(row=12, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        frm.rowconfigure(12, weight=1)

        left = ttk.Frame(views)
        right = ttk.Frame(views)
        views.add(left, weight=2)
        views.add(right, weight=3)
        self.views = views

        ttk.Label(left, text="Grid  格子", font=("", 10, "bold")).pack(anchor="w", padx=4, pady=(0, 2))
        ttk.Label(right, text="Video  監視器", font=("", 10, "bold")).pack(anchor="w", padx=4, pady=(0, 2))
        self.grid_canvas = tk.Canvas(left, background="#f7f7f7", highlightthickness=0)
        self.video_canvas = tk.Canvas(right, background="#f7f7f7", highlightthickness=0)
        self.grid_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.grid_canvas.bind("<Configure>", lambda _e: self._paint())
        self.video_canvas.bind("<Configure>", lambda _e: self._paint())
        self._set_hint("按 Run 後：左格子、右監視器")

    def _slider(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        var: tk.Variable,
        lo: float,
        hi: float,
        res: float,
    ) -> None:
        ttk.Label(parent, text=label, width=16).grid(row=row, column=0, sticky="w")
        ttk.Scale(parent, from_=lo, to=hi, variable=var, command=lambda _v: self._refresh_cmd()).grid(
            row=row, column=1, sticky="ew", padx=8
        )
        is_int = isinstance(var, tk.IntVar) or res >= 1
        spin = ttk.Spinbox(
            parent,
            from_=lo,
            to=hi,
            increment=res,
            width=8,
            textvariable=var,
            command=self._refresh_cmd,
        )
        spin.grid(row=row, column=2, sticky="e")

        def _show(*_a: object) -> None:
            try:
                n = float(var.get())
            except (tk.TclError, ValueError):
                return
            n = min(max(n, lo), hi)
            if is_int:
                rounded = int(round(n))
                if int(var.get()) != rounded:
                    var.set(rounded)
            else:
                rounded_f = round(n, 2)
                if abs(float(var.get()) - rounded_f) > 1e-9:
                    var.set(rounded_f)
            self._refresh_cmd()

        def _commit(_event: object | None = None) -> None:
            try:
                raw = float(spin.get())
            except (tk.TclError, ValueError):
                _show()
                return
            raw = min(max(raw, lo), hi)
            var.set(int(round(raw)) if is_int else round(raw, 2))
            self._refresh_cmd()

        spin.bind("<Return>", _commit)
        spin.bind("<FocusOut>", _commit)
        var.trace_add("write", _show)
        _show()

    def _bind_preview(self) -> None:
        for var in (
            self.source_kind,
            self.video_path,
            self.rtsp_url,
            self.ref,
            self.reid_model,
            self.skeleton,
            self.track,
            self.error_comp,
            self.floor_grid,
            self.quiet,
            self.review_dump,
        ):
            var.trace_add("write", lambda *_a: self._refresh_cmd())

    def _on_kind(self) -> None:
        file_mode = self.source_kind.get() == "file"
        if file_mode:
            self.video_row.grid()
            self.rtsp_row.grid_remove()
        else:
            self.rtsp_row.grid()
            self.video_row.grid_remove()
        self._set_enabled(self.video_row, file_mode)
        self._set_enabled(self.rtsp_row, not file_mode)
        self._refresh_cmd()

    def _set_enabled(self, widget: tk.Misc, on: bool) -> None:
        state = ["!disabled"] if on else ["disabled"]
        try:
            widget.state(state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_enabled(child, on)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇影片",
            initialdir=str(ROOT / "test"),
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")],
        )
        if path:
            self.video_path.set(path)

    def _source(self) -> str:
        if self.source_kind.get() == "rtsp":
            return self.rtsp_url.get().strip()
        return self.video_path.get().strip()

    def _argv(self) -> list[str]:
        src = self._source()
        cmd = [
            "--source",
            src,
            "--ref",
            self.ref.get(),
            "--conf",
            f"{float(self.conf.get()):.2f}",
            "--stride",
            str(int(round(float(self.stride.get())))),
            "--cell-hold",
            str(int(round(float(self.cell_hold.get())))),
            "--min-hits",
            str(int(round(float(self.min_hits.get())))),
            "--out-margin",
            str(int(round(float(self.out_margin.get())))),
            "--no-show",
        ]
        if self.quiet.get():
            cmd.append("--quiet")
        if self.track.get():
            model = self.reid_model.get()
            if model == "none":
                cmd.append("--no-reid")
            else:
                cmd.extend(["--reid-model", model])
        else:
            cmd.append("--no-track")
        if not self.skeleton.get():
            cmd.append("--no-pose-skeleton")
        if not self.floor_grid.get():
            cmd.append("--no-floor-grid")
        if self.review_dump.get():
            cmd.append("--review-dump")
        if self.error_comp.get():
            cmd.extend(["--error-comp", str(DEFAULT_ERROR_COMP)])
        return cmd

    def _refresh_cmd(self) -> None:
        shown = ["python", "detect_grid.py", *[a for a in self._argv() if a != "--no-show"]]
        self.cmd_preview.set(" ".join(shown))

    def _emit_status(self, text: str) -> None:
        line = text.strip()
        if line:
            self.after(0, self.status.set, line.replace("\n", "  "))

    def _push_frame(self, vis: np.ndarray, grid: np.ndarray) -> None:
        self._vis = vis.copy()
        self._grid = grid.copy()
        self.after(0, self._set_hint, "")
        if not self._paint_scheduled:
            self._paint_scheduled = True
            self.after(0, self._paint)

    def _paint_one(
        self,
        canvas: tk.Canvas,
        bgr: np.ndarray | None,
        attr: str,
    ) -> None:
        canvas.delete("frame")
        if bgr is None:
            return
        cw = max(canvas.winfo_width(), 2)
        ch = max(canvas.winfo_height(), 2)
        fitted = _fit_contain(bgr, cw, ch)
        pad = np.full((ch, cw, 3), 247, dtype=np.uint8)
        _blit_center(pad, fitted)
        rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        setattr(self, attr, photo)
        canvas.create_image(0, 0, anchor="nw", image=photo, tags="frame")

    def _paint(self) -> None:
        self._paint_scheduled = False
        self._paint_one(self.grid_canvas, self._grid, "_photo_grid")
        self._paint_one(self.video_canvas, self._vis, "_photo_video")

    def _run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        src = self._source()
        if not src:
            messagebox.showerror("來源", "請填影片路徑或 RTSP 網址。")
            return
        if self.source_kind.get() == "file" and not Path(src).exists():
            messagebox.showerror("來源", f"找不到檔案：\n{src}")
            return
        if self.source_kind.get() == "rtsp" and not src.lower().startswith("rtsp://"):
            messagebox.showerror("來源", "RTSP 網址要以 rtsp:// 開頭。")
            return
        if self.source_kind.get() == "rtsp" and (
            "帳號" in src or "密碼" in src or "user:pass" in src.lower()
        ):
            messagebox.showerror(
                "來源",
                "請把 RTSP 網址裡的帳號、密碼改成攝影機真實帳密。\n"
                "終端機若出現 401 Unauthorized，就是帳密或網址不對。",
            )
            return
        self._stop_ev.clear()
        self._pause_ev.clear()
        self.pause_btn.configure(text="暫停")
        self.status.set("載入模型中…")
        self._set_hint("載入中，請稍候")
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        if self.source_kind.get() == "file":
            self.pause_btn.configure(state=tk.NORMAL)
        argv = self._argv()
        self._worker = threading.Thread(target=self._worker_loop, args=(argv,), daemon=True)
        self._worker.start()

    def _worker_loop(self, argv: list[str]) -> None:
        import detect_grid

        writer = _GuiWriter(self._emit_status)
        code = 0
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                detect_grid.main(
                    argv,
                    on_preview=self._push_frame,
                    stop_event=self._stop_ev,
                    pause_event=self._pause_ev,
                )
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else 1
            if exc.code not in (0, None):
                self._emit_status(str(exc))
        except Exception:
            code = 1
            self._emit_status(traceback.format_exc())
        self.after(0, self._run_finished, code)

    def _run_finished(self, code: int) -> None:
        if not self.status.get().startswith("結束"):
            self.status.set(f"結束（exit {code}）")
        self._worker = None
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.pause_btn.configure(state=tk.DISABLED, text="暫停")
        self._pause_ev.clear()
        self._stop_recording()
        if getattr(self, "_quit_after_run", False):
            self.after(400, self.destroy)

    def _toggle_pause(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            return
        if self.source_kind.get() != "file":
            return
        if self._pause_ev.is_set():
            self._pause_ev.clear()
            self.pause_btn.configure(text="暫停")
            self.status.set("繼續播放")
        else:
            self._pause_ev.set()
            self.pause_btn.configure(text="繼續")
            self.status.set("已暫停")

    def _stop(self) -> None:
        self._pause_ev.clear()
        self._stop_ev.set()
        self.status.set("正在停止…")

    def _on_close(self) -> None:
        self._stop()
        self._stop_recording()
        self.destroy()


def parse_launcher_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="detect_grid 啟動頁")
    p.add_argument("--auto-run", action="store_true", help="開啟後自動 Run（預設 test4）")
    p.add_argument("--record", default="", help="把整個啟動視窗錄成 mp4")
    p.add_argument("--max-seconds", type=float, default=0, help="自動 Run 後最多跑幾秒（0=整支影片）")
    return p.parse_args(argv)


def main() -> None:
    args = parse_launcher_args()
    if sys.platform == "win32":
        try:
            from ctypes import windll

            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    app = Launcher(
        record_path=args.record or None,
        auto_run=args.auto_run,
        max_seconds=args.max_seconds,
    )
    app._quit_after_run = bool(args.record)
    app.mainloop()


if __name__ == "__main__":
    main()
