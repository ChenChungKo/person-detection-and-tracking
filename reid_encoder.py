"""Person Re-ID embedding helper (YOLO26-ReID or OSNet)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OSNET_DIR = ROOT / "models" / "osnet"
# KaiyangZhou OSNet trained on MSMT17 (Hugging Face mirror).
OSNET_HF_REPO = "kaiyangzhou/osnet"
OSNET_HF_FILE = (
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
    "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
OSNET_LOCAL = OSNET_DIR / "osnet_x1_0_msmt17.pth"

# Aliases accepted by --reid-model
OSNET_ALIASES = {
    "osnet",
    "osnet_x1_0",
    "osnet-x1-0",
    "osnet_msmt17",
    "osnet_x1_0_msmt17",
}


def _is_osnet_name(name: str) -> bool:
    key = Path(name).name.lower().replace(".pth", "").replace(".pt", "")
    if key in OSNET_ALIASES:
        return True
    return "osnet" in key


def ensure_osnet_weights(path: Path | None = None) -> Path:
    """Return local OSNet weight path; download from Hugging Face if missing."""
    out = path or OSNET_LOCAL
    out = Path(out)
    if out.exists() and out.stat().st_size > 1_000_000:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    import shutil

    cached = hf_hub_download(repo_id=OSNET_HF_REPO, filename=OSNET_HF_FILE)
    shutil.copy2(cached, out)
    return out


class PersonReIDEncoder:
    """Crop → L2-normalized embedding (YOLO26-ReID ONNX or OSNet)."""

    def __init__(self, model: str = "yolo26n-reid.onnx", device: str | None = None) -> None:
        self.model_name = model
        self.backend = "osnet" if _is_osnet_name(model) else "yolo"
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        if self.backend == "osnet":
            self._init_osnet(model)
        else:
            from ultralytics.trackers.utils.reid import ReID

            self._encoder = ReID(model, device=device)

    def _init_osnet(self, model: str) -> None:
        # Prefer explicit path; else download MSMT17 checkpoint.
        p = Path(model)
        if p.suffix.lower() in {".pth", ".pt", ".tar"} and p.exists():
            weight = p
        else:
            weight = ensure_osnet_weights()

        # Import FeatureExtractor without relying on broken torchreid.utils alias.
        from torchreid.reid.utils import FeatureExtractor

        self._extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=str(weight),
            device=self.device,
            verbose=False,
        )
        self.model_name = f"osnet_x1_0 ({weight.name})"

    def embed_xyxy(
        self,
        frame: np.ndarray,
        xyxy: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        if self.backend == "osnet":
            return self._embed_osnet(frame, xyxy)
        return self._embed_yolo(frame, xyxy)

    def _embed_yolo(
        self,
        frame: np.ndarray,
        xyxy: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        dets = np.array([[cx, cy, w, h]], dtype=np.float32)
        feats = self._encoder(frame, dets)
        if not feats or feats[0] is None:
            return None
        feat = np.asarray(feats[0], dtype=np.float32).reshape(-1)
        return self._l2(feat)

    def _embed_osnet(
        self,
        frame: np.ndarray,
        xyxy: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 8 or y2 - y1 < 16:
            return None
        crop_bgr = frame[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return None
        # FeatureExtractor / PIL expect RGB.
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        try:
            tensor = self._extractor([crop_rgb])
        except Exception:
            return None
        feat = np.asarray(tensor[0].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        return self._l2(feat)

    @staticmethod
    def _l2(feat: np.ndarray) -> np.ndarray | None:
        n = float(np.linalg.norm(feat))
        if n < 1e-6:
            return None
        return feat / n


def resolve_reid_model(name: str) -> str:
    """Resolve CLI model name to a path or downloadable asset name."""
    if _is_osnet_name(name):
        p = Path(name)
        if p.suffix.lower() in {".pth", ".pt", ".tar"} and p.exists():
            return str(p)
        return "osnet_x1_0"
    p = Path(name)
    if p.exists():
        return str(p)
    return name
