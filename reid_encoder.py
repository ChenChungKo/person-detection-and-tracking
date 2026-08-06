"""Person Re-ID embedding helper (YOLO26-ReID or OSNet / OSNet-AIN)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OSNET_DIR = ROOT / "models" / "osnet"
OSNET_HF_REPO = "kaiyangzhou/osnet"

# alias -> (torchreid model_name, HF filename, local cache name)
OSNET_VARIANTS: dict[str, tuple[str, str, str]] = {
    "osnet": (
        "osnet_x1_0",
        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
        "b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "osnet_x1_0_msmt17.pth",
    ),
    "osnet_x1_0": (
        "osnet_x1_0",
        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
        "b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "osnet_x1_0_msmt17.pth",
    ),
    "osnet-x1-0": (
        "osnet_x1_0",
        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
        "b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "osnet_x1_0_msmt17.pth",
    ),
    "osnet_msmt17": (
        "osnet_x1_0",
        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
        "b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "osnet_x1_0_msmt17.pth",
    ),
    "osnet_x1_0_msmt17": (
        "osnet_x1_0",
        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
        "b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "osnet_x1_0_msmt17.pth",
    ),
    # OSNet-AIN: better domain generalization (clothing / camera shift).
    "osnet_ain": (
        "osnet_ain_x1_0",
        "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_"
        "softmax_labsmth_flip_jitter.pth",
        "osnet_ain_x1_0_msmt17.pth",
    ),
    "osnet_ain_x1_0": (
        "osnet_ain_x1_0",
        "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_"
        "softmax_labsmth_flip_jitter.pth",
        "osnet_ain_x1_0_msmt17.pth",
    ),
    "osnet-ain": (
        "osnet_ain_x1_0",
        "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_"
        "softmax_labsmth_flip_jitter.pth",
        "osnet_ain_x1_0_msmt17.pth",
    ),
    "osnet_ain_x1_0_msmt17": (
        "osnet_ain_x1_0",
        "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_"
        "softmax_labsmth_flip_jitter.pth",
        "osnet_ain_x1_0_msmt17.pth",
    ),
    # OSNet-IBN MSMT17 (also stronger generalization).
    "osnet_ibn": (
        "osnet_ibn_x1_0",
        "osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
        "b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "osnet_ibn_x1_0_msmt17.pth",
    ),
    "osnet_ibn_x1_0": (
        "osnet_ibn_x1_0",
        "osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
        "b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "osnet_ibn_x1_0_msmt17.pth",
    ),
}


def _normalize_key(name: str) -> str:
    return Path(name).name.lower().replace(".pth", "").replace(".pt", "").replace(".tar", "")


def _is_osnet_name(name: str) -> bool:
    key = _normalize_key(name)
    if key in OSNET_VARIANTS:
        return True
    return "osnet" in key


def _resolve_variant(name: str) -> tuple[str, str, str]:
    """Return (torchreid_model_name, hf_filename, local_filename)."""
    key = _normalize_key(name)
    if key in OSNET_VARIANTS:
        return OSNET_VARIANTS[key]
    # Explicit weight path: infer architecture from filename.
    lower = key
    if "ain" in lower:
        return OSNET_VARIANTS["osnet_ain_x1_0"]
    if "ibn" in lower:
        return OSNET_VARIANTS["osnet_ibn_x1_0"]
    return OSNET_VARIANTS["osnet_x1_0"]


def ensure_osnet_weights(
    variant: str = "osnet_x1_0",
    path: Path | None = None,
) -> Path:
    """Return local OSNet weight path; download from Hugging Face if missing."""
    arch, hf_file, local_name = _resolve_variant(variant)
    out = Path(path) if path is not None else (OSNET_DIR / local_name)
    if out.exists() and out.stat().st_size > 1_000_000:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    import shutil

    cached = hf_hub_download(repo_id=OSNET_HF_REPO, filename=hf_file)
    shutil.copy2(cached, out)
    return out


class PersonReIDEncoder:
    """Crop → L2-normalized embedding (YOLO26-ReID ONNX or OSNet family)."""

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
        arch, _hf, _local = _resolve_variant(model)
        p = Path(model)
        if p.suffix.lower() in {".pth", ".pt", ".tar"} and p.exists():
            weight = p
        else:
            weight = ensure_osnet_weights(variant=model)

        from torchreid.reid.utils import FeatureExtractor

        self._extractor = FeatureExtractor(
            model_name=arch,
            model_path=str(weight),
            device=self.device,
            verbose=False,
        )
        self.model_name = f"{arch} ({weight.name})"

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
        arch, _hf, _local = _resolve_variant(name)
        return arch
    p = Path(name)
    if p.exists():
        return str(p)
    return name


def _botsort_encoder_from_person_reid(enc: PersonReIDEncoder):
    """Adapt PersonReIDEncoder to Ultralytics BoT-SORT ``(img, xywh_dets)`` API."""

    def encoder(img: np.ndarray, dets: np.ndarray) -> list[np.ndarray | None]:
        out: list[np.ndarray | None] = []
        if dets is None or len(dets) == 0:
            return out
        for det in dets:
            cx, cy, bw, bh = [float(v) for v in det[:4]]
            x1 = int(round(cx - 0.5 * bw))
            y1 = int(round(cy - 0.5 * bh))
            x2 = int(round(cx + 0.5 * bw))
            y2 = int(round(cy + 0.5 * bh))
            feat = enc.embed_xyxy(img, (x1, y1, x2, y2))
            out.append(None if feat is None else np.asarray(feat, dtype=np.float32))
        return out

    return encoder


def patch_ultralytics_botsort_reid(
    shared: PersonReIDEncoder | None = None,
) -> None:
    """Let BoT-SORT ``model: osnet_ain`` use our torchreid OSNet encoder.

    Ultralytics ReID only loads YOLO/ONNX via AutoBackend; OSNet ``.pth`` needs
    this bridge. Call once before the first ``model.track(...)``.
    """
    from ultralytics.trackers.utils import reid as reid_mod

    if getattr(reid_mod, "_5gjump_osnet_patched", False):
        # Refresh shared encoder if provided later.
        if shared is not None:
            reid_mod._5gjump_shared_encoder = shared
        return

    _orig = reid_mod.build_encoder
    reid_mod._5gjump_shared_encoder = shared

    def build_encoder(with_reid: bool, model, device=None):
        if with_reid and model is not None and _is_osnet_name(str(model)):
            shared_enc = getattr(reid_mod, "_5gjump_shared_encoder", None)
            if shared_enc is not None and shared_enc.backend == "osnet":
                return _botsort_encoder_from_person_reid(shared_enc)
            enc = PersonReIDEncoder(str(model), device=device)
            return _botsort_encoder_from_person_reid(enc)
        return _orig(with_reid, model, device)

    reid_mod.build_encoder = build_encoder
    reid_mod._5gjump_osnet_patched = True
