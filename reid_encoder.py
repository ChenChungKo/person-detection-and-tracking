"""Person Re-ID embedding helper (Ultralytics yolo26*-reid)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PersonReIDEncoder:
    """Crop → L2-normalized embedding via Ultralytics ReID ONNX/PT models."""

    def __init__(self, model: str = "yolo26n-reid.onnx", device: str | None = None) -> None:
        from ultralytics.trackers.utils.reid import ReID

        self.model_name = model
        self._encoder = ReID(model, device=device)

    def embed_xyxy(
        self,
        frame: np.ndarray,
        xyxy: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        # Ultralytics ReID expects xywh rows.
        dets = np.array([[cx, cy, w, h]], dtype=np.float32)
        feats = self._encoder(frame, dets)
        if not feats or feats[0] is None:
            return None
        feat = np.asarray(feats[0], dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(feat))
        if n < 1e-6:
            return None
        return feat / n


def resolve_reid_model(name: str) -> str:
    """Return model path/name; bare yolo26*-reid.onnx stays as downloadable asset name."""
    p = Path(name)
    if p.exists():
        return str(p)
    return name
