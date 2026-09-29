"""Face detection with strict confidence / size filters (v2).

MediaPipe FaceLandmarker for 478-pt landmarks, plus a geometric quality score used as the
confidence badge (MediaPipe IMAGE mode does not expose per-face scores). Rejects low-score
and tiny face-like blobs that caused false swaps in v1.
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import vcore

MAX_SIDE = 2048
_tls = threading.local()

# Defaults tuned to cut false positives vs v1 (which used 0.4 / no size floor).
DEFAULT_MIN_CONF = 0.62
DEFAULT_MIN_FACE_FRAC = 0.035   # min(face_w, face_h) / short_side
DEFAULT_MIN_PRESENCE = 0.55


@dataclass
class DetectOpts:
    min_confidence: float = DEFAULT_MIN_CONF
    min_face_frac: float = DEFAULT_MIN_FACE_FRAC
    min_presence: float = DEFAULT_MIN_PRESENCE
    max_faces: int = 6
    tile_rescue: bool = True     # only when few faces found after full-frame pass


@dataclass
class FaceHit:
    """One accepted face: landmarks (478 or 468), confidence 0..1, optional gender label."""
    pts: np.ndarray              # (N, 2|3) float32
    confidence: float
    gender: Optional[str] = None  # "male" | "female" | None
    age: Optional[int] = None
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)

    @property
    def pts468(self) -> np.ndarray:
        return self.pts[:468].astype(np.float32)


def landmarker_path() -> str:
    env = os.environ.get("FSV_LANDMARKER")
    if env:
        return env
    here = Path(__file__).resolve().parent
    cands = [here / "resources" / "face_landmarker.task"]
    if getattr(sys, "_MEIPASS", None):
        cands.insert(0, Path(sys._MEIPASS) / "fsv" / "resources" / "face_landmarker.task")
    for c in cands:
        if c.is_file():
            return str(c)
    raise FileNotFoundError("face_landmarker.task missing")


def _mediapipe_import_shim():
    from . import _install_matplotlib_stub
    _install_matplotlib_stub()


def _landmarker(opts: DetectOpts):
    key = (round(opts.min_confidence, 3), round(opts.min_presence, 3), opts.max_faces)
    cache = getattr(_tls, "lm_cache", None)
    if cache is None:
        cache = {}; _tls.lm_cache = cache
    lm = cache.get(key)
    if lm is None:
        _mediapipe_import_shim()
        from mediapipe.tasks.python import vision, BaseOptions
        with open(landmarker_path(), "rb") as fh:
            data = fh.read()
        conf = float(np.clip(opts.min_confidence, 0.25, 0.95))
        pres = float(np.clip(opts.min_presence, 0.25, 0.95))
        o = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=data),
            running_mode=vision.RunningMode.IMAGE, num_faces=opts.max_faces,
            min_face_detection_confidence=conf, min_face_presence_confidence=pres,
            min_tracking_confidence=conf)
        lm = vision.FaceLandmarker.create_from_options(o)
        cache[key] = lm
    return lm


def geometric_confidence(pts, frame_shape) -> float:
    """0..1 quality score from landmark geometry — rejects blobs / partial faces."""
    p = np.asarray(pts, np.float64)
    if p.shape[0] < 468:
        return 0.0
    h, w = frame_shape[:2]
    short = float(min(h, w))
    x0, y0 = p[:, 0].min(), p[:, 1].min()
    x1, y1 = p[:, 0].max(), p[:, 1].max()
    fw, fh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    size = min(fw, fh) / short
    # eyes / mouth spacing consistency with face width
    from .core import EYE_A, EYE_B, NOSE_IDX
    ex1 = p[EYE_A, :2].mean(0); ex2 = p[EYE_B, :2].mean(0)
    eye_d = float(np.linalg.norm(ex2 - ex1))
    mouth_d = float(np.linalg.norm(p[61, :2] - p[291, :2]))
    nose = p[NOSE_IDX, :2]
    mid_eyes = (ex1 + ex2) / 2
    # eyes roughly horizontal and nose between them
    eye_level = abs(ex1[1] - ex2[1]) / max(eye_d, 1.0)
    nose_between = 1.0 if (min(ex1[0], ex2[0]) - 0.15 * fw) <= nose[0] <= (max(ex1[0], ex2[0]) + 0.15 * fw) else 0.3
    aspect = fw / fh
    aspect_ok = 1.0 if 0.55 <= aspect <= 1.45 else max(0.0, 1.0 - abs(aspect - 1.0))
    eye_ratio = eye_d / fw
    mouth_ratio = mouth_d / fw
    ratio_ok = 1.0 if (0.25 <= eye_ratio <= 0.72 and 0.18 <= mouth_ratio <= 0.65) else 0.35
    size_score = float(np.clip((size - 0.015) / 0.08, 0.0, 1.0))
    level_score = float(np.clip(1.0 - eye_level * 2.5, 0.0, 1.0))
    score = 0.28 * size_score + 0.22 * aspect_ok + 0.22 * ratio_ok + 0.16 * level_score + 0.12 * nose_between
    # slight boost when face is well inside the frame
    margin = min(x0, y0, w - x1, h - y1) / short
    if margin > 0.02:
        score = min(1.0, score + 0.05)
    return float(np.clip(score, 0.0, 1.0))


def _detect_region(img, x0, y0, rw, rh, opts: DetectOpts):
    _mediapipe_import_shim()
    import mediapipe as mp
    crop = np.ascontiguousarray(cv2.cvtColor(img[y0:y0 + rh, x0:x0 + rw], cv2.COLOR_BGR2RGB))
    res = _landmarker(opts).detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    h, w = crop.shape[:2]
    out = []
    for lms in res.face_landmarks:
        if len(lms) < 478:
            continue
        pts = np.array([[l.x * w + x0, l.y * h + y0, l.z * w] for l in lms], dtype=np.float32)
        out.append(pts)
    return out


def _merge_face(faces, f):
    c = f[:, :2].mean(0); fw = f[:, 0].max() - f[:, 0].min()
    for g in faces:
        if np.linalg.norm(g[:, :2].mean(0) - c) < 0.5 * max(fw, g[:, 0].max() - g[:, 0].min()):
            return
    faces.append(f)


def _accept(pts, frame_shape, opts: DetectOpts) -> Optional[FaceHit]:
    conf = geometric_confidence(pts, frame_shape)
    if conf < opts.min_confidence:
        return None
    bb = vcore.bbox(pts)
    short = float(min(frame_shape[:2]))
    size = min(bb[2] - bb[0], bb[3] - bb[1]) / short
    if size < opts.min_face_frac:
        return None
    return FaceHit(pts=pts, confidence=conf, bbox=bb)


def load_image(path):
    """EXIF-aware load (unicode-safe on Windows), downscale to max side 2048 like the reference."""
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Couldn't open that image (supported: JPG, PNG, WebP, BMP, TIFF).")
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def detect_photo(img, opts: Optional[DetectOpts] = None) -> list[FaceHit]:
    """All accepted faces left -> right; optional tiled rescue for small real faces."""
    opts = opts or DetectOpts()
    raw = _detect_region(img, 0, 0, img.shape[1], img.shape[0], opts)
    if opts.tile_rescue and len(raw) < 2:
        for grid in (2, 3):
            h, w = img.shape[:2]
            tw, th = int(w / grid * 1.5), int(h / grid * 1.5)
            for gy in range(grid):
                for gx in range(grid):
                    x0 = int(round((w - tw) * gx / (grid - 1))); y0 = int(round((h - th) * gy / (grid - 1)))
                    for f in _detect_region(img, x0, y0, tw, th, opts):
                        _merge_face(raw, f)
            if len(raw) >= 2:
                break
    hits = []
    for f in raw:
        hit = _accept(f, img.shape, opts)
        if hit is not None:
            hits.append(hit)
    hits.sort(key=lambda h: h.pts[:, 0].mean())
    return hits


def detect_frame(img, expected, opts: Optional[DetectOpts] = None) -> list[np.ndarray]:
    """Back-compat: return list of 468-pt float32 arrays (accepted faces only)."""
    return [h.pts468 for h in detect_frame_hits(img, expected, opts)]


def detect_frame_hits(img, expected, opts: Optional[DetectOpts] = None) -> list[FaceHit]:
    """Video frame detection with filters; tile rescue only if below expected count."""
    opts = opts or DetectOpts()
    raw = _detect_region(img, 0, 0, img.shape[1], img.shape[0], opts)
    if opts.tile_rescue and len(raw) < expected:
        h, w = img.shape[:2]; tw, th = int(w / 2 * 1.5), int(h / 2 * 1.5)
        for gy in range(2):
            for gx in range(2):
                x0 = int(round((w - tw) * gx)); y0 = int(round((h - th) * gy))
                for f in _detect_region(img, x0, y0, tw, th, opts):
                    _merge_face(raw, f[:468] if f.shape[0] > 468 else f)
    hits = []
    for f in raw:
        pts = f[:468] if f.shape[0] >= 468 else f
        hit = _accept(pts, img.shape, opts)
        if hit is not None:
            hit.pts = pts.astype(np.float32)
            hits.append(hit)
    # dedupe by IoU on bboxes
    keep = []
    for h in hits:
        if any(vcore.iou(h.bbox, k.bbox) > 0.3 for k in keep):
            continue
        keep.append(h)
    return keep


def reject_stats(img, opts: Optional[DetectOpts] = None) -> dict:
    """Debug helper: how many raw vs accepted (for selftest / UI)."""
    opts = opts or DetectOpts()
    raw = _detect_region(img, 0, 0, img.shape[1], img.shape[0], opts)
    accepted = [h for f in raw if (h := _accept(f, img.shape, opts)) is not None]
    return dict(raw=len(raw), accepted=len(accepted),
                rejected=len(raw) - len(accepted),
                confidences=[round(geometric_confidence(f, img.shape), 3) for f in raw],
                min_confidence=opts.min_confidence, min_face_frac=opts.min_face_frac)
