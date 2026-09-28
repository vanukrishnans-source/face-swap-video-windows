"""MediaPipe FaceLandmarker detection — port of the reference (faceswap_pipeline.detect with the 478-point
region detector from ai_pipeline, and video_pipeline.detect_frame). One landmarker per thread so pass 1 can
run on several CPU cores (the Z1 Extreme has 8 cores / 16 threads)."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from . import vcore

MAX_SIDE = 2048
_tls = threading.local()


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
    """mediapipe 1.0.x imports matplotlib.pyplot (drawing helpers only) at package import. The packaged app
    doesn't ship matplotlib (~60 MB), so provide an empty stand-in when it's missing."""
    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception:  # noqa: BLE001
        import types
        mpl = types.ModuleType("matplotlib"); plt = types.ModuleType("matplotlib.pyplot")
        mpl.pyplot = plt; sys.modules["matplotlib"] = mpl; sys.modules["matplotlib.pyplot"] = plt


def _landmarker():
    lm = getattr(_tls, "lm", None)
    if lm is None:
        _mediapipe_import_shim()
        from mediapipe.tasks.python import vision, BaseOptions
        # read the model bytes ourselves: MediaPipe's C API can't open non-ASCII Windows paths
        with open(landmarker_path(), "rb") as fh:
            data = fh.read()
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=data),
                                            running_mode=vision.RunningMode.IMAGE, num_faces=6,
                                            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4)
        lm = vision.FaceLandmarker.create_from_options(opts)
        _tls.lm = lm
    return lm


def _detect_region(img, x0, y0, rw, rh):
    _mediapipe_import_shim()
    import mediapipe as mp
    crop = np.ascontiguousarray(cv2.cvtColor(img[y0:y0 + rh, x0:x0 + rw], cv2.COLOR_BGR2RGB))
    res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    h, w = crop.shape[:2]
    return [np.array([[l.x * w + x0, l.y * h + y0, l.z * w] for l in lms], dtype=np.float32)
            for lms in res.face_landmarks if len(lms) >= 478]


def _merge_face(faces, f):
    c = f[:, :2].mean(0); fw = f[:, 0].max() - f[:, 0].min()
    for g in faces:
        if np.linalg.norm(g[:, :2].mean(0) - c) < 0.5 * max(fw, g[:, 0].max() - g[:, 0].min()):
            return
    faces.append(f)


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


def detect_photo(img):
    """All faces (478-pt arrays) left -> right; tiled rescans for small faces (reference fp.detect)."""
    faces = _detect_region(img, 0, 0, img.shape[1], img.shape[0])
    if len(faces) < 2:
        for grid in (2, 3):
            h, w = img.shape[:2]
            tw, th = int(w / grid * 1.5), int(h / grid * 1.5)
            for gy in range(grid):
                for gx in range(grid):
                    x0 = int(round((w - tw) * gx / (grid - 1))); y0 = int(round((h - th) * gy / (grid - 1)))
                    for f in _detect_region(img, x0, y0, tw, th):
                        _merge_face(faces, f)
            if len(faces) >= 2:
                break
    faces.sort(key=lambda p: p[:, 0].mean())
    return faces


def detect_frame(img, expected):
    """reference video_pipeline.detect_frame: full frame, 2x2 tiles if faces are missing, dedupe."""
    faces = [f[:468] for f in _detect_region(img, 0, 0, img.shape[1], img.shape[0])]
    if len(faces) < expected:
        h, w = img.shape[:2]; tw, th = int(w / 2 * 1.5), int(h / 2 * 1.5)
        for gy in range(2):
            for gx in range(2):
                x0 = int(round((w - tw) * gx)); y0 = int(round((h - th) * gy))
                for f in _detect_region(img, x0, y0, tw, th): _merge_face(faces, f[:468])
    return vcore.dedupe([f.astype(np.float32) for f in faces])
