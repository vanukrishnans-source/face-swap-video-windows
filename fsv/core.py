"""Face-swap stages — a port of the parity-tested Python reference (reference/test/ai_pipeline.py).

Same math, same float64 sequencing as the reference (which the Android Kotlin port matches to 1e-6 px /
88–94 dB). Two speed changes that keep results identical or near-identical:
  * sample_bilinear gathers the 4 taps from the source array in its own dtype and converts only the
    gathered values to float64 (the reference converts the whole frame first) -> bit-identical, and
    ~10x less memory traffic on 1080p frames;
  * gauss_blur uses OpenCV's separable filter in float64 with BORDER_REFLECT_101 (same kernel,
    same float32 rounding between passes) -> differences at the 1e-16 level.
All model calls go through `models.run(name, feeds)` so the engine can pick DirectML or CPU.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float64)
ARCFACE_128 = (ARCFACE_112 + np.array([8.0, 0.0])) / 128.0
ARCFACE_112 = ARCFACE_112 / 112.0
FFHQ_512 = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                     [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float64)

# MediaPipe FACEMESH_FACE_OVAL order
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYE_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_B = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
NOSE_IDX = 4

# enhance mode -> (model key, crop size)
RESTORERS = {"gpen256": ("gpen_bfr_256", 256), "gpen512": ("gpen_bfr_512", 512)}


def kps5(p):
    p = np.asarray(p, np.float64)
    ex1 = sum(p[EYE_A, 0]) / 16; ey1 = sum(p[EYE_A, 1]) / 16
    ex2 = sum(p[EYE_B, 0]) / 16; ey2 = sum(p[EYE_B, 1]) / 16
    return np.array([[ex1, ey1], [ex2, ey2], p[NOSE_IDX, :2], p[61, :2], p[291, :2]], np.float64)


def umeyama(src, dst):
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64); n = len(src)
    msx = sum(src[:, 0]) / n; msy = sum(src[:, 1]) / n; mdx = sum(dst[:, 0]) / n; mdy = sum(dst[:, 1]) / n
    a = b = var = 0.0
    for i in range(n):
        sx, sy, dx, dy = src[i, 0] - msx, src[i, 1] - msy, dst[i, 0] - mdx, dst[i, 1] - mdy
        a += sx * dx + sy * dy; b += sx * dy - sy * dx; var += sx * sx + sy * sy
    ca, sb = a / var, b / var
    return np.array([[ca, -sb, mdx - (ca * msx - sb * msy)], [sb, ca, mdy - (sb * msx + ca * msy)]], np.float64)


def invert_affine(M):
    a, b, c = M[0]; d, e, f = M[1]
    D = a * e - b * d; D = 1.0 / D if D != 0 else 0.0
    A11, A22, A12, A21 = e * D, a * D, -b * D, -d * D
    return np.array([[A11, A12, -A11 * c - A12 * f], [A21, A22, -A21 * c - A22 * f]], np.float64)


_grid_lock = threading.Lock()
_grids: dict = {}


def _grid(W, H):
    key = (W, H)
    g = _grids.get(key)
    if g is None:
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        g = (xs, ys)
        if W * H <= 1 << 20:
            with _grid_lock:
                if len(_grids) > 64:
                    _grids.clear()
                _grids[key] = g
    return g


def sample_bilinear(src, A, W, H, replicate=True):
    """Bit-identical to the reference: out[y,x] = bilinear(src, A @ (x, y, 1)); float64 math, float32 out."""
    h, w = src.shape[:2]
    xs, ys = _grid(W, H)
    sx = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
    sy = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    x0 = np.floor(sx); y0 = np.floor(sy); fx = sx - x0; fy = sy - y0
    x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)

    def tap(xi, yi):
        v = src[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)].astype(np.float64)
        if replicate:
            return v
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        return np.where(ok[..., None] if v.ndim == 3 else ok, v, 0.0)

    if src.ndim == 3:
        fx = fx[..., None]; fy = fy[..., None]
    p00, p10, p01, p11 = tap(x0, y0), tap(x0 + 1, y0), tap(x0, y0 + 1), tap(x0 + 1, y0 + 1)
    return ((1 - fy) * ((1 - fx) * p00 + fx * p10) + fy * ((1 - fx) * p01 + fx * p11)).astype(np.float32)


def warp(img, kps, template, size):
    M = umeyama(kps, template * size)
    return sample_bilinear(img, invert_affine(M), size, size, True), M


def gauss_kernel(sigma):
    n = int(np.floor(sigma * 8 + 1 + 0.5)) | 1; c = (n - 1) / 2
    k = np.exp(-((np.arange(n) - c) ** 2) / (2 * sigma * sigma)); return k / k.sum(), int(c)


_ONE = np.ones((1, 1), np.float64)


def gauss_blur(m, sigma):
    """Separable Gaussian, BORDER_REFLECT_101, float64 accumulation, float32 after each pass (as reference)."""
    k, c = gauss_kernel(sigma)
    h, w = m.shape
    if c >= h or c >= w:  # kernel wider than the image: fall back to the reference's multi-reflection
        return _gauss_blur_ref(m, sigma)
    kx = k.reshape(1, -1); ky = k.reshape(-1, 1)
    t = cv2.sepFilter2D(m.astype(np.float32).astype(np.float64), cv2.CV_64F, kx, _ONE,
                        borderType=cv2.BORDER_REFLECT_101).astype(np.float32)
    return cv2.sepFilter2D(t.astype(np.float64), cv2.CV_64F, _ONE, ky,
                           borderType=cv2.BORDER_REFLECT_101).astype(np.float32)


def _gauss_blur_ref(m, sigma):
    k, c = gauss_kernel(sigma); h, w = m.shape

    def refl(i, n):
        i = np.array(i)
        while True:
            i = np.where(i < 0, -i, i); i = np.where(i >= n, 2 * n - 2 - i, i)
            if ((i >= 0) & (i < n)).all(): return i
    xi = refl(np.arange(-c, w + c), w); p = m.astype(np.float32)[:, xi].astype(np.float64)
    acc = np.zeros((h, w))
    for j in range(len(k)): acc += k[j] * p[:, j:j + w]
    t = acc.astype(np.float32)
    yi = refl(np.arange(-c, h + c), h); p = t[yi, :].astype(np.float64)
    acc = np.zeros((h, w))
    for j in range(len(k)): acc += k[j] * p[j:j + h, :]
    return acc.astype(np.float32)


def fill_polygon(size, q):
    xs, ys = _grid(size, size)
    inside = np.zeros((size, size), bool); n = len(q)
    for i in range(n):
        xi, yi = q[i]; xj, yj = q[i - 1]
        cond = (yi > ys) != (yj > ys)
        with np.errstate(divide='ignore', invalid='ignore'):
            xc = (xj - xi) * (ys - yi) / (yj - yi) + xi
        inside ^= cond & (xs < xc)
    return inside.astype(np.float32)


_box_cache: dict = {}


def box_mask(size, blur=0.3):
    key = (size, blur)
    m = _box_cache.get(key)
    if m is None:
        amount = int(size * 0.5 * blur); area = max(amount // 2, 1)
        m = np.ones((size, size), np.float32)
        m[:area, :] = 0; m[-area:, :] = 0; m[:, :area] = 0; m[:, -area:] = 0
        m = gauss_blur(m, amount * 0.25) if amount > 0 else m
        _box_cache[key] = m
    return m


def hull_polygon(pts, M, grow):
    p = np.asarray(pts, np.float64)
    oval = p[FACE_OVAL, :2].copy(); n = len(oval)
    cx = sum(oval[:, 0]) / n; cy = sum(oval[:, 1]) / n
    ux, uy = p[10, 0] - p[152, 0], p[10, 1] - p[152, 1]
    fh = np.sqrt(ux * ux + uy * uy); ux /= (fh + 1e-6); uy /= (fh + 1e-6)
    t = (oval[:, 0] - cx) * ux + (oval[:, 1] - cy) * uy
    tmax = max(abs(v) for v in t)
    push = np.maximum(t, 0) / (tmax + 1e-6) * 0.08 * fh
    ox = oval[:, 0] + push * ux; oy = oval[:, 1] + push * uy
    ox = cx + (ox - cx) * (1 + grow); oy = cy + (oy - cy) * (1 + grow)
    return np.stack([M[0, 0] * ox + M[0, 1] * oy + M[0, 2], M[1, 0] * ox + M[1, 1] * oy + M[1, 2]], 1)


def hull_mask(pts, M, size, grow=0.04, feather=0.06):
    return gauss_blur(fill_polygon(size, hull_polygon(pts, M, grow)), feather * size)


def paste_bbox(M, s, w, h):
    Mi = invert_affine(M)
    xs = [Mi[0, 0] * cx + Mi[0, 1] * cy + Mi[0, 2] for cx, cy in ((0, 0), (s, 0), (0, s), (s, s))]
    ys = [Mi[1, 0] * cx + Mi[1, 1] * cy + Mi[1, 2] for cx, cy in ((0, 0), (s, 0), (0, s), (s, s))]
    x0 = max(int(np.floor(min(xs))) - 2, 0); y0 = max(int(np.floor(min(ys))) - 2, 0)
    x1 = min(int(np.ceil(max(xs))) + 2, w); y1 = min(int(np.ceil(max(ys))) + 2, h)
    return x0, y0, x1, y1


def paste(frame, crop, mask, M, inplace=False):
    h, w = frame.shape[:2]; s = crop.shape[0]
    x0, y0, x1, y1 = paste_bbox(M, s, w, h)
    out = frame if inplace else frame.copy()
    if x1 <= x0 or y1 <= y0: return out
    A = M.copy(); A[0, 2] = M[0, 0] * x0 + M[0, 1] * y0 + M[0, 2]; A[1, 2] = M[1, 0] * x0 + M[1, 1] * y0 + M[1, 2]
    inv = sample_bilinear(crop.astype(np.float32), A, x1 - x0, y1 - y0, True)
    im = np.clip(sample_bilinear(mask.astype(np.float32), A, x1 - x0, y1 - y0, False), 0, 1)[:, :, None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(im * inv + (np.float32(1) - im) * roi + np.float32(0.5), 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- model stages
def embedding(models, img, kps):
    """ArcFace embedding (raw 512-d) of the face at kps; img is BGR uint8."""
    crop, _ = warp(img, kps, ARCFACE_112, 112)
    x = ((crop[:, :, ::-1] - np.float32(127.5)) / np.float32(127.5)).transpose(2, 0, 1)[None]
    return models.run('arcface_w600k_r50', {'input': np.ascontiguousarray(x, np.float32)})[0][0]


def latent_for(models, emb):
    e = emb.astype(np.float64)
    return ((e @ models.emap().astype(np.float64)) / np.sqrt((e * e).sum())).astype(np.float32)[None]


def run_swapper(models, crop, latent):
    x = (crop[:, :, ::-1] / np.float32(255)).transpose(2, 0, 1)[None]
    y = models.run('inswapper_128_fp16', {'target': np.ascontiguousarray(x, np.float32), 'source': latent})[0][0]
    return np.clip(y.transpose(1, 2, 0), 0, 1)[:, :, ::-1] * np.float32(255)


# ---------------------------------------------------------------- colour / hairline blend (v2)
FOREHEAD_KEEP = 0.55   # how much forehead to keep in the oval (rest blends to body/hair)


def reinhard_lab(src_bgr, dst_bgr, mask_f):
    """Match LAB mean/std of src to dst inside mask (Reinhard). mask_f is float32 0..1."""
    m = mask_f > 0.35
    if int(m.sum()) < 40:
        return src_bgr
    s = cv2.cvtColor(np.clip(src_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = cv2.cvtColor(np.clip(dst_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        sm, ss = float(s[..., c][m].mean()), float(s[..., c][m].std()) + 1e-6
        dm, ds = float(d[..., c][m].mean()), float(d[..., c][m].std()) + 1e-6
        ratio = float(np.clip(ds / ss, 0.55, 1.85))
        s[..., c] = (s[..., c] - sm) * ratio + dm
    return cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def hairline_soft_mask(pts, M, size, grow=0.04, feather=0.06):
    """Hull mask with forehead lowered toward brows so hair colour at the hairline stays from the body."""
    p = np.asarray(pts, np.float64)
    oval = p[FACE_OVAL, :2].copy(); n = len(oval)
    cx = sum(oval[:, 0]) / n; cy = sum(oval[:, 1]) / n
    ux, uy = p[10, 0] - p[152, 0], p[10, 1] - p[152, 1]
    fh = np.sqrt(ux * ux + uy * uy) + 1e-6; ux /= fh; uy /= fh
    # lower forehead part of oval toward brows (index 9 is glabella-ish; 10 is top)
    top = p[10, :2]; brow = p[9, :2] if p.shape[0] > 9 else p[8, :2]
    t = (oval[:, 0] - cx) * ux + (oval[:, 1] - cy) * uy
    tmax = max(abs(v) for v in t) + 1e-6
    # pull high points (forehead) down toward brow
    pull = np.maximum(t, 0) / tmax * (1.0 - FOREHEAD_KEEP) * fh * 0.55
    ox = oval[:, 0] - pull * ux; oy = oval[:, 1] - pull * uy
    ox = cx + (ox - cx) * (1 + grow); oy = cy + (oy - cy) * (1 + grow)
    poly = np.stack([M[0, 0] * ox + M[0, 1] * oy + M[0, 2], M[1, 0] * ox + M[1, 1] * oy + M[1, 2]], 1)
    return gauss_blur(fill_polygon(size, poly), feather * size)


def paste_color_matched(frame, crop, mask, M, inplace=False, color_match=True, seamless=False):
    """Paste with optional Reinhard colour match to the destination ROI (skin/body tone) and soft blend."""
    h, w = frame.shape[:2]; s = crop.shape[0]
    x0, y0, x1, y1 = paste_bbox(M, s, w, h)
    out = frame if inplace else frame.copy()
    if x1 <= x0 or y1 <= y0:
        return out
    A = M.copy()
    A[0, 2] = M[0, 0] * x0 + M[0, 1] * y0 + M[0, 2]
    A[1, 2] = M[1, 0] * x0 + M[1, 1] * y0 + M[1, 2]
    inv = sample_bilinear(crop.astype(np.float32), A, x1 - x0, y1 - y0, True)
    im = np.clip(sample_bilinear(mask.astype(np.float32), A, x1 - x0, y1 - y0, False), 0, 1)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    if color_match:
        inv = reinhard_lab(inv, roi, im)
    if seamless and im.max() > 0.5:
        try:
            mu8 = (np.clip(im, 0, 1) * 255).astype(np.uint8)
            # shrink mask a bit for Poisson stability
            k = max(3, (min(x1 - x0, y1 - y0) // 30) | 1)
            mu8 = cv2.erode(mu8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            if cv2.countNonZero(mu8) >= 50:
                mx, my, mw, mh = cv2.boundingRect(mu8)
                center = (mx + mw // 2, my + mh // 2)
                cloned = cv2.seamlessClone(
                    np.clip(inv, 0, 255).astype(np.uint8),
                    np.clip(roi, 0, 255).astype(np.uint8),
                    mu8.copy(), center, cv2.NORMAL_CLONE)
                inv = cloned.astype(np.float32)
        except Exception:  # noqa: BLE001 — fall back to alpha blend
            pass
    im3 = im[:, :, None]
    out[y0:y1, x0:x1] = np.clip(im3 * inv + (np.float32(1) - im3) * roi + np.float32(0.5), 0, 255).astype(np.uint8)
    return out


def swap_face(models, frame, tgt_pts, latent, inplace=False, color_match=True, seamless=False):
    """Landmark-driven swap: warp target face (expression/pose) every frame, inject source identity.

    Expression/mouth come from the *target video* landmarks (ArcFace-128 aligned crop of the
    current frame) — the source photo is identity only via the latent, not a frozen texture paste.
    """
    kps = kps5(tgt_pts)
    crop, M = warp(frame, kps, ARCFACE_128, 128)
    out = run_swapper(models, crop, latent)
    mask = box_mask(128, 0.3) * hairline_soft_mask(tgt_pts, M, 128, 0.04, 0.06)
    return paste_color_matched(frame, out, mask, M, inplace, color_match=color_match, seamless=seamless)


def enhance(models, frame, tgt_pts, restorer='gpen256', blend=0.8, inplace=False, color_match=True):
    name, size = RESTORERS[restorer]
    crop, M = warp(frame, kps5(tgt_pts), FFHQ_512, size)
    x = ((crop[:, :, ::-1] / np.float32(255) - np.float32(0.5)) / np.float32(0.5)).transpose(2, 0, 1)[None]
    y = models.run(name, {'input': np.ascontiguousarray(x, np.float32)})[0][0]
    y = ((np.clip(y.transpose(1, 2, 0), -1, 1) + np.float32(1)) / np.float32(2))[:, :, ::-1] * np.float32(255)
    y = crop * np.float32(1 - blend) + y * np.float32(blend)
    mask = box_mask(size, 0.3) * hairline_soft_mask(tgt_pts, M, size, 0.06, 0.05)
    return paste_color_matched(frame, y, mask, M, inplace, color_match=color_match, seamless=False)


def process_frame(models, frame, faces, enhance_mode, color_match=True, seamless=False, temporal_ema=0.0, prev_out=None):
    """faces: list of (pts468 float32, latent). Swap then (optionally) enhance.

    temporal_ema: if >0 and prev_out given, EMA-blend the final frame toward prev_out to cut flicker
    without freezing expression (landmarks still come from the current frame).
    """
    out = frame.copy()
    for pts, lat in faces:
        swap_face(models, out, pts, lat, inplace=True, color_match=color_match, seamless=seamless)
        if enhance_mode:
            enhance(models, out, pts, enhance_mode, 0.8, inplace=True, color_match=color_match)
    if temporal_ema and prev_out is not None and prev_out.shape == out.shape:
        a = float(np.clip(temporal_ema, 0.0, 0.45))
        out = np.clip((1.0 - a) * out.astype(np.float32) + a * prev_out.astype(np.float32), 0, 255).astype(np.uint8)
    return out
