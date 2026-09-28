"""AI face-swap reference pipeline (v2.0) - the Kotlin port must match this stage by stage.

Stages
  1. MediaPipe FaceLandmarker (478 pts, same detection + tiled rescans as v1.1) -> 5 keypoints
     [eye on image-left, eye on image-right, nose tip, mouth corner left, mouth corner right]
  2. ArcFace w600k_r50: 112x112 crop aligned to the arcface template, RGB, (x-127.5)/127.5 -> 512-d
     latent = emb @ emap / ||emb||   (emap = last initializer of inswapper_128)
  3. inswapper_128: 128x128 crop (arcface_128 template), RGB/255, + latent -> 128x128 RGB [0,1]
  4. paste back: inverse affine, soft mask = feathered box * feathered face-hull (from mesh)
  5. restorer (GPEN-BFR / GFPGAN / CodeFormer) on 512 ffhq-aligned crop, blended, pasted with feathered mask
Similarity transforms use a closed-form least-squares (Umeyama) fit, easy to reproduce in Kotlin.
"""
import os, sys, time
import numpy as np, cv2, onnx, onnxruntime as ort
from onnx import numpy_helper
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import faceswap_pipeline as fp

MODELS = os.environ.get('AI_MODELS', '/workspace/tools/ai_models')
THREADS = int(os.environ.get('ORT_THREADS', '0'))

ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float64)
ARCFACE_128 = (ARCFACE_112 + np.array([8.0, 0.0])) / 128.0     # insightface norm_crop(128)
ARCFACE_112 = ARCFACE_112 / 112.0                               # all templates normalised to [0,1]
FFHQ_512 = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                     [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float64)

# ---------------------------------------------------------------- detection
def _detect_region478(img, x0, y0, rw, rh):
    crop = np.ascontiguousarray(cv2.cvtColor(img[y0:y0 + rh, x0:x0 + rw], cv2.COLOR_BGR2RGB))
    res = fp._landmarker.detect(fp.mp.Image(image_format=fp.mp.ImageFormat.SRGB, data=crop))
    h, w = crop.shape[:2]
    return [np.array([[l.x * w + x0, l.y * h + y0, l.z * w] for l in lms], dtype=np.float32)
            for lms in res.face_landmarks if len(lms) >= 478]
fp._detect_region = _detect_region478
detect = fp.detect
load = fp.load

NOSE_IDX = int(os.environ.get('NOSE_IDX', '4'))
EYE_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_B = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
def kps5(p):
    """5 keypoints in insightface order from the 468-point MediaPipe mesh:
    eye centres = mean of the 16 eye-contour points (the iris refinement points 468+ can drift badly
    on tilted/closed-eye faces), nose tip = 4, mouth corners 61/291. Ordering follows the face
    (mesh semantics), so it stays correct for rolled heads. float64, sequential sums."""
    p = np.asarray(p, np.float64)
    ex1 = sum(p[EYE_A, 0]) / 16; ey1 = sum(p[EYE_A, 1]) / 16
    ex2 = sum(p[EYE_B, 0]) / 16; ey2 = sum(p[EYE_B, 1]) / 16
    return np.array([[ex1, ey1], [ex2, ey2], p[NOSE_IDX, :2], p[61, :2], p[291, :2]], np.float64)

def umeyama(src, dst):
    """Least-squares similarity (rotation+uniform scale+translation) mapping src->dst, 2x3 (float64).
    Plain sequential sums so the Kotlin port reproduces it exactly."""
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

def sample_bilinear(src, A, W, H, replicate=True):
    """out[y,x] = bilinear sample of src at A @ (x, y, 1) (A: 2x3 dst->src). Taps outside the image are
    clamped (replicate) or zero. Math in float64, result float32. src: HxW or HxWxC (uint8/float32)."""
    h, w = src.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    sx = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
    sy = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    x0 = np.floor(sx); y0 = np.floor(sy); fx = sx - x0; fy = sy - y0
    x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)
    s = src.astype(np.float64)
    def tap(xi, yi):
        if replicate:
            return s[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)]
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        v = s[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)]
        return np.where(ok[..., None] if v.ndim == 3 else ok, v, 0.0)
    if src.ndim == 3: fx = fx[..., None]; fy = fy[..., None]
    p00, p10, p01, p11 = tap(x0, y0), tap(x0 + 1, y0), tap(x0, y0 + 1), tap(x0 + 1, y0 + 1)
    return ((1 - fy) * ((1 - fx) * p00 + fx * p10) + fy * ((1 - fx) * p01 + fx * p11)).astype(np.float32)

def warp(img, kps, template, size):
    """Aligned face crop (float32, same channel order as img, values 0..255) + the 2x3 frame->crop matrix."""
    M = umeyama(kps, template * size)
    return sample_bilinear(img, invert_affine(M), size, size, True), M

def gauss_kernel(sigma):
    n = int(np.floor(sigma * 8 + 1 + 0.5)) | 1; c = (n - 1) / 2
    k = np.exp(-((np.arange(n) - c) ** 2) / (2 * sigma * sigma)); return k / k.sum(), int(c)

def gauss_blur(m, sigma):
    """Separable Gaussian, BORDER_REFLECT_101, float64 accumulation, float32 output after each pass."""
    k, c = gauss_kernel(sigma); h, w = m.shape
    def refl(i, n):
        i = np.array(i)
        while True:
            i = np.where(i < 0, -i, i); i = np.where(i >= n, 2 * n - 2 - i, i)
            if ((i >= 0) & (i < n)).all(): return i
    xi = refl(np.arange(-c, w + c), w); p = m.astype(np.float32)[:, xi].astype(np.float64)
    acc = np.zeros((h, w)); 
    for j in range(len(k)): acc += k[j] * p[:, j:j + w]
    t = acc.astype(np.float32)
    yi = refl(np.arange(-c, h + c), h); p = t[yi, :].astype(np.float64)
    acc = np.zeros((h, w))
    for j in range(len(k)): acc += k[j] * p[j:j + h, :]
    return acc.astype(np.float32)

def fill_polygon(size, q):
    """1.0 where the pixel centre (x, y) is inside polygon q (crossing-number rule), else 0."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    inside = np.zeros((size, size), bool); n = len(q)
    for i in range(n):
        xi, yi = q[i]; xj, yj = q[i - 1]
        cond = (yi > ys) != (yj > ys)
        with np.errstate(divide='ignore', invalid='ignore'):
            xc = (xj - xi) * (ys - yi) / (yj - yi) + xi
        inside ^= cond & (xs < xc)
    return inside.astype(np.float32)

# ---------------------------------------------------------------- sessions (lazy)
_sess = {}
def session(name):
    if name not in _sess:
        so = ort.SessionOptions()
        if THREADS: so.intra_op_num_threads = THREADS; so.inter_op_num_threads = 1
        _sess[name] = ort.InferenceSession(os.path.join(MODELS, name + '.onnx'), so, providers=['CPUExecutionProvider'])
    return _sess[name]

_emap = {}
def emap(model='inswapper_128_fp16'):
    if model not in _emap:
        m = onnx.load(os.path.join(MODELS, model + '.onnx'))
        _emap[model] = numpy_helper.to_array(m.graph.initializer[-1]).astype(np.float32)
    return _emap[model]

_box_cache = {}
def box_mask(size, blur=0.3):
    key = (size, blur)
    if key not in _box_cache:
        amount = int(size * 0.5 * blur); area = max(amount // 2, 1)
        m = np.ones((size, size), np.float32)
        m[:area, :] = 0; m[-area:, :] = 0; m[:, :area] = 0; m[:, -area:] = 0
        _box_cache[key] = gauss_blur(m, amount * 0.25) if amount > 0 else m
    return _box_cache[key]

FACE_OVAL = fp.FACE_OVAL
def hull_polygon(pts, M, grow):
    """Face-oval of the target mesh in crop coords; the upper half is pushed up by 8% of the face
    height (MediaPipe's oval stops at the hairline) and the whole outline grown by `grow`."""
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

def paste(frame, crop, mask, M):
    """Blend the (float32) crop back into the uint8 frame through the inverse of M with a soft mask.
    Only the face bounding box is touched. The crop is sampled at M @ (x, y) directly (no inversion)."""
    h, w = frame.shape[:2]; s = crop.shape[0]
    x0, y0, x1, y1 = paste_bbox(M, s, w, h)
    if x1 <= x0 or y1 <= y0: return frame.copy()
    A = M.copy(); A[0, 2] = M[0, 0] * x0 + M[0, 1] * y0 + M[0, 2]; A[1, 2] = M[1, 0] * x0 + M[1, 1] * y0 + M[1, 2]
    inv = sample_bilinear(crop.astype(np.float32), A, x1 - x0, y1 - y0, True)
    im = np.clip(sample_bilinear(mask.astype(np.float32), A, x1 - x0, y1 - y0, False), 0, 1)[:, :, None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    out = frame.copy()
    out[y0:y1, x0:x1] = np.clip(im * inv + (np.float32(1) - im) * roi + np.float32(0.5), 0, 255).astype(np.uint8)
    return out

# ---------------------------------------------------------------- stages
SWAPPER = 'inswapper_128_fp16'
def embedding(img, kps):
    """ArcFace embedding (raw, 512) of the face at kps. img is BGR (uint8)."""
    crop, _ = warp(img, kps, ARCFACE_112, 112)
    x = ((crop[:, :, ::-1] - np.float32(127.5)) / np.float32(127.5)).transpose(2, 0, 1)[None]
    return session('arcface_w600k_r50').run(None, {'input': np.ascontiguousarray(x, np.float32)})[0][0], crop

def latent_for(emb):
    """inswapper source input: emb @ emap / ||emb||  (float64 math, float32 result)."""
    e = emb.astype(np.float64)
    return ((e @ emap(SWAPPER).astype(np.float64)) / np.sqrt((e * e).sum())).astype(np.float32)[None]

def run_swapper(crop, latent):
    """crop: BGR float32 128x128 (0..255). returns BGR float32 0..255."""
    x = (crop[:, :, ::-1] / np.float32(255)).transpose(2, 0, 1)[None]
    y = session(SWAPPER).run(None, {'target': np.ascontiguousarray(x, np.float32), 'source': latent})[0][0]
    return np.clip(y.transpose(1, 2, 0), 0, 1)[:, :, ::-1] * np.float32(255)

def swap_face(frame, tgt_pts, latent, timings=None, dump=None):
    kps = kps5(tgt_pts)
    crop, M = warp(frame, kps, ARCFACE_128, 128)
    t = time.perf_counter()
    out = run_swapper(crop, latent)
    if timings is not None: timings.append(('swap', time.perf_counter() - t))
    mask = box_mask(128, 0.3) * hull_mask(tgt_pts, M, 128, 0.04, 0.06)
    res = paste(frame, out, mask, M)
    if dump is not None: dump.update(swap_kps=kps, swap_M=M, swap_crop=crop, swap_out=out, swap_mask=mask, swap_result=res)
    return res

RESTORERS = {'gpen512': ('gpen_bfr_512', 512), 'gpen256': ('gpen_bfr_256', 256), 'gfpgan': ('gfpgan_1.4', 512)}
def enhance(frame, tgt_pts, restorer='gpen512', blend=0.8, timings=None, dump=None):
    name, size = RESTORERS[restorer]
    crop, M = warp(frame, kps5(tgt_pts), FFHQ_512, size)
    x = ((crop[:, :, ::-1] / np.float32(255) - np.float32(0.5)) / np.float32(0.5)).transpose(2, 0, 1)[None]
    t = time.perf_counter()
    y = session(name).run(None, {'input': np.ascontiguousarray(x, np.float32)})[0][0]
    if timings is not None: timings.append(('enhance', time.perf_counter() - t))
    y = ((np.clip(y.transpose(1, 2, 0), -1, 1) + np.float32(1)) / np.float32(2))[:, :, ::-1] * np.float32(255)
    y = crop * np.float32(1 - blend) + y * np.float32(blend)
    mask = box_mask(size, 0.3) * hull_mask(tgt_pts, M, size, 0.06, 0.05)
    res = paste(frame, y, mask, M)
    if dump is not None: dump.update(enh_M=M, enh_crop=crop, enh_out=y, enh_mask=mask, enh_result=res)
    return res

def swap(a, b, rotation=0, restorer='gpen512', blend=0.8, timings=None, faces=None, dumps=None):
    """Put the faces of photo b onto the faces of photo a (paired left to right, rotated by `rotation`)."""
    fa, fb = faces if faces else (detect(a), detect(b))
    if not fa: raise RuntimeError('No face found in the main photo')
    if not fb: raise RuntimeError('No face found in the faces photo')
    n = min(len(fa), len(fb))
    out = a.copy(); pairs = []
    for i in range(n):
        si = (i + rotation) % len(fb) if len(fb) >= 2 else 0
        d = {} if dumps is not None else None
        t = time.perf_counter()
        emb, acrop = embedding(b, kps5(fb[si]))
        if timings is not None: timings.append(('arcface', time.perf_counter() - t))
        lat = latent_for(emb)
        if d is not None: d.update(src_kps=kps5(fb[si]), arc_crop=acrop, emb=emb, latent=lat)
        out = swap_face(out, fa[i], lat, timings, d)
        pairs.append((si, i))
        if d is not None: dumps.append(d)
    if restorer:
        for i in range(n):
            d = dumps[i] if dumps is not None else None
            out = enhance(out, fa[i], restorer, blend, timings, d)
    return out, fa, fb, pairs

if __name__ == '__main__':
    pa, pb, outp = sys.argv[1:4]
    a, b = load(pa), load(pb)
    tm = []
    res, fa, fb, pairs = swap(a, b, timings=tm)
    print('faces', len(fa), len(fb), pairs, [(k, round(v, 2)) for k, v in tm])
    cv2.imwrite(outp, res, [cv2.IMWRITE_JPEG_QUALITY, 93])
