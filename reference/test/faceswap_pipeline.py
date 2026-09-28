"""Desktop reference implementation of the Couple Face Swap pipeline.
Mirrors app/src/main/java/.../FaceSwapper.kt step by step:
  1. load (EXIF-aware), downscale to max side 2048
  2. MediaPipe FaceLandmarker (478 pts, numFaces=6); use first 468 mesh points
  3. sort faces left->right by centre x, pair by index (optional cyclic rotation = "flip")
  4. per pair: piecewise-affine warp of source mesh triangles onto target mesh
     (fixed canonical triangulation, painter's order by depth)
  5. Reinhard LAB colour transfer (masked stats), seamlessClone NORMAL_CLONE on an ROI,
     then feathered alpha blend of the clone over the original to soften the seam.
"""
import sys, os
import numpy as np, cv2
from mediapipe.tasks.python import vision, BaseOptions
import mediapipe as mp

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'face_landmarker.task')
TRIS_FILE = os.path.join(HERE, '..', 'tools', 'face_triangles.txt')  # v1 classic pipeline (assets removed in v2.0)
MAX_SIDE = 2048
TRIS = np.loadtxt(TRIS_FILE, dtype=np.int32)
_mir = np.loadtxt(os.path.join(os.path.dirname(TRIS_FILE), 'face_mirror.txt'), dtype=np.int32)
MIRROR, SIDE = _mir[:, 0], _mir[:, 1]
MIRROR_TRIS = MIRROR[TRIS]
TRI_SIDE = np.where((SIDE[TRIS] >= 0).all(1), 1, np.where((SIDE[TRIS] <= 0).all(1), -1, 0))
TRI_SIDE[(SIDE[TRIS] == 0).all(1)] = 0
OVAL_SHRINK = 0.93     # pull the mask outline 7% towards the face centre (stay on skin, away from hair/ears)
FOREHEAD_KEEP = 0.45   # keep 45% of forehead height above the brows in the blend mask (hair/bangs)
YAW_MIRROR_RATIO = 0.4  # if one half of the source face is < 40% the width of the other (strongly turned head),
                        # texture that hidden half from the mirrored visible half


def signed_area(p):
    u, v = p[..., 1, :] - p[..., 0, :], p[..., 2, :] - p[..., 0, :]
    return (u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]) / 2
# landmarks of the jaw/face oval (MediaPipe FACEMESH_FACE_OVAL order)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # applies EXIF orientation
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


_landmarker = None
def detect(img):
    global _landmarker
    if _landmarker is None:
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
                                            running_mode=vision.RunningMode.IMAGE, num_faces=6,
                                            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4)
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    faces = _detect_region(img, 0, 0, img.shape[1], img.shape[0])
    if len(faces) < 2:
        # small faces (full-body shots): the detector works on a 128px input, so also scan
        # overlapping tiles (2x2 then 3x3) and merge new faces
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


def _detect_region(img, x0, y0, rw, rh):
    crop = np.ascontiguousarray(cv2.cvtColor(img[y0:y0 + rh, x0:x0 + rw], cv2.COLOR_BGR2RGB))
    res = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    h, w = crop.shape[:2]
    return [np.array([[l.x * w + x0, l.y * h + y0, l.z * w] for l in lms[:468]], dtype=np.float32)
            for lms in res.face_landmarks]


def _merge_face(faces, f):
    c = f[:, :2].mean(0); fw = f[:, 0].max() - f[:, 0].min()
    for g in faces:
        if np.linalg.norm(g[:, :2].mean(0) - c) < 0.5 * max(fw, g[:, 0].max() - g[:, 0].min()):
            return
    faces.append(f)


def hidden_side(pts):
    """Return the canonical side (+1/-1) of the source face that is mostly hidden by head yaw, or 0."""
    nose = pts[1, :2]
    w234 = np.linalg.norm(pts[234, :2] - nose); w454 = np.linalg.norm(pts[454, :2] - nose)
    if min(w234, w454) > YAW_MIRROR_RATIO * max(w234, w454):
        return 0
    return int(SIDE[234]) if w234 < w454 else int(SIDE[454])


def warp_face(src_img, src_pts, dst_pts, dst_shape):
    """Piecewise affine warp of src face onto dst geometry. Returns warped image + coverage mask."""
    h, w = dst_shape[:2]
    out = np.zeros((h, w, 3), np.uint8)
    cover = np.zeros((h, w), np.uint8)
    depth = dst_pts[TRIS, 2].mean(axis=1)
    order = np.argsort(-depth)  # far first, near last
    use_mirror = np.zeros(len(TRIS), bool)
    hidden = hidden_side(src_pts)
    if hidden != 0:
        use_mirror = TRI_SIDE == hidden
    for ti in order:
        i, j, k = TRIS[ti]
        si, sj, sk = MIRROR_TRIS[ti] if use_mirror[ti] else (i, j, k)
        t_src = src_pts[[si, sj, sk], :2]
        t_dst = dst_pts[[i, j, k], :2]
        u, v = t_dst[1] - t_dst[0], t_dst[2] - t_dst[0]
        area = abs(u[0] * v[1] - u[1] * v[0]) / 2
        if area < 0.5:
            continue
        rs = cv2.boundingRect(t_src); rd = cv2.boundingRect(t_dst)
        xs, ys, ws, hs = rs; xd, yd, wd, hd = rd
        # clip rects to images
        if ws <= 0 or hs <= 0 or wd <= 0 or hd <= 0: continue
        sx0, sy0 = max(xs, 0), max(ys, 0); sx1, sy1 = min(xs + ws, src_img.shape[1]), min(ys + hs, src_img.shape[0])
        dx0, dy0 = max(xd, 0), max(yd, 0); dx1, dy1 = min(xd + wd, w), min(yd + hd, h)
        if sx1 <= sx0 or sy1 <= sy0 or dx1 <= dx0 or dy1 <= dy0: continue
        patch = src_img[sy0:sy1, sx0:sx1]
        M = cv2.getAffineTransform((t_src - [sx0, sy0]).astype(np.float32), (t_dst - [dx0, dy0]).astype(np.float32))
        warped = cv2.warpAffine(patch, M, (dx1 - dx0, dy1 - dy0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        m = np.zeros((dy1 - dy0, dx1 - dx0), np.uint8)
        cv2.fillConvexPoly(m, np.round(t_dst - [dx0, dy0]).astype(np.int32), 255, lineType=cv2.LINE_8)
        roi = out[dy0:dy1, dx0:dx1]
        roi[m > 0] = warped[m > 0]
        cover[dy0:dy1, dx0:dx1] |= m
    return out, cover


def reinhard(src, dst, mask):
    """Match LAB mean/std of src (inside mask) to dst (inside mask)."""
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = mask > 0
    for c in range(3):
        sm, ss = s[..., c][m].mean(), s[..., c][m].std() + 1e-6
        dm, ds = d[..., c][m].mean(), d[..., c][m].std() + 1e-6
        ratio = np.clip(ds / ss, 0.5, 2.0)
        s[..., c] = (s[..., c] - sm) * ratio + dm
    return cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def mask_polygon(pts):
    """Face-oval polygon with the forehead lowered towards the brows (avoids hair / bangs)."""
    oval = pts[FACE_OVAL, :2].copy()
    up = pts[9, :2] - pts[152, :2]
    up /= (np.linalg.norm(up) + 1e-6)
    h = (oval - pts[9, :2]) @ up
    lift = np.where(h > 0, h * (1 - FOREHEAD_KEEP), 0)
    oval = oval - lift[:, None] * up[None, :]
    c = pts[:, :2].mean(axis=0)
    return c + (oval - c) * OVAL_SHRINK


# facial-feature landmark groups (never treated as occlusion)
FEATURE_GROUPS = [
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 33, 133, 160, 159, 158, 144, 145, 153, 246, 7, 163, 154, 155, 157, 173, 161],
    [300, 293, 334, 296, 336, 285, 295, 282, 283, 276, 263, 362, 387, 386, 385, 373, 374, 380, 466, 249, 390, 381, 382, 384, 398, 388],
    [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    [98, 327, 2, 4, 5, 195, 197, 94, 19, 1, 64, 294, 129, 358, 240, 460],
]
OCCLUSION_THRESH = 3.0


def occlusion_mask(warped, mask, pts, fsize):
    """Pixels of the warped SOURCE face that don't look like its skin (hair strands, hands...)
    outside eyes/brows/nose/mouth and above the mouth line. Returned as 255 = occluded."""
    h, w = mask.shape
    feat = np.zeros((h, w), np.uint8)
    for g in FEATURE_GROUPS:
        cv2.fillConvexPoly(feat, cv2.convexHull(np.round(pts[g, :2]).astype(np.int32)), 255)
    dk = max(3, int(fsize * 0.10) | 1)   # generous: keeps glasses rims / lashes / brows
    feat = cv2.dilate(feat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk)))
    ck = max(3, int(fsize * 0.30) | 1)
    core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
    core = cv2.bitwise_and(core, cv2.bitwise_not(feat))
    if cv2.countNonZero(core) < 30:
        return np.zeros((h, w), np.uint8)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = core > 0
    mean = np.array([lab[..., c][m].mean() for c in range(3)], np.float32)
    std = np.array([lab[..., c][m].std() for c in range(3)], np.float32)
    std = np.maximum(std, [12.0, 4.0, 4.0])
    wts = np.array([0.35, 1.0, 1.0], np.float32)  # lighting varies L a lot; hue/chroma is more telling
    d = np.sqrt((((lab - mean) / std) ** 2 * wts).sum(axis=2))
    occ = ((d > OCCLUSION_THRESH) & (mask > 0) & (feat == 0)).astype(np.uint8) * 255
    # only above the mouth-corner line (keep beards / chins)
    up = pts[9, :2] - pts[152, :2]; up /= (np.linalg.norm(up) + 1e-6)
    mouth = (pts[61, :2] + pts[291, :2]) / 2
    yy, xx = np.mgrid[0:h, 0:w]
    above = ((xx - mouth[0]) * up[0] + (yy - mouth[1]) * up[1]) > 0
    occ[~above] = 0
    ok = max(3, int(fsize * 0.02) | 1)
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)))
    # keep only blobs that reach the outer band of the face (hair / hands come in from the edge)
    bk = max(3, int(fsize * 0.20) | 1)
    band = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bk, bk)))))
    n, labels = cv2.connectedComponents(occ, connectivity=8)
    if n > 1:
        touching = np.unique(labels[(band > 0) & (occ > 0)])
        occ = np.where(np.isin(labels, touching[touching > 0]), 255, 0).astype(np.uint8)
    occ = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok * 2 + 1, ok * 2 + 1)))
    return occ


def swap_one(out, src_img, src_pts, dst_pts):
    h, w = out.shape[:2]
    x, y, bw, bh = cv2.boundingRect(dst_pts[:, :2].astype(np.float32))
    pad = int(0.25 * max(bw, bh)) + 4
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
    roi = out[y0:y1, x0:x1].copy()
    dst_local = dst_pts.copy(); dst_local[:, 0] -= x0; dst_local[:, 1] -= y0
    warped, cover = warp_face(src_img, src_pts, dst_local, roi.shape)
    fsize = max(bw, bh)
    # mask: face-oval polygon of the target, eroded, intersected with coverage
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(mask_polygon(dst_local)).astype(np.int32)), 255)
    mask &= cover
    k = max(3, int(fsize * 0.04) | 1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    occ = occlusion_mask(warped, mask, dst_local, fsize)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(occ))
    mask[:2, :] = 0; mask[-2:, :] = 0; mask[:, :2] = 0; mask[:, -2:] = 0
    if cv2.countNonZero(mask) < 50:
        return out
    corrected = reinhard(warped, roi, mask)
    mx, my, mw, mh = cv2.boundingRect(mask)
    center = (mx + mw // 2, my + mh // 2)
    try:
        cloned = cv2.seamlessClone(corrected, roi, mask.copy(), center, cv2.NORMAL_CLONE)  # clone may modify mask in-place
    except cv2.error:
        cloned = corrected
    # feathered alpha: fully cloned in the interior, smooth fall-off at the border
    fk = max(3, int(fsize * 0.08) | 1)
    alpha = cv2.GaussianBlur(cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fk // 2 | 1, fk // 2 | 1))), (fk, fk), 0)
    alpha = (alpha.astype(np.float32) / 255.0)[..., None]
    blended = cloned.astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def swap(a, b, rotation=0):
    fa, fb = detect(a), detect(b)
    if not fa: raise RuntimeError('No face found in the main photo')
    if not fb: raise RuntimeError('No face found in the faces photo')
    n = min(len(fa), len(fb))
    out = a.copy()
    pairs = []
    for i in range(n):
        si = (i + rotation) % len(fb) if len(fb) >= 2 else 0
        out = swap_one(out, b, fb[si], fa[i]); pairs.append((si, i))
    return out, fa, fb, pairs


if __name__ == '__main__':
    pa = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'A.jpg')
    pb = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'B.jpg')
    outp = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, 'result.jpg')
    rot = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    a, b = load(pa), load(pb)
    res, fa, fb, pairs = swap(a, b, rot)
    print('faces A:', len(fa), 'faces B:', len(fb), 'pairs (B->A):', pairs)
    cv2.imwrite(outp, res, [cv2.IMWRITE_JPEG_QUALITY, 92])
    H = 700
    def rs(im): return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H))
    tiles = []
    for im, lab in ((a, 'A: main photo'), (b, 'B: faces from'), (res, 'Result')):
        t = rs(im); cv2.putText(t, lab, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4); cv2.putText(t, lab, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
        tiles.append(t)
    cmp_path = os.path.join(os.path.dirname(outp), os.path.basename(outp).replace('result', 'comparison'))
    cv2.imwrite(cmp_path, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
