"""Video face-swap reference (v2.1). The Kotlin port (VideoCore.kt) mirrors the tracking, smoothing and pairing.

Pass 1: decode (trim, reduced fps, <=720p short side, dims multiple of 16) -> MediaPipe detect every processed
        frame (full frame, 2x2 tiles if fewer faces than expected) -> greedy IoU/centre tracker -> per-track
        zero-lag smoothing (One-Euro run forward and backward, averaged) of all 468 mesh points ->
        pairing: tracks visible in the first frame that shows the most faces (up to #source faces), sorted
        left->right, get source faces (i + rotation) % n; later tracks inherit from a non-overlapping paired
        track with the nearest centre.
Pass 2: decode again -> per frame swap every assigned visible track with the cached source latent (AiEngine
        per-frame swap = ai_pipeline.swap_face) -> optional enhancer -> encode H.264, copy audio (trimmed).
"""
import os, sys, json, math, subprocess, time
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ai_pipeline as ap
fp = ap.fp

# ------------------------------------------------------------------ frame selection / geometry
def out_size(w, h, max_short=720):
    s = min(1.0, max_short / min(w, h))
    W, H = int(round(w * s)), int(round(h * s))
    return W - W % 16, H - H % 16, s

def frame_times(duration, src_fps, start, end, fps):
    """Indices of source frames to process: first frame at/after each output slot k/fps."""
    idx = []; slot = 0; n = int(math.floor(duration * src_fps + 1e-6))
    for i in range(n):
        t = i / src_fps
        if t < start - 1e-9 or t >= end - 1e-9: continue
        if t - start >= slot / fps - 1e-9:
            idx.append(i); slot += 1
            while slot / fps <= t - start + 1e-9: slot += 1
    return idx

def prep(frame, W, H, s):
    if s < 1: frame = cv2.resize(frame, (int(round(frame.shape[1] * s)), int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
    y0 = (frame.shape[0] - H) // 2; x0 = (frame.shape[1] - W) // 2
    return np.ascontiguousarray(frame[y0:y0 + H, x0:x0 + W])

# ------------------------------------------------------------------ detection
def detect_frame(img, expected):
    faces = [f[:468] for f in fp._detect_region(img, 0, 0, img.shape[1], img.shape[0])]
    if len(faces) < expected:
        h, w = img.shape[:2]; tw, th = int(w / 2 * 1.5), int(h / 2 * 1.5)
        for gy in range(2):
            for gx in range(2):
                x0 = int(round((w - tw) * gx)); y0 = int(round((h - th) * gy))
                for f in fp._detect_region(img, x0, y0, tw, th): fp._merge_face(faces, f[:468])
    return dedupe([f.astype(np.float32) for f in faces])

def is_dup(a, b):
    """Two detections of the same face (MediaPipe occasionally returns duplicates)."""
    ba, bb = bbox(a), bbox(b)
    if iou(ba, bb) > 0.3: return True
    ca = ((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2); cb = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1]) < 0.5 * min(ba[2] - ba[0], bb[2] - bb[0])

def dedupe(faces):
    keep = []
    for f in faces:
        if not any(is_dup(f, k) for k in keep): keep.append(f)
    return keep

# ------------------------------------------------------------------ tracking (mirrored in Kotlin FaceTracker)
def bbox(p):
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0

def match_score(tb, db):
    s = iou(tb, db)
    if s >= 0.1: return s
    tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2; dcx, dcy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
    r = 0.5 * max(tb[2] - tb[0], db[2] - db[0]); d = math.sqrt((tcx - dcx) ** 2 + (tcy - dcy) ** 2)
    return 0.1 * (1 - d / r) if d < r else 0.0

def track(dets, max_gap):
    """dets: list over frames of lists of (468,2/3) arrays. Returns tracks: list of dict frame->pts (468x2 float64)."""
    tracks = []; last = []   # last: (frame, bbox) per track
    for f, ds in enumerate(dets):
        boxes = [bbox(d) for d in ds]
        act = [t for t in range(len(tracks)) if f - last[t][0] <= max_gap]
        cand = []
        for t in act:
            for j, b in enumerate(boxes):
                s = match_score(last[t][1], b)
                if s > 0: cand.append((-s, t, j))
        cand.sort()
        used_t, used_d = set(), set()
        for _, t, j in cand:
            if t in used_t or j in used_d: continue
            used_t.add(t); used_d.add(j)
            tracks[t][f] = ds[j][:, :2].astype(np.float64); last[t] = (f, boxes[j])
        for j, d in enumerate(ds):
            if j not in used_d:
                tracks.append({f: d[:, :2].astype(np.float64)}); last.append((f, boxes[j]))
    return tracks

# ------------------------------------------------------------------ smoothing (One-Euro forward + backward)
MIN_CUTOFF = 1.0; BETA = 3.0; D_CUTOFF = 1.0
def alpha(cutoff, te):
    tau = 1.0 / (2 * math.pi * cutoff); return 1.0 / (1.0 + tau / te)

def one_euro(seq, te):
    """seq: (L, N, 2). Common cutoff for the whole face from its centroid speed in face-widths/s."""
    out = np.empty_like(seq); out[0] = seq[0]; s_hat = 0.0
    ad = alpha(D_CUTOFF, te)
    for i in range(1, len(seq)):
        c1 = seq[i].mean(0); c0 = seq[i - 1].mean(0)
        w = seq[i][:, 0].max() - seq[i][:, 0].min()
        raw = math.sqrt((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) / te / max(w, 1.0)
        s_hat = ad * raw + (1 - ad) * s_hat
        a = alpha(MIN_CUTOFF + BETA * s_hat, te)
        out[i] = a * seq[i] + (1 - a) * out[i - 1]
    return out

MAX_FILL = 2   # detection drop-outs of <= 2 frames are bridged by linear interpolation
def fill_gaps(tr, max_fill=MAX_FILL):
    frames = sorted(tr); out = dict(tr)
    for a, b in zip(frames, frames[1:]):
        if 1 < b - a <= max_fill + 1:
            for f in range(a + 1, b):
                u = (f - a) / (b - a); out[f] = (1 - u) * tr[a] + u * tr[b]
    return out

def smooth_track(tr, fps):
    tr = fill_gaps(tr)
    te = 1.0 / fps; frames = sorted(tr); out = {}
    runs = []; cur = [frames[0]]
    for f in frames[1:]:
        if f == cur[-1] + 1: cur.append(f)
        else: runs.append(cur); cur = [f]
    runs.append(cur)
    for run in runs:
        seq = np.stack([tr[f] for f in run])
        fw = one_euro(seq, te); bw = one_euro(seq[::-1].copy(), te)[::-1]
        sm = (fw + bw) / 2
        for k, f in enumerate(run): out[f] = sm[k]
    return out

# ------------------------------------------------------------------ pairing
def pair_tracks(tracks, n_src, rotation):
    """Returns list: source index per track (or -1)."""
    nf = max(max(t) for t in tracks) + 1 if tracks else 0
    # pairing only considers "solid" tracks (>= 8 frames, or >= 25% of a very short clip), unless none exist
    solid_len = min(max(3, nf // 4), 8)
    solid = [k for k, t in enumerate(tracks) if len(t) >= solid_len] or list(range(len(tracks)))
    vis = [[k for k in solid if f in tracks[k]] for f in range(nf)]
    target = min(n_src, max((len(v) for v in vis), default=0))
    assign = [-1] * len(tracks)
    if target == 0: return assign, -1
    pf = next(f for f in range(nf) if len(vis[f]) >= target)
    cx = lambda k, f: tracks[k][f][:, 0].mean()
    wd = lambda k, f: float(np.ptp(tracks[k][f][:, 0]))
    biggest = sorted(vis[pf], key=lambda k: -wd(k, pf))[:max(n_src, 1)] if n_src >= 2 else vis[pf]
    first = sorted(biggest, key=lambda k: cx(k, pf))
    for i, k in enumerate(first):
        assign[k] = (i + rotation) % n_src if n_src >= 2 else 0
    # later / other tracks inherit from a paired track they don't overlap in time (re-appearance)
    order = sorted([k for k in range(len(tracks)) if assign[k] < 0], key=lambda k: min(tracks[k]))
    for k in order:
        fk = set(tracks[k]); f0 = min(tracks[k]); c = tracks[k][f0].mean(0)
        best = None
        for j in range(len(tracks)):
            if assign[j] < 0 or j == k or fk & set(tracks[j]): continue
            # nearest in time-adjacent position of track j
            fj = min(tracks[j], key=lambda f: abs(f - f0))
            d = float(np.linalg.norm(tracks[j][fj].mean(0) - c))
            if best is None or d < best[0]: best = (d, j)
        if best is not None:
            assign[k] = assign[best[1]]
    return assign, pf

# ------------------------------------------------------------------ job
def run(video, faces_photo, out_mp4, start=0.0, end=None, fps=15.0, rotation=0, enhance=None, dump_dir=None, log=print):
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n / src_fps; end = min(end or duration, duration)
    fps = min(fps, src_fps)
    W, H, s = out_size(w, h)
    sel = frame_times(duration, src_fps, start, end, fps); selset = {i: k for k, i in enumerate(sel)}
    log(f'{w}x{h} {src_fps}fps {duration:.2f}s -> {W}x{H}, {len(sel)} frames @ {fps} fps')
    src = ap.load(faces_photo); fb = [f[:468].astype(np.float32) for f in ap.detect(src)]
    if not fb: raise RuntimeError('No face found in the faces photo')
    t0 = time.perf_counter()
    dets = []; i = 0
    while True:
        ok, fr = cap.read()
        if not ok or i > sel[-1]: break
        if i in selset: dets.append(detect_frame(prep(fr, W, H, s), min(len(fb), 2)))
        i += 1
    t_det = time.perf_counter() - t0
    tracks = track(dets, max_gap=int(round(fps)))
    sm = [smooth_track(t, fps) for t in tracks] if os.environ.get('VP_SMOOTH', '1') != '0' else [fill_gaps(t) for t in tracks]
    assign, pf = pair_tracks(tracks, len(fb), rotation)
    log(f'tracks {len(tracks)} lengths {[len(t) for t in tracks]} assign {assign} pairing frame {pf}; detection {t_det:.1f}s')
    if pf < 0: raise RuntimeError('No faces found in the video')
    lat = {}
    for si in set(a for a in assign if a >= 0):
        emb, _ = ap.embedding(src, ap.kps5(fb[si])); lat[si] = ap.latent_for(emb)
    cap = cv2.VideoCapture(video); i = 0; k = 0
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps),
                            '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', out_mp4 + '.video.mp4'],
                           stdin=subprocess.PIPE)
    t_swap = 0.0; per_face = []
    if dump_dir: os.makedirs(dump_dir, exist_ok=True)
    while True:
        ok, fr = cap.read()
        if not ok or i > sel[-1]: break
        if i in selset:
            frame = prep(fr, W, H, s); out = frame.copy(); t1 = time.perf_counter()
            for ti, a in enumerate(assign):
                if a >= 0 and k in sm[ti]:
                    tf = time.perf_counter()
                    out = ap.swap_face(out, sm[ti][k].astype(np.float32), lat[a])
                    if enhance: out = ap.enhance(out, sm[ti][k].astype(np.float32), enhance, 0.8)
                    per_face.append(time.perf_counter() - tf)
            t_swap += time.perf_counter() - t1
            if dump_dir and k in (0, 1, 2, len(sel) // 2, len(sel) - 1):
                np.save(f'{dump_dir}/frame{k}_in_rgb.npy', frame[:, :, ::-1].copy()); np.save(f'{dump_dir}/frame{k}_out_rgb.npy', out[:, :, ::-1].copy())
            enc.stdin.write(out.tobytes()); k += 1
            if k % 20 == 0: log(f'  frame {k}/{len(sel)}')
        i += 1
    enc.stdin.close(); enc.wait()
    # audio passthrough (trimmed) if the source has an audio stream
    has_audio = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', video],
                               capture_output=True, text=True).stdout.strip() != ''
    if has_audio:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', out_mp4 + '.video.mp4', '-ss', str(start), '-to', str(end), '-i', video,
                        '-map', '0:v', '-map', '1:a', '-c', 'copy', '-shortest', out_mp4], check=True)
        os.remove(out_mp4 + '.video.mp4')
    else:
        os.replace(out_mp4 + '.video.mp4', out_mp4)
    if dump_dir:
        json.dump({'W': W, 'H': H, 'fps': fps, 'src_fps': src_fps, 'sel': sel, 'assign': assign, 'pairing_frame': pf,
                   'n_src': len(fb), 'rotation': rotation}, open(f'{dump_dir}/meta.json', 'w'))
        np.save(f'{dump_dir}/fb.npy', np.stack(fb)); np.save(f'{dump_dir}/src_rgb.npy', src[:, :, ::-1].copy())
        with open(f'{dump_dir}/dets.json', 'w') as fh:
            json.dump([[d[:, :2].astype(float).round(6).tolist() for d in ds] for ds in dets], fh)
        with open(f'{dump_dir}/tracks.json', 'w') as fh:
            json.dump({'raw': [{str(f): p.round(6).tolist() for f, p in t.items()} for t in tracks],
                       'smooth': [{str(f): p.round(6).tolist() for f, p in t.items()} for t in sm]}, fh)
        for ti, t in enumerate(sm):
            for fk, p in t.items():
                if fk in (0, 1, 2, len(sel) // 2, len(sel) - 1): np.save(f'{dump_dir}/smooth_t{ti}_f{fk}.npy', p)
    stats = dict(frames=len(sel), tracks=len(tracks), assign=assign, detect_s=t_det, swap_s=t_swap,
                 per_face_s=float(np.mean(per_face)) if per_face else 0, has_audio=has_audio, W=W, H=H, fps=fps)
    log(json.dumps(stats))
    return stats

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('video'); p.add_argument('faces'); p.add_argument('out')
    p.add_argument('--start', type=float, default=0); p.add_argument('--end', type=float, default=None)
    p.add_argument('--fps', type=float, default=15); p.add_argument('--rotation', type=int, default=0)
    p.add_argument('--enhance', default=None); p.add_argument('--dump', default=None)
    a = p.parse_args()
    run(a.video, a.faces, a.out, a.start, a.end, a.fps, a.rotation, a.enhance, a.dump)
