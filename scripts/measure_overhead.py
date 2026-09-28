"""Measure the non-model (CPU) cost per face of the swap / enhance stages with a zero-cost fake model,
at 720p and 1080p, to ground the Ally X speed estimates."""
import sys, time, json
sys.path.insert(0, '.')
import numpy as np, cv2
from fsv import core, detect, vcore, media

class Fake:
    def run(self, name, feeds):
        if name == 'inswapper_128_fp16': return [np.full((1, 3, 128, 128), 0.5, np.float32)]
        if name.startswith('gpen'):
            s = 256 if '256' in name else 512; return [np.zeros((1, 3, s, s), np.float32)]
        return [np.random.rand(1, 512).astype(np.float32)]
    def emap(self): return np.eye(512, dtype=np.float32)

fr = media.read_frame_at('/workspace/faceswap-app/test/video/mixkit_48205.mp4', 2.0)
out = {}
for short in (720, 1080):
    s = short / 720
    img = cv2.resize(fr, (int(1280 * s), int(720 * s)), interpolation=cv2.INTER_CUBIC) if s != 1 else fr
    t = time.perf_counter(); dets = detect.detect_frame(img, 2); tdet = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(5): detect.detect_frame(img, 2)
    tdet = (time.perf_counter() - t) / 5
    lat = np.zeros((1, 512), np.float32); m = Fake()
    res = {'detect_ms_per_frame': round(tdet * 1000, 1)}
    for mode in (None, 'gpen256', 'gpen512'):
        faces = [(d.astype(np.float32), lat) for d in dets]
        core.process_frame(m, img, faces, mode)
        t = time.perf_counter(); n = 5
        for _ in range(n): core.process_frame(m, img, faces, mode)
        res[f'overhead_ms_per_face_{mode or "off"}'] = round((time.perf_counter() - t) / n / len(faces) * 1000, 1)
    out[f'{img.shape[1]}x{img.shape[0]}'] = res
print(json.dumps(out, indent=1))
