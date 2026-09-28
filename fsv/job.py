"""Two-pass video face-swap job (same algorithm as the Android app / Python reference), tuned for the Ally X.

Pass 1: decode -> MediaPipe detection per selected frame (several landmarkers in parallel) -> tracking,
        forward/backward One-Euro smoothing, left-to-right pairing (cached: Flip only re-runs pass 2).
Pass 2: per frame swap (+ enhancer) in worker threads (model calls serialised on the GPU, pre/post-processing
        overlapped), frames written in order to the H.264 encoder; then the trimmed original audio is muxed in.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import core, detect, media, vcore
from .engine import Engine
from .models import ENHANCER_HQ, ENHANCER_LIGHT, ModelStore

log = logging.getLogger("fsv")

MAX_CLIP_S = 300.0        # 5 minutes — see README "Limits" for the reasoning
MAX_FPS = 60.0
ENHANCE_SPECS = {"gpen256": ENHANCER_LIGHT, "gpen512": ENHANCER_HQ}
ENHANCE_LABEL = {None: "Off", "gpen256": "Light", "gpen512": "HQ"}


class Cancelled(Exception):
    pass


@dataclass
class Settings:
    start: float = 0.0
    length: float = 10.0
    fps: float = 30.0            # 0 = original (capped at 60)
    max_short: int = 1080        # short-side cap
    align: int = 2
    enhance: Optional[str] = "gpen256"
    rotation: int = 0
    device: str = "auto"         # auto | dml | cpu
    out_dir: str = ""
    sequential_decode: bool = False

    def effective_fps(self, src_fps):
        f = self.fps if self.fps and self.fps > 0 else min(src_fps, MAX_FPS)
        return min(f, src_fps)


@dataclass
class Analysis:
    key: tuple
    W: int
    H: int
    s: float
    fps: float
    times: list
    dets: list
    tracks: list
    smooth: list
    n_src: int
    detect_s: float


@dataclass
class PhotoFaces:
    path: str
    img: np.ndarray
    faces: list                  # 478-pt arrays, left -> right


def default_out_dir() -> Path:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            import uuid
            fid = uuid.UUID("{18989B1D-99B5-455B-841C-AB7C74E4DDFC}")      # FOLDERID_Videos
            guid = (ctypes.c_byte * 16).from_buffer_copy(fid.bytes_le)
            p = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(p)) == 0:
                base = Path(p.value); ctypes.windll.ole32.CoTaskMemFree(p)
                return base / "FaceSwap"
        except Exception:  # noqa: BLE001
            pass
    return Path.home() / "Videos" / "FaceSwap"


def cpu_workers():
    n = os.cpu_count() or 4
    det = int(os.environ.get("FSV_DET_THREADS", "0") or 0) or max(1, min(4, n // 4))
    work = int(os.environ.get("FSV_WORKERS", "0") or 0) or max(1, min(4, n // 4))
    return det, work


def load_photo(path) -> PhotoFaces:
    img = detect.load_image(path)
    faces = detect.detect_photo(img)
    return PhotoFaces(str(path), img, faces)


class Job:
    """Holds the engine (sessions stay open between preview / runs) and the pass-1 cache."""

    def __init__(self, store: ModelStore, device="auto"):
        self.store = store
        self.device = device
        self.engine: Optional[Engine] = None
        self.analysis: Optional[Analysis] = None
        self.bench: dict = {}
        self.overhead: dict = {}

    # ------------------------------------------------------------ engine
    def get_engine(self, device=None) -> Engine:
        device = device or self.device
        if self.engine is None or self.engine.device != device:
            if self.engine: self.engine.close()
            self.engine = Engine(self.store, device)
            self.device = device
        return self.engine

    def benchmark(self, device=None):
        eng = self.get_engine(device)
        eng.prepare(None)
        self.bench = eng.benchmark()
        return self.bench

    def recommended_enhance(self):
        """HQ when the GPU makes it cheap, else Light, else Off (only modes whose model is downloaded)."""
        have = {m for m, s in ENHANCE_SPECS.items() if self.store.is_installed(s)}
        b = self.bench
        if "gpen512" in have and "gpen512" in b and b["swap"] + b["gpen512"] <= 0.35:
            return "gpen512"
        if "gpen256" in have:
            return "gpen256"
        return "gpen512" if "gpen512" in have else None

    def estimate(self, n_frames, n_faces, enhance, W=1920, H=1080):
        """Rough seconds for the whole job (detection + swap), from the micro-benchmark."""
        _, work = cpu_workers()
        b = self.bench or {"swap": 0.7, "gpen256": 0.15, "gpen512": 1.3}
        model = b.get("swap", 0.7) + (b.get(enhance, 0.0) if enhance else 0.0)
        scale = max(0.4, (W * H) / (1920 * 1080))
        over = self.overhead.get(enhance, (0.025 + {None: 0, "gpen256": 0.02, "gpen512": 0.06}[enhance]) * scale)
        per_frame = max(model * n_faces, (model + over) * n_faces / work)
        det = n_frames * 0.03 * scale / cpu_workers()[0]
        return det + n_frames * per_frame

    # ------------------------------------------------------------ pass 1
    def analyze(self, info: media.VideoInfo, photo: PhotoFaces, st: Settings, progress=None, cancel=None) -> Analysis:
        fps = st.effective_fps(info.fps)
        end = min(st.start + min(st.length, MAX_CLIP_S), info.duration)
        W, H, s = vcore.out_size(info.width, info.height, st.max_short, st.align)
        n_src = len(photo.faces)
        key = (info.path, os.path.getmtime(info.path), round(st.start, 4), round(end, 4), fps, W, H, min(n_src, 2))
        if self.analysis is not None and self.analysis.key == key:
            return self.analysis
        total = max(1, media.count_selected(info, st.start, end, fps))
        n_det, _ = cpu_workers()
        expected = min(n_src, 2)
        t0 = time.perf_counter()
        times, dets = [], []
        inflight = collections.deque()

        def det_job(fr):
            return detect.detect_frame(vcore.prep(fr, W, H, s), expected)

        with ThreadPoolExecutor(n_det, thread_name_prefix="detect") as pool:
            for k, t, fr in media.iter_frames(info.path, st.start, end, fps, cancel, st.sequential_decode):
                times.append(t)
                inflight.append(pool.submit(det_job, fr))
                while len(inflight) > 2 * n_det:
                    dets.append(inflight.popleft().result())
                    self._tick(progress, "detect", len(dets), total, t0)
                if cancel and cancel(): raise Cancelled()
            while inflight:
                dets.append(inflight.popleft().result())
                self._tick(progress, "detect", len(dets), total, t0)
        if cancel and cancel(): raise Cancelled()
        if not dets:
            raise ValueError("No frames in the selected range.")
        tracks = vcore.track(dets, max_gap=int(round(fps)))
        smooth = [vcore.smooth_track(t, fps) for t in tracks]
        self.analysis = Analysis(key, W, H, s, fps, times, dets, tracks, smooth, n_src, time.perf_counter() - t0)
        return self.analysis

    @staticmethod
    def _tick(progress, stage, done, total, t0, extra=None):
        if progress is None: return
        el = time.perf_counter() - t0
        rate = done / el if el > 0 else 0
        eta = (total - done) / rate if rate > 0 else None
        d = dict(stage=stage, done=done, total=total, rate=rate, eta=eta)
        if extra: d.update(extra)
        progress(d)

    # ------------------------------------------------------------ preview (single frame, no tracking)
    def preview(self, info, photo: PhotoFaces, st: Settings, t=None):
        eng = self.get_engine(st.device)
        eng.prepare(st.enhance)
        W, H, s = vcore.out_size(info.width, info.height, st.max_short, st.align)
        t = st.start if t is None else t
        fr = media.read_frame_at(info.path, t)
        if fr is None: raise ValueError("Couldn't read that part of the video.")
        frame = vcore.prep(fr, W, H, s)
        dets = detect.detect_frame(frame, min(len(photo.faces), 2))
        assign = vcore.pair_single_frame(dets, len(photo.faces), st.rotation)
        lat = self._latents(eng, photo, assign)
        faces = [(dets[i].astype(np.float32), lat[a]) for i, a in enumerate(assign) if a >= 0]
        t0 = time.perf_counter()
        out = core.process_frame(eng, frame, faces, st.enhance)
        el = time.perf_counter() - t0
        if faces and self.bench:
            model = self.bench.get("swap", 0) + (self.bench.get(st.enhance, 0) if st.enhance else 0)
            self.overhead[st.enhance] = max(0.0, el / len(faces) - model)
        return frame, out, dets, assign

    @staticmethod
    def _latents(eng, photo, assign):
        lat = {}
        for si in sorted(set(a for a in assign if a >= 0)):
            emb = core.embedding(eng, photo.img, core.kps5(photo.faces[si][:468]))
            lat[si] = core.latent_for(eng, emb)
        return lat

    # ------------------------------------------------------------ full job
    def run(self, info: media.VideoInfo, photo: PhotoFaces, st: Settings, out_path=None, progress=None,
            cancel=None, dump_dir=None):
        if not photo.faces: raise ValueError("No face found in the faces photo.")
        if st.enhance and not self.store.is_installed(ENHANCE_SPECS[st.enhance]):
            raise ValueError(f"The {ENHANCE_LABEL[st.enhance]} enhancer isn't downloaded yet.")
        t_start = time.perf_counter()
        an = self.analyze(info, photo, st, progress, cancel)
        assign, pf = vcore.pair_tracks(an.tracks, an.n_src, st.rotation)
        if pf < 0: raise ValueError("No faces found in the selected part of the video.")
        eng = self.get_engine(st.device)
        dev = eng.prepare(st.enhance)
        lat = self._latents(eng, photo, assign)
        fb468 = [f[:468].astype(np.float32) for f in photo.faces]

        out_dir = Path(st.out_dir) if st.out_dir else default_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_path is None:
            stem = "".join(c for c in Path(info.path).stem if c.isalnum() or c in "-_ ")[:40].strip() or "video"
            out_path = out_dir / f"FaceSwap_{stem}_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        out_path = Path(out_path)
        tmp_video = out_path.with_name(out_path.stem + ".video.tmp.mp4")
        n = len(an.times)
        enc = media.Encoder(tmp_video, an.W, an.H, an.fps, info)
        _, n_work = cpu_workers()
        dump_idx = {0, 1, 2, n // 2, n - 1} if dump_dir else set()
        if dump_dir: os.makedirs(dump_dir, exist_ok=True)
        t0 = time.perf_counter()
        last_thumb = [0.0]

        def work(k, frame):
            faces = [(an.smooth[ti][k].astype(np.float32), lat[a]) for ti, a in enumerate(assign)
                     if a >= 0 and k in an.smooth[ti]]
            return core.process_frame(eng, frame, faces, st.enhance)

        ok = False
        try:
            inflight = collections.deque()
            written = 0

            def drain_one():
                nonlocal written
                k, frame, fut = inflight.popleft()
                out = fut.result()
                enc.write(out)
                if k in dump_idx:
                    np.save(f"{dump_dir}/frame{k}_in_rgb.npy", frame[:, :, ::-1].copy())
                    np.save(f"{dump_dir}/frame{k}_out_rgb.npy", out[:, :, ::-1].copy())
                written += 1
                extra = {}
                now = time.perf_counter()
                if now - last_thumb[0] > 0.5 or written == n:
                    last_thumb[0] = now; extra = {"thumb": out, "before": frame}
                self._tick(progress, "swap", written, n, t0, extra)

            with ThreadPoolExecutor(n_work, thread_name_prefix="swap") as pool:
                for k, t, fr in media.iter_frames(info.path, st.start, an.times[-1] + 1e-3, an.fps, cancel, st.sequential_decode):
                    if k >= n: break
                    frame = vcore.prep(fr, an.W, an.H, an.s)
                    inflight.append((k, frame, pool.submit(work, k, frame)))
                    while len(inflight) > n_work + 1:
                        drain_one()
                    if cancel and cancel(): raise Cancelled()
                while inflight:
                    drain_one()
            enc.close()
            if written != n:
                raise RuntimeError(f"Decoded {written} frames, expected {n}")
            if progress: progress(dict(stage="mux", done=n, total=n))
            length = n / an.fps
            note = media.mux_audio(tmp_video, info.path, st.start, length, out_path, info)
            ok = True
        except BaseException:
            enc.kill()
            raise
        finally:
            if not ok:
                for p in (tmp_video, out_path):
                    try: Path(p).unlink(missing_ok=True)
                    except OSError: pass
        swap_s = time.perf_counter() - t0
        res = dict(path=str(out_path), frames=n, fps=an.fps, W=an.W, H=an.H, duration=n / an.fps,
                   encoder=enc.encoder, audio=note, device=dev.label(), device_active=dev.active,
                   per_model=dict(dev.per_model), enhance=ENHANCE_LABEL[st.enhance], tracks=len(an.tracks),
                   track_lengths=[len(t) for t in an.tracks], assign=assign, pairing_frame=pf,
                   detect_s=round(an.detect_s, 2), swap_s=round(swap_s, 2),
                   total_s=round(time.perf_counter() - t_start, 2), size=out_path.stat().st_size)
        if dump_dir:
            self._dump(dump_dir, info, an, assign, pf, photo, st)
        if progress: progress(dict(stage="done", done=n, total=n, result=res))
        return res

    @staticmethod
    def _dump(d, info, an, assign, pf, photo, st):
        """Same layout as reference video_pipeline --dump (for the parity comparison)."""
        sel = [int(round(t * info.fps)) for t in an.times]
        json.dump({"W": an.W, "H": an.H, "fps": an.fps, "src_fps": info.fps, "sel": sel, "assign": assign,
                   "pairing_frame": pf, "n_src": len(photo.faces), "rotation": st.rotation}, open(f"{d}/meta.json", "w"))
        np.save(f"{d}/fb.npy", np.stack([f[:468].astype(np.float32) for f in photo.faces]))
        np.save(f"{d}/src_rgb.npy", photo.img[:, :, ::-1].copy())
        with open(f"{d}/dets.json", "w") as fh:
            json.dump([[x[:, :2].astype(float).round(6).tolist() for x in ds] for ds in an.dets], fh)
        with open(f"{d}/tracks.json", "w") as fh:
            json.dump({"raw": [{str(f): p.round(6).tolist() for f, p in t.items()} for t in an.tracks],
                       "smooth": [{str(f): p.round(6).tolist() for f, p in t.items()} for t in an.smooth]}, fh)
