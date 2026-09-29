"""Model download manifest (same URLs / SHA-256 as the Android Face Swap Video app).

InsightFace models (ArcFace, inswapper) are licensed for personal / non-commercial
research use only — commercial use needs a licence from insightface.ai.
GPEN (Alibaba DAMO) has no published licence; treat as research / personal use.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

GH = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"
HF = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"


@dataclass(frozen=True)
class ModelSpec:
    file: str
    label: str
    bytes: int
    sha256: str

    @property
    def urls(self) -> list[str]:
        if self.file == "gender_age.onnx":
            return [
                "https://huggingface.co/datasets/Alltitude/insightface/resolve/main/genderage.onnx",
                "https://huggingface.co/uwg/upscaler/resolve/main/Face_Restore/FaceFusion/gender_age.onnx",
            ]
        return [GH + self.file, HF + self.file]


ARCFACE = ModelSpec(
    "arcface_w600k_r50.onnx", "Face identity (ArcFace)", 174_388_474,
    "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
)
SWAPPER = ModelSpec(
    "inswapper_128_fp16.onnx", "Face swap (inswapper 128)", 277_680_829,
    "c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420",
)
ENHANCER_HQ = ModelSpec(
    "gpen_bfr_512.onnx", "Detail enhancer (GPEN 512)", 284_340_240,
    "d5f066b9068a8b74217f9712e28e875a6144629b108a6f7355acbdb3a2832c54",
)
ENHANCER_LIGHT = ModelSpec(
    "gpen_bfr_256.onnx", "Light detail enhancer (GPEN 256)", 75_792_988,
    "bad8bf0426873828df2dbf4e3b3d9ababba9da7965b8b72426569486f7ae5c25",
)

REQUIRED = [ARCFACE, SWAPPER]
REQUIRED_BYTES = sum(s.bytes for s in REQUIRED)

GENDER_AGE = ModelSpec(
    "gender_age.onnx", "Gender / age estimate", 1_322_532,
    "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
)

OPTIONAL = [ENHANCER_LIGHT, ENHANCER_HQ, GENDER_AGE]


def default_models_dir() -> Path:
    env = os.environ.get("FSV_MODELS")
    if env:
        return Path(env)
    # Portable: next to the exe / package, so the Ally X user keeps models with the app.
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    return base / "FaceSwapVideo" / "models"


class ModelStore:
    """SHA-256 verified download with pause/resume and GitHub → Hugging Face mirror fallback."""

    def __init__(self, dir: Path | None = None, specs: Iterable[ModelSpec] = REQUIRED):
        self.dir = Path(dir or default_models_dir())
        self.dir.mkdir(parents=True, exist_ok=True)
        self.specs = list(specs)

    def path(self, s: ModelSpec) -> Path:
        return self.dir / s.file

    def _part(self, s: ModelSpec) -> Path:
        return self.dir / (s.file + ".part")

    def _marker(self, s: ModelSpec) -> Path:
        return self.dir / (s.file + ".ok")

    def is_installed(self, s: ModelSpec) -> bool:
        f, m = self.path(s), self._marker(s)
        if not f.is_file() or f.stat().st_size != s.bytes or not m.is_file():
            return False
        return m.read_text(encoding="utf-8").strip() == s.sha256

    def all_installed(self) -> bool:
        return all(self.is_installed(s) for s in self.specs)

    def bytes_present(self) -> int:
        n = 0
        for s in self.specs:
            if self.is_installed(s):
                n += s.bytes
            else:
                p = self._part(s)
                if p.is_file():
                    n += min(p.stat().st_size, s.bytes)
        return n

    def import_from(self, folder, specs=None, progress=None) -> list[str]:
        """Adopt model files already on disk (e.g. copied from another PC): verify SHA-256, then hard-link or
        copy them into the store. Returns the names imported."""
        got = []
        for s in (list(specs) if specs is not None else self.specs):
            src = Path(folder) / s.file
            if self.is_installed(s) or not src.is_file() or src.stat().st_size != s.bytes:
                continue
            if self._sha256(src, s.bytes, progress, s.file, None) != s.sha256:
                continue
            dest = self.path(s)
            if dest.exists():
                dest.unlink()
            link_or_copy(src, dest)
            self._marker(s).write_text(s.sha256 + "\n", encoding="utf-8")
            got.append(s.file)
        return got

    def ensure(
        self,
        specs: Optional[Iterable[ModelSpec]] = None,
        progress: Optional[Callable[[str, int, int, float, bool], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> None:
        for s in (list(specs) if specs is not None else self.specs):
            if self.is_installed(s):
                continue
            self._download(s, progress, cancelled)

    def _download(
        self,
        s: ModelSpec,
        progress: Optional[Callable[[str, int, int, float, bool], None]],
        cancelled: Optional[Callable[[], bool]],
    ) -> None:
        part = self._part(s)
        have = part.stat().st_size if part.is_file() else 0
        last_err: Exception | None = None
        for url in s.urls:
            try:
                self._fetch(url, part, have, s.bytes, s.file, progress, cancelled)
                break
            except Exception as e:  # noqa: BLE001 — try next mirror
                last_err = e
                have = part.stat().st_size if part.is_file() else 0
        else:
            raise RuntimeError(f"Failed to download {s.file}: {last_err}")

        if progress:
            progress(s.file, s.bytes, s.bytes, 0.0, True)
        digest = self._sha256(part, s.bytes, progress, s.file, cancelled)
        if digest != s.sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {s.file}: got {digest}, expected {s.sha256}")
        dest = self.path(s)
        # Atomic-ish rename
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        part.replace(tmp)
        tmp.replace(dest)
        self._marker(s).write_text(s.sha256 + "\n", encoding="utf-8")

    def _fetch(
        self,
        url: str,
        part: Path,
        have: int,
        total: int,
        label: str,
        progress: Optional[Callable[[str, int, int, float, bool], None]],
        cancelled: Optional[Callable[[], bool]],
    ) -> None:
        headers = {"User-Agent": "FaceSwapVideo-Windows/2.0"}
        if have > 0:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        t0 = time.perf_counter()
        got = have
        with urllib.request.urlopen(req, timeout=60) as resp:
            mode = "ab" if have and resp.status == 206 else "wb"
            if mode == "wb":
                got = 0
            with open(part, mode) as fh:
                while True:
                    if cancelled and cancelled():
                        raise RuntimeError("cancelled")
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    elapsed = max(time.perf_counter() - t0, 1e-3)
                    if progress:
                        progress(label, got, total, (got - have) / elapsed, False)
        if got < total:
            raise RuntimeError(f"Incomplete download of {label}: {got}/{total}")

    @staticmethod
    def _sha256(
        path: Path,
        expected_size: int,
        progress: Optional[Callable[[str, int, int, float, bool], None]],
        label: str,
        cancelled: Optional[Callable[[], bool]],
    ) -> str:
        h = hashlib.sha256()
        done = 0
        with open(path, "rb") as fh:
            while True:
                if cancelled and cancelled():
                    raise RuntimeError("cancelled")
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(label, done, expected_size, 0.0, True)
        return h.hexdigest()


def link_or_copy(src: Path, dest: Path) -> None:
    """Prefer hard-link / symlink of a local model cache into the store (CI / selftest)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        try:
            os.symlink(src, dest)
        except OSError:
            import shutil
            shutil.copy2(src, dest)
