# Third-party components

## Bundled with the portable Windows build

| Component | Licence | Notes |
|---|---|---|
| **FFmpeg** (BtbN `win64-lgpl-shared` build) | LGPL 2.1+ | H.264 via `h264_amf` (AMD), `h264_mf`, or `libopenh264` (Cisco BSD). No GPL encoders (no libx264). Source: https://github.com/BtbN/FFmpeg-Builds. Full licence texts ship in the `ffmpeg/` folder next to the app. |
| **OpenH264** (pulled in by the FFmpeg LGPL build) | BSD-2-Clause | Cisco OpenH264. |
| **ONNX Runtime DirectML** (`onnxruntime-directml`) | MIT | Microsoft. DirectML EP for the Radeon 780M; automatic CPU fallback. |
| **MediaPipe** FaceLandmarker | Apache-2.0 | Google. Bundled `face_landmarker.task`. |
| **OpenCV** (headless) | Apache-2.0 | Video decode via its built-in FFmpeg. |
| **Qt 6 / PySide6 Essentials** | LGPLv3 | GUI. Dynamically linked; you can replace the Qt DLLs. |
| **NumPy** | BSD-3-Clause | |
| **PyInstaller bootloader** | GPL with linking exception | https://pyinstaller.org |

## Downloaded on first run (not redistributed in the zip)

| File | Size | Source | Licence |
|---|---|---|---|
| `arcface_w600k_r50.onnx` | 174.4 MB | FaceFusion models-3.0.0 (InsightFace) | InsightFace **non-commercial** |
| `inswapper_128_fp16.onnx` | 277.7 MB | FaceFusion models-3.0.0 (InsightFace) | InsightFace **non-commercial** |
| `gpen_bfr_256.onnx` (optional Light) | 75.8 MB | FaceFusion / Alibaba DAMO GPEN | research / personal |
| `gpen_bfr_512.onnx` (optional HQ) | 284.3 MB | FaceFusion / Alibaba DAMO GPEN | research / personal |

Primary URL: `https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/<file>`  
Mirror: `https://huggingface.co/facefusion/models-3.0.0/resolve/main/<file>`  
SHA-256 values match the Android Face Swap Video app's `ModelStore` manifest.

## Test media

Mixkit stock clips (Mixkit Stock Video Free License) and a CC0 Wikimedia Commons couple photo — see `testdata/SOURCES.md`.


## gender_age.onnx (InsightFace)
Bundled 1.32 MB gender/age estimator. Same InsightFace personal / non-commercial research terms as ArcFace / inswapper.
