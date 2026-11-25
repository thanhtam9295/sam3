---
name: windows-portable-build
description: How to build, smoke-test and hand over the portable Windows executable of the SAM 3 command-line analysis tool with PyInstaller. Use when asked to package, build an .exe, prepare a Windows release, or change packaging/windows/.
---

# Windows portable build

Full guide: `docs/deployment/WINDOWS_PORTABLE.md`.

1. On a Windows x64 machine with an NVIDIA GPU, Python 3.12 and Git:
   ```powershell
   powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
   ```
2. Smoke-test: `dist\sam3-portable\sam3-cli.exe --help` and
   `sam3-cli.exe segment --image assets\images\truck.jpg --prompt truck --checkpoint <local .pt> --out out\`.
3. The bundle **never** contains checkpoints, tokens or data. Weights are passed with
   `--checkpoint` or downloaded by the end user under their own HF licence acceptance.
4. The CI workflow `windows-portable.yml` is manual (`workflow_dispatch`) and only
   uploads a short-lived workflow artifact. Creating a GitHub Release or
   distributing the .exe requires explicit human approval.
5. Record the build (commit, lock file, CUDA/torch versions) in `docs/DEVLOG.md`.
