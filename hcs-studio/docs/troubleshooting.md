# Troubleshooting

**Where are the logs?** Each run writes `logs/run.log` in its run folder. The application
writes `%USERPROFILE%\.hcs_studio\logs\hcs-studio.log` (`~/.hcs_studio/logs` on macOS and
Linux). An unexpected error in the application is shown in a message box and recorded
there; the application keeps running and your data is not changed. Please include that
file when reporting a problem.

**Does my installation work?** Run `hcs-studio selftest`; every line should read PASS.

| Problem | What to do |
|---|---|
| "The run needs about N GB but only M GB is free" | Free space, choose another output folder, or run without saving corrected images (`--no-images`, or untick *Save corrected images*). |
| A run stopped with "the output disk is full" | Free space, then **Resume a run folder…** (or `hcs-studio run --resume <folder>`). Completed wells are kept. |
| A well failed with "out of memory" | Use fewer worker processes, or set a memory limit (*Memory limit* in the run dialog, `--max-memory-gb`). Large Z stacks: add a Z projection early in the pipeline. |
| The computer becomes unresponsive during a run | Same as above. The automatic worker count assumes about 40 × the field size per worker (0.5 GB measured for 2-channel 1080 × 1080 fields); deconvolution and Cellpose need more. |
| Runs are slow | `run_manifest.json` → `performance` shows the time per step. Object features take most of the time (61 % on the reference plate), Haralick texture most of all; switch texture and granularity off in *Measure: objects* if you do not need them. Reading is rarely the bottleneck. Cellpose on the CPU is about 15 s per 512 × 512 field; use the `cuda` installer and an NVIDIA GPU. |
| "CPU only (PyTorch found no CUDA device)" | Install the `cuda` variant, update the NVIDIA driver, and check `hcs-studio info`. |
| "HCS Studio: graphics" at start, or the viewer stays black on a remote desktop | The image viewer (napari) needs OpenGL 2.1 from the graphics driver. Remote Desktop without GPU acceleration, virtual machines and old drivers do not provide it. Ask IT to enable the group policy *Use hardware graphics adapters for all Remote Desktop Services sessions*, update the graphics driver, or start HCS Studio on the PC itself. The command line (`hcs-studio run`, `report`, `pharm`, …) works without OpenGL. Experimental, not verified on Windows: set `HCS_SOFTWARE_OPENGL=1` before starting for software rendering (Mesa). |
| Flat-field correction says it needs more fields | It estimates the illumination from at least `min_fields` fields across the plate; run it on more wells, or remove the step for small tests. |
| "This run folder was made with a different pipeline…" when resuming | Resuming requires the same image steps, plate, metadata and layout. Start a new run. (Changing only table steps — gates, classifiers — is fine.) |
| Ctrl+C in the command line | The run stops after the current field and prints the `--resume` command (exit code 130). Press Ctrl+C twice to quit at once. |
| Opening a field from Results says the run did not save images | The run was made with `--no-images`; open the raw field from the Plate tab. |
