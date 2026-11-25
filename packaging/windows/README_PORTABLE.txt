SAM 3 portable command-line tool (Windows x64)
==============================================

Requirements: Windows 10/11 x64, NVIDIA GPU + recent driver (CUDA 12.8 capable).
No installation needed: unzip anywhere and run from a terminal.

  sam3-cli.exe --help
  sam3-cli.exe segment --image photo.jpg --prompt "red car" --checkpoint C:\models\sam3.pt --out results

Model weights are NOT included. Obtain them yourself from
https://huggingface.co/facebook/sam3 after accepting the licence there.
All processing happens on this computer; no images are uploaded.
Licence: see LICENSE.txt (SAM License).
