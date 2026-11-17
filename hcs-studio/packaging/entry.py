"""Entry point of the frozen application (PyInstaller). Both executables use it:
``HCS Studio.exe`` (windowed) starts the GUI; ``hcs-studio.exe`` (console) is the CLI."""

import multiprocessing
import sys

if __name__ == "__main__":
    # Must run first: worker processes of a frozen app start this executable again.
    multiprocessing.freeze_support()
    from hcs_studio.app import main

    sys.exit(main())
