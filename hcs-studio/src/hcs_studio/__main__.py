"""``python -m hcs_studio`` entry point."""

import multiprocessing
import sys

from hcs_studio.app import main

if __name__ == "__main__":
    # Worker processes use the 'spawn' start method; frozen (installer) builds need this.
    multiprocessing.freeze_support()
    sys.exit(main())
