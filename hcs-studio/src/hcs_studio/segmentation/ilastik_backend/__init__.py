"""ilastik as a segmentation and classification backend (Phase 16, PROMPT_ILASTIK.md).

ilastik is never imported into this environment: it runs as a separate program (headless
command line, or Fiji's ilastik4ij plugin). This package holds the settings and detection,
the axis-order and HDF5 utilities, and (later steps) the runners and pipeline steps.
"""
