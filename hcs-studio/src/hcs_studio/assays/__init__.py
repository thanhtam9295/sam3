"""Oncology assay modules (Phase 20): one-click analyses calibrated on each plate's controls.

Each assay is a table step (it runs after segmentation and measurement, on the object tables)
built on :class:`~hcs_studio.assays.base.AssayModule`: channel roles mapped to the plate's
channels, a preset pipeline, a calibration on the run's control wells (stored in the run),
registered readouts, assay-specific QC warnings and a report page in ``assays/<name>/``.
"""
