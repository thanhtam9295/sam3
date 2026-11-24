# Cell Painting reference data

`test_SQ00014613.sqlite` (a CellProfiler / cytominer-database SQLite with two cells: tables
`Image`, `Nuclei`, `Cells`, `Cytoplasm`) and `test_SQ00014613.csv.gz` (pycytominer's merged
single-cell table of it) are copied unchanged from pycytominer's test data,
`tests/test_data/cytominer_database_example_data/`, at commit
89f976a35df98d843c900a7e4679d49acf2c0f8b of https://github.com/cytomining/pycytominer
(BSD 3-Clause License, Copyright (c) 2019, Broad Institute of MIT and Harvard; the full
licence is in `LICENSE-pycytominer.md`).

They are used as known answers for importing CellProfiler outputs (`tests/test_cellpainting.py`).

`BR00121431/` is a subset of pycytominer's `tests/test_data/collate/analysis/2021_04_20_Target2/
BR00121431/analysis/` (same commit and licence): real CellProfiler ExportToSpreadsheet CSVs of a
JUMP Cell Painting plate (wells A01 and A02, two sites each). Kept: the image metadata, the first
30 cytoplasm objects of each site with their cells and nuclei, floats rounded to 6 significant
digits (gzip). It is the reference for reproducing pycytominer's profiling steps
(`tests/test_cellpainting_profiles.py`).
