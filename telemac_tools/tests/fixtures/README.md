# HEC-RAS Test Fixtures

Place real HEC-RAS geometry HDF5 files here (.g##.hdf).

Source: https://github.com/fema-ffrd/rashdf (requires git lfs clone)

Tests in test_real_hecras.py auto-discover all .hdf files in this directory.
When no files are present, tests are skipped.
