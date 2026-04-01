# Baseline Environment

This file records the baseline implementation and package versions used for the initial SLDP refresh validation work.

## Canonical Baseline

- Baseline package: installed `sldp` in the `sldp` conda environment
- Package path: `/home/tobyc/data/miniforge3/envs/sldp/lib/python3.10/site-packages/sldp`
- Purpose: all early regression checks should compare repository behavior against this installed implementation, not against the checked-in source tree

## Environment Snapshot

Captured on `2026-04-01`.

| Package | Version | Path |
| --- | --- | --- |
| `python` | `3.10.19` | `conda env: sldp` |
| `sldp` | `unknown` | `/home/tobyc/data/miniforge3/envs/sldp/lib/python3.10/site-packages/sldp/__init__.py` |
| `numpy` | `2.2.6` | `/home/tobyc/data/miniforge3/envs/sldp/lib/python3.10/site-packages/numpy/__init__.py` |
| `pandas` | `2.3.3` | `/home/tobyc/data/miniforge3/envs/sldp/lib/python3.10/site-packages/pandas/__init__.py` |
| `scipy` | `1.15.3` | `/home/tobyc/data/miniforge3/envs/sldp/lib/python3.10/site-packages/scipy/__init__.py` |
| `matplotlib` | `3.10.7` | `/home/tobyc/data/miniforge3/envs/sldp/lib/python3.10/site-packages/matplotlib/__init__.py` |
| `pysnptools` | `unknown` | `/home/tobyc/data/miniforge3/envs/sldp/lib/python3.10/site-packages/pysnptools/__init__.py` |
| `gprim` | `unknown` | `/home/tobyc/data/miniforge3/envs/sldp/lib/python3.10/site-packages/gprim/__init__.py` |
| `ypy` | `unknown` | `/home/tobyc/data/miniforge3/envs/sldp/lib/python3.10/site-packages/ypy/__init__.py` |

## Fixture Generation Notes

- `plink` and `plink2` are not installed in the baseline environment.
- `pysnptools` provides `Bed.write(...)`, so the test fixture can generate PLINK bed/bim/fam data directly in Python.
- The first regression fixture should remain intentionally small and deterministic.

## Validation Policy

- Core workflows to compare:
  - `preprocessrefpanel`
  - `preprocessannot`
  - `preprocesspheno`
  - `sldp`
- `storyteller.py` is optional and excluded from the first validation pass.
- Numerical outputs should be compared using tolerances, not byte-for-byte equality.
- Equivalence work should preserve intended baseline workflows, not known baseline bugs.
- For `sldp`, the canonical comparison path is the explicit preprocessed workflow using `--pss-chr`, not implicit auto-preprocessing.

## Baseline Quirks Discovered During Phase 0/1

- `sldp` assumes annotation name discovery via chromosome `22`, even when `--chroms` excludes `22`.
- `sldp` expects `bfile_reg_chr` to contain only regression or printed SNPs, not the full reference panel.
- `sldp -fastp` is broken in the installed baseline because it drops `pfast` and `zfast` instead of `p_fast` and `z_fast`.
- `sldp` with no background annotations hits an `UnboundLocalError` in `chunkstats.residualize` in the installed baseline.
- the refreshed repository adds an explicit `--preprocess` mode, but the default `sldp` path remains analysis-only so baseline-style comparisons still use preprocessed inputs
- The canonical end-to-end baseline fixture therefore uses:
  - a regression-only `bfile_reg_chr`
  - a placeholder chromosome `22` annotation file for name discovery
  - one background annotation
  - `-bothp` rather than `-fastp`
