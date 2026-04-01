# SLDP Refresh Implementation Plan

## Baseline And Scope

- Baseline implementation: the installed `sldp` package in the `sldp` conda environment at `/home/tobyc/data/miniforge3/envs/sldp/lib/python3.10/site-packages/sldp`.
- Target implementation: this repository checkout.
- Minimum supported Python version: `3.11`.
- Initial validation target: match the installed baseline behavior and numerical outputs, except where behavior is intentionally improved or fixed.
- Initial out-of-scope area: `storyteller.py` is optional and not required for the first pass.

## Goals

1. Make the codebase reliable on modern Python and pandas.
2. Improve maintainability and testability before aggressive optimization.
3. Preserve current CLI and output behavior where it still makes sense.
4. Build a regression harness that compares repo outputs against the installed baseline.
5. Remove `gprim` and `ypy` after parity is established.
6. Apply low-risk performance improvements in the first pass when they naturally fall out of cleanup.
7. Follow with benchmark-guided optimization work.

## Constraints And Principles

- Treat the installed `sldp` package as the canonical short-term reference, even if it contains local fixes not present in this repo.
- Prefer small, behavior-preserving changes.
- Use tolerance-based numerical comparisons rather than byte-for-byte output comparisons.
- Preserve CLI flags, output file names, and output schemas unless there is a strong reason to change them.
- Avoid large redesigns before a safety net is in place.

## Current Findings

### Repository shape

- Core modules are concentrated in:
  - `src/sldp/sldp.py`
  - `src/sldp/preprocessannot.py`
  - `src/sldp/preprocesspheno.py`
  - `src/sldp/preprocessrefpanel.py`
  - `src/sldp/weights.py`
  - `src/sldp/chunkstats.py`
  - `src/sldp/config.py`
  - `src/sldp/storyteller.py`
- There are currently no tests or benchmarks in the repository.

### Known modernization issues in the checked-in repo

- `preprocesspheno.py` uses `DataFrame.append`, which is removed in `pandas>=2`.
- `preprocesspheno.py` uses `open(...).next()`, which is invalid on Python 3.
- `sldp.py` uses `se` before assignment in the `more_stats` block.
- `sldp.py` drops `pfast` and `zfast`, but the produced column names are `p_fast` and `z_fast`.
- `sldp.py` references `sanno` instead of `sannot` in a preprocessing status line.
- Several pandas calls use deprecated parsing style such as `delim_whitespace=True`.

### Current dependency surface

The package currently depends on `gprim` and `ypy`, but only uses a small subset of each package:

- `gprim.dataset.Dataset`
- `gprim.annotation.Annotation`
- `gprim.annotation.smart_merge`
- `gprim.annotation.reconciled_to`
- `ypy.memo.memoized`
- `ypy.memo.reset`
- `ypy.fs.makedir`
- `ypy.fs.makedir_for_file`
- `ypy.fs.mem`
- `ypy.pretty.print_namespace`
- `ypy.iter.grouper` indirectly via `gprim.dataset`

This is small enough to internalize after parity is established.

## Phase Plan

## Phase 0: Freeze The Baseline

### Purpose

Define exactly what behavior the refreshed code must match.

### Tasks

1. Record the installed baseline package location and keep it fixed as the comparison target.
2. Capture baseline environment versions:
   - Python
   - numpy
   - pandas
   - scipy
   - matplotlib
   - pysnptools
   - gprim
   - ypy
3. Define the workflows that must be supported and compared:
   - `preprocessrefpanel`
   - `preprocessannot`
   - `preprocesspheno`
   - `sldp`
4. Decide how outputs will be compared:
   - exact text comparison for small metadata files when stable
   - tolerance-based comparison for numerical tables and arrays

### Deliverables

- A short baseline note checked into the repo or test docs.
- Version snapshot of the reference environment.

### Acceptance Criteria

- There is no ambiguity about what implementation is considered correct during the migration.

## Phase 1: Build A Tiny Reproducible Fixture

### Purpose

Create a small synthetic dataset that exercises the core pipeline quickly and deterministically.

### Tasks

1. Build a tiny fixture that includes:
   - one or two chromosomes
   - a miniature PLINK reference panel
   - `ld_blocks`
   - `print_snps`
   - a small `sumstats` file
   - one or two small annotations
2. Ensure the fixture exercises:
   - block iteration
   - allele reconciliation
   - typed and untyped SNP handling
   - SVD generation
   - weight inversion
   - final genome-wide result generation
3. Run the installed baseline implementation on the fixture and save the expected outputs.
4. Document fixture generation and assumptions.

### Deliverables

- A committed test fixture under a path such as `tests/fixtures/`.
- Baseline outputs generated from the installed `sldp` package.

### Acceptance Criteria

- The complete pipeline runs in seconds on the fixture.
- Fixture outputs are stable and can be used for regression tests.

## Phase 2: Add Regression Tests Before Refactoring

### Purpose

Protect behavior before making structural changes.

### Tasks

1. Add `pytest` test infrastructure.
2. Add focused tests for:
   - `config.add_default_params`
   - `weights.invert_weights`
   - `chunkstats.get_est`
   - `chunkstats.jackknife_se`
   - `chunkstats.residualize`
   - `chunkstats.signflip`
   - annotation reconciliation behavior
3. Add smoke tests for each CLI command on the fixture.
4. Add end-to-end regression tests comparing repo outputs to the installed baseline outputs.
5. Define numeric tolerances for output comparison.

### Deliverables

- A runnable test suite.
- Regression tests covering the full core pipeline.

### Acceptance Criteria

- The repository has a safety net capable of detecting behavioral regressions.

## Phase 3: Python 3.11 And Pandas Stabilization

### Purpose

Make the current code reliable under modern Python without yet removing legacy dependencies.

### Tasks

1. Fix runtime issues already identified in the checked-in repository.
2. Replace Python 2-era and removed pandas APIs.
3. Remove obsolete `__future__` imports.
4. Normalize file and gzip handling where necessary.
5. Replace deprecated pandas parsing options with current equivalents.
6. Keep `gprim` and `ypy` in place during this stage to minimize scope.

### Candidate fixes already known

- Replace `DataFrame.append` usage.
- Replace `open(...).next()` with `next(open(...))` or a safer context-managed helper.
- Fix the `se` ordering bug in `sldp.py`.
- Fix fast p-value output column cleanup in `sldp.py`.
- Fix the `sanno` typo.

### Deliverables

- A Python `3.11` compatible codebase that still uses the legacy helper packages.

### Acceptance Criteria

- Tests pass on Python `3.11`.
- Core CLIs run correctly on the fixture.
- Output matches the installed baseline within tolerance.

## Phase 4: Maintainability Refactor

### Purpose

Improve structure, readability, and testability while preserving behavior.

### Tasks

1. Separate CLI parsing from computational logic.
2. Move shared logic into smaller internal helpers where this reduces duplication.
3. Remove unnecessary globals and implicit state.
4. Introduce modern path handling with `pathlib` where appropriate.
5. Add type hints across public functions and core internals.
6. Introduce typed data structures or dataclasses only where they reduce complexity.
7. Keep changes incremental and behavior-preserving.

### Refactor targets

- Make main entry points thin wrappers around reusable functions.
- Make core computational functions independently testable.
- Clarify data flow between preprocessing steps and the main regression workflow.

### Deliverables

- Cleaner module structure.
- Type-annotated, more testable compute functions.

### Acceptance Criteria

- Tests still pass with no material behavior change.
- The main computational logic is no longer tightly coupled to CLI parsing.

## Phase 5: Internalize `gprim` And `ypy`

### Purpose

Remove stale dependencies after behavioral parity is established.

### Tasks

1. Reimplement only the subset of `gprim` and `ypy` that `sldp` actually uses.
2. Add internal modules such as:
   - `sldp.dataset`
   - `sldp.annotation`
   - `sldp.cache`
   - `sldp.fs`
   - `sldp.pretty`
3. Replace imports module-by-module rather than all at once.
4. Remove `gprim` and `ypy` from package dependencies when all tests pass.
5. Update docs and installation instructions.

### Notes

- `Dataset` and `Annotation` behavior should initially remain compatibility-focused.
- Internal replacements should be minimal and closely aligned with current use cases.

### Deliverables

- A self-contained `sldp` package without runtime dependence on `gprim` or `ypy`.

### Acceptance Criteria

- Regression tests pass unchanged.
- Packaging metadata no longer requires `gprim` or `ypy`.

## Phase 6: Low-Risk First-Pass Performance Improvements

### Purpose

Capture easy performance wins during cleanup, without changing algorithmic behavior.

### Candidate improvements

1. Avoid repeated `pd.concat` inside loops when list accumulation or direct assignment is cheaper.
2. Cache repeated annotation name/header reads.
3. Precompute repeated index lookups such as annotation positions.
4. Reduce repeated creation of single-row dataframes.
5. Use arrays instead of dataframes in hot paths where labels are not needed.
6. Avoid repeated reading of the same SVD and metadata files when safe.

### Deliverables

- Small, measured speedups that do not require algorithm changes.

### Acceptance Criteria

- No regression in numerical outputs.
- Observable runtime or memory improvement on the fixture and early benchmarks.

## Phase 7: Benchmark-Guided Optimization

### Purpose

Optimize only after parity and maintainability work are complete.

### Tasks

1. Add benchmark scripts or a reproducible benchmark workflow.
2. Measure runtime and memory for:
   - `preprocessrefpanel`
   - `preprocessannot`
   - `preprocesspheno`
   - `sldp`
3. Profile likely hotspots:
   - SVD and `R2` construction in `preprocessrefpanel.py`
   - per-block matrix operations in `preprocessannot.py`
   - repeated I/O and block processing in `preprocesspheno.py`
   - sign-flipping in `chunkstats.signflip`
4. Prioritize improvements based on measured cost, not guesswork.

### Optimization directions to evaluate

- more array-centric internals in hotspots
- reduced dataframe overhead
- lower-memory sign-flipping implementation
- smarter caching or reuse of loaded matrices
- optional parallelism by chromosome or block if safe and reproducible

### Deliverables

- Benchmark results and targeted optimization patches.

### Acceptance Criteria

- Measurable improvement on representative workloads.
- Numerical outputs remain within tolerance of baseline.

## Phase 8: Packaging, Tooling, And Documentation

### Purpose

Make the refreshed package easier to develop, test, and install.

### Tasks

1. Move toward `pyproject.toml` as the primary packaging entry point.
2. Add development dependencies for:
   - `pytest`
   - `ruff`
   - `mypy`
3. Add CI for tests, linting, and type checks.
4. Update README installation instructions for Python `3.11+`.
5. Document the regression fixture and benchmark workflow.
6. Update dependency guidance after `gprim` and `ypy` are removed.

### Deliverables

- Modernized packaging and contributor workflow.
- Updated documentation.

### Acceptance Criteria

- A new developer can install the package, run tests, and understand the validation workflow.

## Recommended Order Of Work

1. Freeze baseline implementation and environment details.
2. Build the synthetic fixture and capture baseline outputs.
3. Add regression tests and CLI smoke tests.
4. Fix Python `3.11` and pandas compatibility issues.
5. Refactor for maintainability and add type hints.
6. Internalize `gprim` and `ypy`.
7. Apply low-risk performance wins.
8. Run benchmark-guided optimization work.
9. Finish packaging, CI, and documentation.

## Initial Acceptance Criteria For The Refresh

- The repository runs on Python `3.11`.
- Core workflows match the installed baseline on the fixture within defined tolerances.
- The codebase has automated tests covering both focused internals and end-to-end behavior.
- The code is easier to read, type-check, and maintain.
- `gprim` and `ypy` are removed by the end of the migration, unless a specific blocker is found.
- Performance work is driven by benchmarks after parity is established.

## Risks And Mitigations

### Risk: the installed baseline differs from the repo in non-obvious ways

- Mitigation: compare against the installed package directly, not assumptions based on the repo.

### Risk: a tiny fixture misses real-world edge cases

- Mitigation: design the fixture to cover allele flips, missing SNPs, duplicate IDs, and block-level edge cases where possible.

### Risk: performance refactors subtly alter floating-point results

- Mitigation: delay substantial optimization until after regression tests are in place.

### Risk: legacy helper replacement changes behavior

- Mitigation: internalize `gprim` and `ypy` in small steps with regression checks after each step.

## Implementation Notes For Early Work

- `storyteller.py` should be left untouched unless it blocks broader cleanup.
- The first implementation pass should focus on `preprocesspheno.py`, `sldp.py`, and test scaffolding because they contain the clearest compatibility and correctness risks.
- Internal replacement of `gprim` and `ypy` should be done after the code is already stable on Python `3.11`.
