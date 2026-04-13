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

## Next Execution Plan

This section supersedes the older open-ended phase descriptions below for the next round of work. It is written to be implemented incrementally by coding agents with clear boundaries, validation commands, and acceptance criteria.

### Phase A: Make Equivalence Explicit

#### Objective

Turn baseline parity from a documented assumption into a reproducible, automated check against the installed `sldp` package in the `sldp` conda environment.

#### Tasks

1. Add a dedicated live-baseline test or script that runs the installed commands in the `sldp` conda environment:
   - `preprocessrefpanel`
   - `preprocessannot`
   - `preprocesspheno`
   - `sldp`
2. Run the repo implementation on the same copied fixture in an isolated temp directory.
3. Compare the following outputs between installed baseline and repo implementation:
   - final `.gwresults`
   - phenotype `info`
   - selected `.pss.gz` files
   - selected annotation `.info` and `.RV.gz` files
   - selected `R.npz` and `R2.npz` outputs
4. Keep the current frozen-artifact regression test as the fast path.
5. Document which known baseline bugs are intentionally excluded from parity requirements.

#### Deliverables

- A runnable script or pytest-marked integration test for live baseline parity.
- Comparison helpers for tabular, gzip, and `.npz` artifacts.
- Updated test docs describing fast regression vs live baseline parity.

#### Suggested Implementation Boundaries

- One agent adds reusable comparison helpers under `tests/`.
- One agent adds the live-baseline execution harness.
- One agent updates docs and fixture instructions.

#### Validation

```bash
conda run -n sldp python -m pytest tests/equivalence
```

#### Acceptance Criteria

- A contributor can run one documented command and verify repo outputs against the installed package.
- Parity checks cover intermediate artifacts, not only final `toy.gwresults`.

### Phase B: Deduplicate Shared Workflow Helpers

#### Objective

Remove duplicated file-loading and path-shaping logic from the main workflow modules.

#### Tasks

1. Introduce a small shared internal module for repeated workflow helpers.
2. Centralize:
   - LD block loading
   - print-SNP loading
   - any repeated path normalization helpers that are semantically identical
3. Decide and document one MHC-handling policy.
4. Update `sldp.py`, `preprocessannot.py`, `preprocesspheno.py`, and `preprocessrefpanel.py` to use the shared helpers.
5. Add focused tests proving the shared helper behavior.

#### Deliverables

- A shared helper module with tests.
- Removal of duplicated `_load_ldblocks` and `_load_print_snps` implementations.

#### Suggested Implementation Boundaries

- One agent extracts the helper module and updates preprocessors.
- One agent updates `sldp.py` call sites and tests.

#### Validation

```bash
source .venv/bin/activate && pytest tests/current/test_preprocessrefpanel.py tests/current/test_preprocessannot.py tests/current/test_preprocesspheno.py tests/current/test_sldp_helpers.py
```

#### Acceptance Criteria

- Shared loaders exist in one place only.
- All callers use the same semantics for MHC filtering and file parsing.

### Phase C: Finish The Maintainability Refactor Of `sldp.py`

#### Objective

Reduce the remaining orchestration density in the main SLDP command without changing behavior.

#### Tasks

1. Split `src/sldp/sldp.py` into smaller internal concerns:
   - processed-input validation and preprocessing delegation
   - annotation and phenotype loading
   - per-block statistic collection
   - result assembly and output writing
2. Keep CLI parser construction in the entrypoint module.
3. Move large internal helpers into one or more workflow-oriented modules only where the extraction is behavior-preserving and reduces coupling.
4. Preserve the existing public CLI and `run(args)` contract.
5. Add helper-level tests around any newly extracted units.

#### Deliverables

- Smaller main module with clearer ownership boundaries.
- New internal workflow modules if justified by the extraction.

#### Suggested Implementation Boundaries

- One agent extracts processed-input handling.
- One agent extracts block-statistics collection.
- One agent updates tests and keeps imports stable.

#### Validation

```bash
source .venv/bin/activate && pytest tests/current/test_sldp_helpers.py tests/current/test_sldp_preprocessing_hooks.py tests/equivalence/test_phase1_regression.py
```

#### Acceptance Criteria

- `src/sldp/sldp.py` becomes materially smaller and easier to scan.
- No behavior changes in current tests or equivalence checks.

### Phase D: Remove Statefulness That Blocks Parallelization

#### Objective

Prepare the code for later parallel execution by isolating global mutable state and side-effect-heavy numeric routines.

#### Tasks

1. Replace global RNG usage with explicit generator plumbing where feasible.
   - start with `chunkstats.signflip`
2. Evaluate whether `memo.py` should remain global, become scoped, or be removed from hot-path orchestration.
3. Reduce unnecessary in-place mutation of shared dataframes in the heaviest workflow helpers.
4. Add tests to lock down deterministic behavior when a seed or generator is provided.

#### Deliverables

- Deterministic stochastic helpers that do not rely on global `numpy` RNG state.
- A documented memoization policy for future parallel execution.

#### Suggested Implementation Boundaries

- One agent handles RNG plumbing and tests.
- One agent handles memoization scoping or documented containment.

#### Validation

```bash
source .venv/bin/activate && pytest tests/current/test_chunkstats.py tests/current/test_chunkstats_signflip.py tests/current/test_local_helpers.py
```

#### Acceptance Criteria

- Core numeric randomness can be reproduced without mutating process-global RNG state.
- Memoization behavior is explicit and no longer an invisible cross-task hazard.

### Phase E: Close Test Coverage Gaps

#### Objective

Add focused tests in the areas that are still structurally risky or under-verified.

#### Tasks

1. Expand `weights` tests to cover all supported modes.
2. Add targeted `storyteller` tests or explicitly document it as out of scope if it will remain optional.
3. Add tests for background-annotation and no-background edge cases where only some modes are currently exercised.
4. Add artifact-comparison helpers that assert tolerances consistently.

#### Deliverables

- New tests for currently thinly covered numeric and plotting paths.
- Clear documentation for any intentionally untested optional functionality.

#### Validation

```bash
source .venv/bin/activate && pytest tests/current
```

#### Acceptance Criteria

- All supported `weights` modes are directly exercised.
- Optional paths are either tested or explicitly declared non-goals.

### Phase F: Add Lightweight Automation And Measurement

#### Objective

Make it cheap to keep the maintainability and equivalence gains intact.

#### Tasks

1. Add CI for:
   - `ruff check src tests`
   - `mypy src`
   - `pytest`
2. Add a small benchmark or timing harness for the tiny fixture and at least one representative command.
3. Document the standard validation command set in `README.md`.

#### Deliverables

- CI workflow configuration.
- One repeatable benchmark or timing script.

#### Validation

```bash
conda run -n sldp python -m ruff check src tests
conda run -n sldp python -m mypy src
conda run -n sldp python -m pytest
```

#### Acceptance Criteria

- Repo health checks run automatically.
- Future performance work starts from measured baselines instead of intuition.

## Recommended Agent Order

1. Phase A first, so all later refactors are guarded by stronger parity checks.
2. Phase B next, because it is small, low risk, and removes obvious drift.
3. Phase C after B, once shared helpers exist.
4. Phase D before any true parallel implementation.
5. Phase E as gaps become clearer during B-D.
6. Phase F in parallel with or immediately after A-E.

## Definition Of Done For This Refactor Track

The maintainability-and-equivalence track should be considered complete when all of the following are true:

- baseline parity is runnable against the installed `sldp` package, not only frozen artifacts
- duplicated workflow loaders are removed
- `sldp.py` is no longer the dominant orchestration bottleneck
- global RNG and memoization behavior are explicit enough for future parallel work
- current and equivalence tests cover the supported workflow surface with documented exceptions
- lint, type checking, and tests run automatically in CI

## Constraints And Principles

- Treat the installed `sldp` package as the canonical short-term reference, even if it contains local fixes not present in this repo.
- Prefer small, behavior-preserving changes.
- Use tolerance-based numerical comparisons rather than byte-for-byte output comparisons.
- Preserve CLI flags, output file names, and output schemas unless there is a strong reason to change them.
- Avoid large redesigns before a safety net is in place.

## Current Findings

## Current Status Snapshot

- Baseline environment and reference package location are recorded in `BASELINE_ENVIRONMENT.md`.
- The repository now has automated tests split by intent:
  - `tests/equivalence/` for baseline-comparison coverage
  - `tests/current/` for current-package unit and integration coverage
  - `tests/fixtures/phase1_tiny/` for the canonical deterministic fixture
- The refreshed code passes `pytest`, `ruff`, and `mypy` in the `sldp` conda environment.
- Core modules have been modernized for Python 3 and refactored into smaller helpers.
- The subset of `gprim` and `ypy` used by the package has been internalized into local `sldp` modules.
- The main `sldp` CLI is analysis-only by default and now supports an explicit `--preprocess` flag that builds only missing artifacts.
- Current equivalence coverage is anchored on the canonical tiny fixture and final `.gwresults` output. Intermediate artifact equivalence checks are a logical next step.

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
- There is now a focused automated test suite, but benchmark coverage has not been added yet.

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

## Progress Summary

- Phase 0: complete
- Phase 1: complete for the tiny deterministic fixture and captured baseline outputs
- Phase 2: substantially complete for current regression coverage, with room to expand equivalence checks to intermediate artifacts and additional supported workflows
- Phase 3: complete for the refreshed core workflows under current validation
- Phase 4: substantially complete for the main modules and CLI structure
- Phase 5: substantially complete for the `gprim` and `ypy` surfaces used by the package
- Phase 6: not started in a benchmarked way
- Phase 7: not started
- Phase 8: partially complete for local tooling and docs, but CI and fuller contributor guidance remain open

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

### Current Status

- Completed. The baseline package location, environment snapshot, validation policy, and known baseline quirks are documented in `BASELINE_ENVIRONMENT.md`.

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

### Current Status

- Completed for the initial fixture. `tests/fixtures/phase1_tiny/` provides a deterministic two-chromosome synthetic dataset, fixture-generation docs, and captured baseline artifacts.

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

### Current Status

- Substantially complete.
- The repository now has `pytest` coverage split into `tests/equivalence/` and `tests/current/`.
- The current equivalence suite verifies final `.gwresults` parity on the canonical fixture.
- Next expansion for this phase: add intermediate artifact comparisons for `.pss.gz`, `.RV.gz`, `.info`, `R.npz`, and `R2.npz`, plus a few more supported workflow variants where useful.

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

### Current Status

- Completed for the currently validated workflows. The major Python 3 and pandas incompatibilities called out above have been fixed, and the refreshed package passes fixture-based validation and focused tests.

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

### Current Status

- Substantially complete for `sldp`, `preprocessannot`, `preprocesspheno`, and `preprocessrefpanel`.
- Entry points now build parsers separately from execution, and the core modules are decomposed into smaller helpers.
- The main behavioral refinement made after the initial refactor is explicit opt-in preprocessing via `--preprocess`, preserving analysis-only default behavior.

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

### Current Status

- Substantially complete for the package code paths currently under test. Local modules now cover the `gprim` and `ypy` surfaces used by SLDP, and package metadata has been updated accordingly.

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

### Current Status

- Not yet started in a measured way. Some cleanup may have produced incidental wins, but there is no benchmark-backed performance work yet.

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

### Current Status

- Not started.

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

### Current Status

- Partially complete.
- `pyproject.toml`, local tooling, fixture docs, and baseline notes are in place.
- Remaining work includes CI, fuller contributor workflow documentation, and keeping README/testing docs aligned as equivalence coverage expands.

## Recommended Resume Point

The next concrete work item should be expanding `tests/equivalence/` beyond final `.gwresults` comparison.

Recommended next steps:

1. Add artifact-level equivalence tests for:
   - processed phenotype `info` and `.pss.gz`
   - processed annotation `.info` and `.RV.gz`
   - reference-panel `R.npz` and `R2.npz`
2. Decide whether to generate those expectations from frozen baseline artifacts only or from a mix of frozen artifacts and fresh baseline-runtime checks.
3. Add one or two more supported equivalence paths only if they materially increase confidence without encoding baseline bugs.

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
