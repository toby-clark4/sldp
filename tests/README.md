# Test Suite Layout

This test suite is split by intent.

## `tests/equivalence/`

Tests in this directory compare the refreshed repository against the previous installed baseline implementation or frozen baseline artifacts derived from it.

These tests should cover only workflows and outputs that are part of the intended compatibility contract.

Known baseline bugs should not be treated as required behavior here.

There are two equivalence modes:

- fast frozen-artifact regression: compares repo output against checked-in baseline artifacts and runs in normal `pytest`
- live baseline parity: compares repo output against the installed `sldp` package in the `sldp` conda environment and is marked with `@pytest.mark.live_baseline`

Run the live installed-package parity checks explicitly with:

```bash
conda run -n sldp python -m pytest -m live_baseline tests/equivalence
```

## `tests/current/`

Tests in this directory validate the current repository on its own terms.

This includes:

- unit tests for extracted helpers and local compatibility modules
- integration-style tests for preprocessing helpers and main-command behavior
- tests for intentional fixes and enhancements such as explicit opt-in preprocessing

## `tests/fixtures/`

Fixtures used by both suites live here.

The canonical tiny baseline-comparison fixture is `tests/fixtures/phase1_tiny/`.
