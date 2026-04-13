from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.equivalence.helpers import BASELINE_RESULTS, copy_phase1_fixture, run_repo_phase1_pipeline


class TestPhase1Regression:
    def test_phase1_fixture_matches_baseline_gwresults(self, tmp_path: Path) -> None:
        fixture_copy, config_path = copy_phase1_fixture(tmp_path, "repo_regression")

        expected = pd.read_csv(BASELINE_RESULTS, sep="\t")

        run_repo_phase1_pipeline(fixture_copy, config_path)

        actual = pd.read_csv(fixture_copy / "generated" / "results" / "toy.gwresults", sep="\t")

        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )
