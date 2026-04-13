from __future__ import annotations

from pathlib import Path

import pytest

from tests.equivalence.helpers import (
    assert_phase1_outputs_equal,
    copy_phase1_fixture,
    run_baseline_phase1_pipeline,
    run_repo_phase1_pipeline,
)


@pytest.mark.live_baseline
class TestPhase1LiveBaseline:
    def test_phase1_fixture_matches_installed_baseline_outputs(self, tmp_path: Path) -> None:
        baseline_fixture, baseline_config = copy_phase1_fixture(tmp_path, "installed_baseline")
        repo_fixture, repo_config = copy_phase1_fixture(tmp_path, "repo_output")

        run_baseline_phase1_pipeline(baseline_fixture, baseline_config)
        run_repo_phase1_pipeline(repo_fixture, repo_config)

        assert_phase1_outputs_equal(repo_fixture, baseline_fixture)
