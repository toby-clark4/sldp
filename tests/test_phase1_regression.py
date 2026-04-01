from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase1_tiny"
BASELINE_RESULTS = FIXTURE_ROOT / "generated" / "results" / "toy.gwresults"


def _rewrite_config_paths(
    config_path: Path, source_fixture: Path, target_fixture: Path
) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_prefix = str(source_fixture)
    source_relative_prefix = str(source_fixture.relative_to(ROOT))
    target_prefix = str(target_fixture)

    def rewrite_value(value: str) -> str:
        if value.startswith(source_prefix):
            return value.replace(source_prefix, target_prefix, 1)
        if value.startswith(source_relative_prefix):
            return value.replace(source_relative_prefix, target_prefix, 1)
        return value

    rewritten = {
        key: rewrite_value(value) if isinstance(value, str) else value
        for key, value in config.items()
    }
    config_path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")


def _run_repo_module(module: str, args: list[str], env: dict[str, str]) -> None:
    command = [sys.executable, "-m", module, *args]
    subprocess.run(command, check=True, cwd=ROOT, env=env)


class TestPhase1Regression:
    def test_phase1_fixture_matches_baseline_gwresults(self, tmp_path: Path) -> None:
        fixture_copy = tmp_path / "phase1_tiny"
        shutil.copytree(FIXTURE_ROOT, fixture_copy)

        config_path = fixture_copy / "fixture_config.json"
        _rewrite_config_paths(config_path, FIXTURE_ROOT, fixture_copy)

        expected = pd.read_csv(BASELINE_RESULTS, sep="\t")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")

        _run_repo_module(
            "sldp.preprocessrefpanel",
            ["--config", str(config_path), "--chroms", "1", "2"],
            env,
        )
        _run_repo_module(
            "sldp.preprocessannot",
            [
                "--config",
                str(config_path),
                "--sannot-chr",
                str(fixture_copy / "data" / "annot" / "toy_annot."),
                str(fixture_copy / "data" / "annot" / "toy_background."),
                "--chroms",
                "1",
                "2",
            ],
            env,
        )
        _run_repo_module(
            "sldp.preprocesspheno",
            [
                "--config",
                str(config_path),
                "--sumstats-stem",
                str(fixture_copy / "data" / "sumstats" / "toy"),
                "--chroms",
                "1",
                "2",
            ],
            env,
        )
        _run_repo_module(
            "sldp.sldp",
            [
                "--config",
                str(config_path),
                "--outfile-stem",
                str(fixture_copy / "generated" / "results" / "toy"),
                "--pss-chr",
                f"{fixture_copy / 'data' / 'sumstats' / 'toy.KG3.95'}/",
                "--sannot-chr",
                str(fixture_copy / "data" / "annot" / "toy_annot."),
                "--background-sannot-chr",
                str(fixture_copy / "data" / "annot" / "toy_background."),
                "--chroms",
                "1",
                "2",
                "--seed",
                "123",
                "-bothp",
            ],
            env,
        )

        actual = pd.read_csv(
            fixture_copy / "generated" / "results" / "toy.gwresults", sep="\t"
        )

        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )
