from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase1_tiny"
BASELINE_RESULTS = FIXTURE_ROOT / "generated" / "results" / "toy.gwresults"

PHASE1_OUTPUT_FILES = [
    "generated/results/toy.gwresults",
    "generated/svd/0.R.npz",
    "generated/svd/0.R2.npz",
    "generated/svd/1.R.npz",
    "generated/svd/1.R2.npz",
    "generated/svd/2.R.npz",
    "generated/svd/2.R2.npz",
    "generated/svd/3.R.npz",
    "generated/svd/3.R2.npz",
    "data/sumstats/toy.KG3.95/info",
    "data/sumstats/toy.KG3.95/1.pss.gz",
    "data/sumstats/toy.KG3.95/2.pss.gz",
    "data/annot/toy_annot.1.info",
    "data/annot/toy_annot.2.info",
    "data/annot/toy_annot.1.RV.gz",
    "data/annot/toy_annot.2.RV.gz",
    "data/annot/toy_background.1.info",
    "data/annot/toy_background.2.info",
    "data/annot/toy_background.1.RV.gz",
    "data/annot/toy_background.2.RV.gz",
]


def rewrite_config_paths(config_path: Path, source_fixture: Path, target_fixture: Path) -> None:
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

    rewritten = {key: rewrite_value(value) if isinstance(value, str) else value for key, value in config.items()}
    config_path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")


def copy_phase1_fixture(tmp_path: Path, name: str) -> tuple[Path, Path]:
    fixture_copy = tmp_path / name / "phase1_tiny"
    shutil.copytree(FIXTURE_ROOT, fixture_copy)
    config_path = fixture_copy / "fixture_config.json"
    rewrite_config_paths(config_path, FIXTURE_ROOT, fixture_copy)
    return fixture_copy, config_path


def run_command(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def repo_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def baseline_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def run_repo_module(module: str, args: list[str], env: dict[str, str] | None = None) -> None:
    run_command([sys.executable, "-m", module, *args], env=env or repo_env())


def run_baseline_command(command_name: str, args: list[str], env: dict[str, str] | None = None) -> None:
    run_command(["conda", "run", "-n", "sldp", command_name, *args], env=env or baseline_env())


def phase1_pipeline_args(fixture_root: Path, config_path: Path, *, num_proc: int | None = None) -> dict[str, list[str]]:
    preprocess_common = ["--config", str(config_path), "--chroms", "1", "2"]
    if num_proc is not None:
        preprocess_common.extend(["--num-proc", str(num_proc)])

    return {
        "preprocessrefpanel": [*preprocess_common],
        "preprocessannot": [
            *preprocess_common,
            "--sannot-chr",
            str(fixture_root / "data" / "annot" / "toy_annot."),
            str(fixture_root / "data" / "annot" / "toy_background."),
        ],
        "preprocesspheno": [
            *preprocess_common,
            "--sumstats-stem",
            str(fixture_root / "data" / "sumstats" / "toy"),
        ],
        "sldp": [
            "--config",
            str(config_path),
            "--outfile-stem",
            str(fixture_root / "generated" / "results" / "toy"),
            "--pss-chr",
            f"{fixture_root / 'data' / 'sumstats' / 'toy.KG3.95'}/",
            "--sannot-chr",
            str(fixture_root / "data" / "annot" / "toy_annot."),
            "--background-sannot-chr",
            str(fixture_root / "data" / "annot" / "toy_background."),
            "--chroms",
            "1",
            "2",
            "--seed",
            "123",
            "-bothp",
        ],
    }


def run_repo_phase1_pipeline(fixture_root: Path, config_path: Path, *, num_proc: int | None = None) -> None:
    args = phase1_pipeline_args(fixture_root, config_path, num_proc=num_proc)
    run_repo_module("sldp.preprocessrefpanel", args["preprocessrefpanel"])
    run_repo_module("sldp.preprocessannot", args["preprocessannot"])
    run_repo_module("sldp.preprocesspheno", args["preprocesspheno"])
    run_repo_module("sldp.sldp", args["sldp"])


def run_baseline_phase1_pipeline(fixture_root: Path, config_path: Path) -> None:
    args = phase1_pipeline_args(fixture_root, config_path)
    run_baseline_command("preprocessrefpanel", args["preprocessrefpanel"])
    run_baseline_command("preprocessannot", args["preprocessannot"])
    run_baseline_command("preprocesspheno", args["preprocesspheno"])
    run_baseline_command("sldp", args["sldp"])


def assert_tabular_equal(left_path: Path, right_path: Path) -> None:
    left = pd.read_csv(left_path, sep="\t")
    right = pd.read_csv(right_path, sep="\t")
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-9, atol=1e-9)


def assert_npz_equal(left_path: Path, right_path: Path) -> None:
    left = np.load(left_path)
    right = np.load(right_path)

    assert set(left.files) == set(right.files)
    for key in left.files:
        np.testing.assert_allclose(left[key], right[key], rtol=1e-9, atol=1e-9)


def assert_phase1_outputs_equal(left_fixture: Path, right_fixture: Path) -> None:
    for relative_path in PHASE1_OUTPUT_FILES:
        left_path = left_fixture / relative_path
        right_path = right_fixture / relative_path
        assert left_path.exists(), f"missing expected output: {left_path}"
        assert right_path.exists(), f"missing expected output: {right_path}"
        if left_path.suffix == ".npz":
            assert_npz_equal(left_path, right_path)
        else:
            assert_tabular_equal(left_path, right_path)
