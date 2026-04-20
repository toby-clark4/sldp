from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sldp.io.dataset import Dataset
from sldp.preprocessrefpanel import (
    _best_svd,
    _load_ldblocks,
    _load_print_snps,
    _prepare_chromosome_snps,
    _save_block_svds,
    _svd_output_path,
    run,
)


FIXTURE_ROOT = Path("tests/fixtures/phase1_tiny")
FIXTURE_BFILE = FIXTURE_ROOT / "data" / "refpanel" / "toy_ref."


class TestPreprocessRefpanelHelpers:
    def test_load_ldblocks_reads_fixture_blocks(self) -> None:
        ldblocks = _load_ldblocks(str(FIXTURE_ROOT / "data" / "ld_blocks.bed"))

        assert ldblocks.shape == (4, 3)
        assert ldblocks.iloc[0].to_dict() == {"chr": "chr1", "start": 50, "end": 350}

    def test_load_print_snps_marks_output_snps(self) -> None:
        print_snps = _load_print_snps(str(FIXTURE_ROOT / "data" / "print_snps.txt"))

        assert print_snps["printsnp"].all()
        assert print_snps.iloc[0].SNP == "rs1"

    def test_prepare_chromosome_snps_marks_fixture_print_subset(self) -> None:
        dataset = Dataset(str(FIXTURE_BFILE))
        print_snps = _load_print_snps(str(FIXTURE_ROOT / "data" / "print_snps.txt"))

        snps = _prepare_chromosome_snps(dataset, 1, print_snps)

        assert snps["printsnp"].sum() == 4
        assert snps.loc[snps["printsnp"], "SNP"].tolist() == ["rs1", "rs2", "rs4", "rs5"]

    def test_best_svd_returns_nonnegative_spectrum(self) -> None:
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        u, singular_values = _best_svd(matrix)

        assert u.shape == (2, 2)
        assert singular_values.shape == (2,)
        assert np.all(singular_values >= 0)

    def test_save_block_svds_writes_r_and_r2_outputs(self, tmp_path: Path) -> None:
        x_print = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
        svd_stem = tmp_path / "svd" / "block_"
        svd_stem.parent.mkdir(parents=True, exist_ok=True)

        _save_block_svds(x_print, block_name=7, svd_stem=svd_stem, spectrum_percent=95, num_print_snps=2)

        r_path = _svd_output_path(svd_stem, 7, "R")
        r2_path = _svd_output_path(svd_stem, 7, "R2")
        assert r_path.exists()
        assert r2_path.exists()

        r = np.load(r_path)
        r2 = np.load(r2_path)
        assert set(r.files) == {"U", "svs"}
        assert set(r2.files) == {"U", "svs"}

    def test_run_dispatches_one_task_per_chromosome(self, monkeypatch, tmp_path: Path) -> None:
        captured: dict[str, object] = {}

        monkeypatch.setattr("sldp.preprocessrefpanel.fs.makedir_for_file", lambda path: None)
        monkeypatch.setattr(
            "sldp.preprocessrefpanel._load_ldblocks",
            lambda path: np.array([path]) if False else _load_ldblocks(str(FIXTURE_ROOT / "data" / "ld_blocks.bed")),
        )
        monkeypatch.setattr("sldp.preprocessrefpanel._load_print_snps", lambda path: _load_print_snps(str(FIXTURE_ROOT / "data" / "print_snps.txt")))

        def fake_execute_tasks(tasks, worker_fn, num_proc: int):
            del worker_fn
            captured["tasks"] = list(tasks)
            captured["num_proc"] = num_proc
            return []

        monkeypatch.setattr("sldp.preprocessrefpanel.execute_tasks", fake_execute_tasks)

        args = argparse.Namespace(
            bfile_chr=str(FIXTURE_BFILE),
            svd_stem=str(tmp_path / "svd" / "block_"),
            ld_blocks=str(FIXTURE_ROOT / "data" / "ld_blocks.bed"),
            print_snps=str(FIXTURE_ROOT / "data" / "print_snps.txt"),
            chroms=[1, 2],
            num_proc=2,
            spectrum_percent=95,
        )

        run(args)

        assert captured["num_proc"] == 2
        assert [task.chrom for task in captured["tasks"]] == [1, 2]
