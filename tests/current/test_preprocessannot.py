from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sldp.io.annotation import Annotation
from sldp.io.dataset import Dataset
from sldp.preprocessannot import (
    _compute_rv_values,
    _load_ldblocks,
    _load_print_snps,
    _prepare_chromosome_annotation,
    _write_annotation_info,
    _write_rv_output,
    run,
)


FIXTURE_ROOT = Path("tests/fixtures/phase1_tiny")
REFPANEL = Dataset(str(FIXTURE_ROOT / "data" / "refpanel" / "toy_ref."))
ANNOT = Annotation(str(FIXTURE_ROOT / "data" / "annot" / "toy_annot."))


class TestPreprocessAnnotHelpers:
    def test_prepare_chromosome_annotation_reconciles_fixture_annotation(self) -> None:
        print_snps = _load_print_snps(str(FIXTURE_ROOT / "data" / "print_snps.txt"))

        snps, names, result_names = _prepare_chromosome_annotation(REFPANEL, ANNOT, 1, print_snps, alpha=-1)

        assert names == ["annot_signal"]
        assert result_names == ["annot_signal.R"]
        assert snps["printsnp"].sum() == 4
        assert "MAF" in snps.columns

    def test_write_annotation_info_creates_expected_summary_file(self, tmp_path: Path) -> None:
        print_snps = _load_print_snps(str(FIXTURE_ROOT / "data" / "print_snps.txt"))
        snps, names, _ = _prepare_chromosome_annotation(REFPANEL, ANNOT, 1, print_snps, alpha=-1)
        output = tmp_path / "annot.info"

        _write_annotation_info(snps, names, str(output))

        info = pd.read_csv(output, sep="\t", index_col=0)
        assert info.loc["annot_signal", "M"] == len(snps)
        assert info.loc["annot_signal", "supp"] > 0

    def test_compute_rv_values_and_write_output_produce_nonempty_results(self, tmp_path: Path) -> None:
        print_snps = _load_print_snps(str(FIXTURE_ROOT / "data" / "print_snps.txt"))
        ldblocks = _load_ldblocks(str(FIXTURE_ROOT / "data" / "ld_blocks.bed"))
        snps, names, result_names = _prepare_chromosome_annotation(REFPANEL, ANNOT, 1, print_snps, alpha=-1)

        _compute_rv_values(REFPANEL, ldblocks, 1, snps, names, result_names)
        output = tmp_path / "toy.RV.gz"
        _write_rv_output(snps, names, result_names, output)

        written = pd.read_csv(output, sep="\t", compression="gzip")
        assert written.columns.tolist() == ["SNP", "A1", "A2", "annot_signal", "annot_signal.R"]
        assert len(written) == 4
        assert written["annot_signal.R"].abs().sum() > 0

    def test_run_dispatches_annotation_chromosome_tasks_in_parallel_mode(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        monkeypatch.setattr("sldp.preprocessannot._load_ldblocks", lambda path: _load_ldblocks(str(FIXTURE_ROOT / "data" / "ld_blocks.bed")))
        monkeypatch.setattr("sldp.preprocessannot._load_print_snps", lambda path: _load_print_snps(str(FIXTURE_ROOT / "data" / "print_snps.txt")))

        def fake_execute_tasks(tasks, worker_fn, num_proc: int):
            del worker_fn
            captured["tasks"] = list(tasks)
            captured["num_proc"] = num_proc
            return []

        monkeypatch.setattr("sldp.preprocessannot.execute_tasks", fake_execute_tasks)

        args = argparse.Namespace(
            bfile_chr=str(FIXTURE_ROOT / "data" / "refpanel" / "toy_ref."),
            sannot_chr=[str(FIXTURE_ROOT / "data" / "annot" / "toy_annot.")],
            ld_blocks=str(FIXTURE_ROOT / "data" / "ld_blocks.bed"),
            print_snps=str(FIXTURE_ROOT / "data" / "print_snps.txt"),
            chroms=[1, 2],
            num_proc=2,
            alpha=-1,
        )

        run(args)

        assert captured["num_proc"] == 2
        assert [(task.annot_stem, task.chrom) for task in captured["tasks"]] == [
            (str(FIXTURE_ROOT / "data" / "annot" / "toy_annot."), 1),
            (str(FIXTURE_ROOT / "data" / "annot" / "toy_annot."), 2),
        ]
