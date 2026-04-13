from __future__ import annotations

from pathlib import Path

import pandas as pd

from sldp.annotation import Annotation
from sldp.dataset import Dataset
from sldp.preprocessannot import (
    _compute_rv_values,
    _load_ldblocks,
    _load_print_snps,
    _prepare_chromosome_annotation,
    _write_annotation_info,
    _write_rv_output,
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
