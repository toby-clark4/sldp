from __future__ import annotations

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
