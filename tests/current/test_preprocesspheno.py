from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sldp.dataset import Dataset
from sldp.preprocesspheno import (
    _estimate_h2g,
    _filter_sumstats,
    _load_ldblocks,
    _load_print_snps,
    _process_chromosome,
    _read_ld_scores,
    _read_sumstats,
    _write_info_file,
)


FIXTURE_ROOT = Path("tests/fixtures/phase1_tiny")
REFPANEL = Dataset(str(FIXTURE_ROOT / "data" / "refpanel" / "toy_ref."))


class TestPreprocessPhenoHelpers:
    def test_read_sumstats_and_filter_sumstats_use_fixture_inputs(self) -> None:
        print_snps = _load_print_snps(str(FIXTURE_ROOT / "data" / "print_snps.txt"))
        sumstats = _read_sumstats(str(FIXTURE_ROOT / "data" / "sumstats" / "toy"))

        filtered = _filter_sumstats(sumstats, print_snps)

        assert len(sumstats) == 7
        assert len(filtered) == 7
        assert filtered["SNP"].tolist()[0] == "rs1"

    def test_read_ld_scores_and_estimate_h2g_work_on_fixture(self) -> None:
        print_snps = _load_print_snps(str(FIXTURE_ROOT / "data" / "print_snps.txt"))
        sumstats = _filter_sumstats(_read_sumstats(str(FIXTURE_ROOT / "data" / "sumstats" / "toy")), print_snps)
        ld_scores, read_m_file = _read_ld_scores(str(FIXTURE_ROOT / "data" / "ldscores."))
        ssld = pd.merge(sumstats, ld_scores, on="SNP", how="left")
        M = sum(read_m_file(FIXTURE_ROOT / "data" / f"ldscores.{chrom}.l2.M_5_50") for chrom in range(1, 23))

        h2g, sigma2g, meanchi2, k_value = _estimate_h2g(ssld, M)

        assert meanchi2 > 1.0
        assert sigma2g > 0
        assert h2g > 0
        assert k_value > 0

    def test_write_info_file_creates_metadata_file(self, tmp_path: Path) -> None:
        sumstats = _read_sumstats(str(FIXTURE_ROOT / "data" / "sumstats" / "toy"))

        _write_info_file(tmp_path, "toy", h2g=0.5, sigma2g=0.25, sumstats=sumstats)

        info = pd.read_csv(tmp_path / "info", sep="\t")
        assert info.loc[0, "pheno"] == "toy"
        assert info.loc[0, "h2g"] == 0.5

    def test_process_chromosome_generates_weighted_outputs(self) -> None:
        print_snps = _load_print_snps(str(FIXTURE_ROOT / "data" / "print_snps.txt"))
        sumstats = _filter_sumstats(_read_sumstats(str(FIXTURE_ROOT / "data" / "sumstats" / "toy")), print_snps)
        ldblocks = _load_ldblocks(str(FIXTURE_ROOT / "data" / "ld_blocks.bed"))

        snps = _process_chromosome(
            refpanel=REFPANEL,
            ldblocks=ldblocks,
            chrom=1,
            print_snps=print_snps,
            sumstats=sumstats,
            svd_stem=f"{FIXTURE_ROOT / 'generated' / 'svd'}/",
            sigma2g=0.0016723498888065237,
        )

        assert snps["printsnp"].sum() == 4
        assert snps["typed"].sum() == 4
        assert np.isfinite(snps.loc[snps["printsnp"], "Winv_ahat_I"]).all()
        assert np.isfinite(snps.loc[snps["printsnp"], "Winv_ahat_h"]).all()
