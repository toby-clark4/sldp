from __future__ import annotations

import numpy as np

from sldp.dataset import Dataset


FIXTURE_BFILE = "tests/fixtures/phase1_tiny/data/refpanel/toy_ref."


class TestDataset:
    def test_bim_df_and_frq_df_load_fixture_metadata(self) -> None:
        dataset = Dataset(FIXTURE_BFILE)

        bim = dataset.bim_df(1)
        frq = dataset.frq_df(1)

        assert bim.columns.tolist() == ["CHR", "SNP", "CM", "BP", "A1", "A2"]
        assert frq.columns.tolist() == ["CHR", "SNP", "A1", "A2", "MAF", "NCHROBS"]
        assert bim.iloc[0].SNP == "rs1"

    def test_stdX_returns_standardized_matrix(self) -> None:
        dataset = Dataset(FIXTURE_BFILE)

        matrix = dataset.stdX(1, (0, 3))

        assert matrix.shape == (4, 3)
        np.testing.assert_allclose(matrix.mean(axis=0), np.zeros(3), atol=1e-10)

    def test_block_data_yields_expected_blocks_without_genotypes(self) -> None:
        dataset = Dataset(FIXTURE_BFILE)
        ldblocks = dataset.bim_df(1).assign(chr="chr1")[["chr", "BP"]].copy()
        ldblocks["start"] = [50, 350, 999, 999, 999, 999]
        ldblocks["end"] = [350, 700, 1000, 1000, 1000, 1000]
        ldblocks = ldblocks.iloc[:2][["chr", "start", "end"]]
        meta = dataset.bim_df(1)

        blocks = list(
            dataset.block_data(ldblocks, 1, meta=meta, genos=False, verbose=0)
        )

        assert len(blocks) == 2
        first_block, _, first_meta, first_index = blocks[0]
        assert first_block["start"] == 50
        assert first_meta is not None
        assert first_meta["SNP"].tolist() == ["rs1", "rs2", "rs3"]
        assert list(first_index) == [0, 1, 2]
