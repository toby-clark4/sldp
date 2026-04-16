from __future__ import annotations

from pathlib import Path

from sldp.io.workflow_io import load_ldblocks, load_print_snps


class TestLoadLdblocks:
    def test_load_ldblocks_removes_mhc_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "ld_blocks.bed"
        path.write_text("chr6\t25684500\t25684600\nchr1\t10\t20\n", encoding="utf-8")

        ldblocks = load_ldblocks(str(path))

        assert ldblocks.shape == (1, 3)
        assert ldblocks.iloc[0].chr == "chr1"

    def test_load_ldblocks_can_keep_mhc_blocks(self, tmp_path: Path) -> None:
        path = tmp_path / "ld_blocks.bed"
        path.write_text("chr6\t25684500\t25684600\nchr1\t10\t20\n", encoding="utf-8")

        ldblocks = load_ldblocks(str(path), remove_mhc=False)

        assert ldblocks.shape == (2, 3)
        assert ldblocks.iloc[0].chr == "chr6"


class TestLoadPrintSnps:
    def test_load_print_snps_marks_all_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "print_snps.txt"
        path.write_text("rs1\nrs2\n", encoding="utf-8")

        print_snps = load_print_snps(str(path))

        assert print_snps.columns.tolist() == ["SNP", "printsnp"]
        assert print_snps["printsnp"].all()
