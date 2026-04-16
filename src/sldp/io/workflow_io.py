from __future__ import annotations

import pandas as pd


MHC_BP_RANGE = (25684587, 35455756)


def load_ldblocks(path: str, *, remove_mhc: bool = True) -> pd.DataFrame:
    """Load LD blocks, optionally excluding blocks that overlap the MHC region."""

    ldblocks = pd.read_csv(path, sep=r"\s+", header=None, names=["chr", "start", "end"])
    if not remove_mhc:
        return ldblocks

    mhc_start, mhc_end = MHC_BP_RANGE
    mhcblocks = (ldblocks.chr == "chr6") & (ldblocks.end > mhc_start) & (ldblocks.start < mhc_end)
    return ldblocks[~mhcblocks]


def load_print_snps(path: str) -> pd.DataFrame:
    """Load the set of SNPs retained in printed or processed outputs."""

    print_snps = pd.read_csv(path, header=None, names=["SNP"])
    print_snps["printsnp"] = True
    return print_snps
