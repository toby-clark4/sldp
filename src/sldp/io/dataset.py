from collections.abc import Iterator
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed

from sldp.utils import memo


def _chunked_indices(length: int, size: int) -> Iterator[list[int]]:
    """Yield contiguous index chunks of at most `size` elements."""

    for start in range(0, length, size):
        yield list(range(start, min(start + size, length)))


class Dataset:
    """Wrapper around chromosome-split PLINK genotype files used by SLDP."""

    def __init__(self, bfile_chr: str, assembly: str = "hg19") -> None:
        self.bfile_chr = bfile_chr
        self.assembly = assembly

    @property
    def path(self) -> str:
        """Return the parent directory of the chromosome file stem."""

        return f"{Path(self.bfile_chr).parent}/"

    @memo.memoized
    def data(self, chrnum: int) -> Bed:
        """Return the PLINK bed reader for a chromosome."""

        return Bed(self.bfile(chrnum), count_A1=False)

    def stdX(self, chrnum: int, bounds: tuple[int, int]) -> np.ndarray:
        """Return standardized genotypes for a contiguous SNP interval."""

        return self.stdX_it(chrnum, range(bounds[0], bounds[1]))

    def stdX_it(self, chrnum: int, indices: range | list[int]) -> np.ndarray:
        """Return standardized genotypes for specific SNP indices."""

        genotypes = self.data(chrnum)[:, indices].read()
        genotypes.standardize()
        return genotypes.val

    def bfile(self, chrnum: int) -> str:
        """Return the chromosome-specific PLINK file stem."""

        return f"{self.bfile_chr}{chrnum}"

    def bimfile(self, chrnum: int) -> str:
        """Return the `.bim` path for a chromosome."""

        return self.bfile(chrnum) + ".bim"

    def frq_file(self, chrnum: int) -> str:
        """Return the `.frq` path for a chromosome."""

        return self.bfile(chrnum) + ".frq"

    @memo.memoized
    def frq_df(self, chrnum: int) -> pd.DataFrame:
        """Load the chromosome frequency dataframe."""

        return pd.read_csv(self.frq_file(chrnum), sep=r"\s+", header=0)

    @memo.memoized
    def bim_df(self, chrnum: int) -> pd.DataFrame:
        """Load the chromosome BIM metadata dataframe."""

        return pd.read_csv(
            self.bimfile(chrnum),
            names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
            sep="\t",
        )

    def M(self, chrnum: int) -> int:
        """Return the number of SNPs on a chromosome."""

        return len(self.bim_df(chrnum))

    def totalM(self, chroms: list[int] | None = None) -> int:
        """Return the total number of SNPs across requested chromosomes."""

        if chroms is None:
            chroms = [c for c in range(1, 23) if Path(self.bimfile(c)).exists()]
        return sum(self.M(chrom) for chrom in chroms)

    def N(self) -> int:
        """Return the number of individuals in the reference panel."""

        chrom = min(c for c in range(1, 23) if Path(self.bimfile(c)).exists())
        return self.data(chrom).iid_count

    def block_data(
        self,
        ldblocks: pd.DataFrame,
        c: int,
        meta: pd.DataFrame | None = None,
        chunksize: int = 15,
        genos: bool = True,
        verbose: int = 2,
    ) -> Iterator[tuple[pd.Series, np.ndarray | None, pd.DataFrame | None, pd.Index]]:
        """Yield per-LD-block metadata and optionally standardized genotypes."""

        chr_blocks = ldblocks[ldblocks.chr == f"chr{c}"]
        snps = self.bim_df(c)

        t0 = time.time()
        for block_nums in _chunked_indices(len(chr_blocks), chunksize):
            chunk_blocks = chr_blocks.iloc[block_nums]
            blockstarts_ind = np.searchsorted(snps.BP.values, chunk_blocks.start.values)
            blockends_ind = np.searchsorted(snps.BP.values, chunk_blocks.end.values)
            if verbose >= 1 and len(blockstarts_ind) > 0:
                print(f"{time.time() - t0} : chr {c} snps {blockstarts_ind[0]} - {blockends_ind[-1]}")

            if genos:
                Xchunk = self.stdX(c, (blockstarts_ind[0], blockends_ind[-1]))
                print("read in chunk")
            else:
                Xchunk = None

            metachunk = meta.iloc[blockstarts_ind[0] : blockends_ind[-1]] if meta is not None else None

            blockends_ind = blockends_ind - blockstarts_ind[0]
            blockstarts_ind = blockstarts_ind - blockstarts_ind[0]
            for i, start_ind, end_ind in zip(chunk_blocks.index, blockstarts_ind, blockends_ind):
                if verbose >= 2:
                    print(f"{time.time() - t0} : processing ld block {i} , {end_ind - start_ind} snps")
                X = Xchunk[:, start_ind:end_ind] if genos and Xchunk is not None else None
                metablock = metachunk.iloc[start_ind:end_ind] if metachunk is not None else None
                index = metachunk.iloc[start_ind:end_ind].index if metachunk is not None else snps.iloc[start_ind:end_ind].index
                yield chr_blocks.loc[i], X, metablock, index
