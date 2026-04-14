from functools import reduce
import itertools

import numpy as np
import pandas as pd

from sldp import fs, memo


COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
BASES = tuple(COMPLEMENT.keys())
STRAND_AMBIGUOUS = {"".join(x): x[0] == COMPLEMENT[x[1]] for x in itertools.product(BASES, BASES) if x[0] != x[1]}
VALID_SNPS = {x for x in map("".join, itertools.product(BASES, BASES)) if x[0] != x[1] and not STRAND_AMBIGUOUS[x]}
MATCH_ALLELES = {
    x
    for x in map("".join, itertools.product(VALID_SNPS, VALID_SNPS))
    if ((x[0] == x[2]) and (x[1] == x[3]))
    or ((x[0] == COMPLEMENT[x[2]]) and (x[1] == COMPLEMENT[x[3]]))
    or ((x[0] == x[3]) and (x[1] == x[2]))
    or ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))
}
FLIP_ALLELES = {x for x in MATCH_ALLELES if ((x[0] == x[3]) and (x[1] == x[2])) or ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))}
_METADATA_COLUMNS = {"SNP", "CHR", "CM", "BP", "A1", "A2"}


def smart_merge(
    x: pd.DataFrame | list[pd.DataFrame],
    y: pd.DataFrame | list[pd.DataFrame] | None = None,
    how: str = "inner",
    fail_if_nonmatching: bool = False,
    drop_from_y: list[str] | None = None,
    key: str = "SNP",
) -> pd.DataFrame:
    """Merge dataframes efficiently when keyed rows already match in order."""

    drop_from_y = drop_from_y or []
    y_frames: list[pd.DataFrame]
    if y is None:
        y_frames = []
    elif isinstance(y, pd.DataFrame):
        y_frames = [y]
    else:
        y_frames = list(y)

    if not isinstance(x, pd.DataFrame):
        x_frames = list(x)
        y_frames = x_frames[1:] + y_frames
        x = x_frames[0]

    matching = True
    cleaned_frames: list[pd.DataFrame] = []
    for frame in y_frames:
        cleaned = frame.drop(columns=drop_from_y, errors="ignore")
        cleaned_frames.append(cleaned)
        if len(x) != len(cleaned) or (x[key].values != cleaned[key].values).any():
            matching = False

    x = x.reset_index(drop=True)
    if matching:
        return pd.concat(
            [x] + [frame.reset_index(drop=True).drop(columns=key) for frame in cleaned_frames],
            axis=1,
        )

    if fail_if_nonmatching:
        raise ValueError("smart_merge found nonmatching keyed rows")

    result = x
    for frame in cleaned_frames:
        result = pd.merge(result, frame.reset_index(drop=True), how=how, on=key)
    return result


def reconciled_to(
    ref: pd.DataFrame,
    df: pd.DataFrame,
    colnames: list[str],
    othercolnames: list[str] | None = None,
    signed: bool = True,
    missing_val: float | int = 0,
    key: str = "SNP",
) -> pd.DataFrame:
    """Align annotation or summary-stat alleles to a reference SNP dataframe."""

    othercolnames = othercolnames or []
    result = smart_merge(
        ref,
        df[[key, "A1", "A2", *colnames, *othercolnames]].rename(columns={"A1": "A1_df", "A2": "A2_df"}),
        how="left",
        key=key,
    )
    print(len(result), "snps after merging")
    if len(result) != len(ref):
        print("WARNING: merged data frame is not the same length as reference data frame")
        print("   check for duplicate snps in one of the two dataframes")

    missing = result.A1_df.isnull()
    print("of", len(result), "snps in merge,", missing.sum(), "were missing in df")
    result.loc[missing, colnames] = missing_val
    result.loc[missing, "A1_df"] = "-"
    result.loc[missing, "A2_df"] = "-"

    a1234 = (result.A1 + result.A2 + result.A1_df + result.A2_df).str.upper()
    match = ~missing & a1234.isin(MATCH_ALLELES)
    n_mismatch = (~missing & ~match).sum()
    print(
        "of",
        (~missing).sum(),
        "remaining snps,",
        n_mismatch,
        "are a) present in ref and df, b) do not have matching sets of alleles that are both valid,",
    )
    result.loc[~missing & ~match, colnames] = missing_val

    if signed:
        flip = match & a1234.isin(FLIP_ALLELES)
        n_flip = flip.sum()
        print(
            "of the remaining",
            match.sum(),
            "snps,",
            n_flip,
            "snps need flipping and",
            (match & ~flip).sum(),
            "snps matched and did not need flipping",
        )
        result.loc[flip, colnames] *= -1

    return result.drop(columns=["A1_df", "A2_df"])


class Annotation:
    """Wrapper around SLDP annotation file stems and their associated tabular outputs."""

    def __init__(self, stem_chr: str, signed: bool = True) -> None:
        self.stem_chr = stem_chr
        self.signed = signed

    def filestem(self, chrnum: int | str = "", mkdir: bool = False) -> str:
        """Return the chromosome-specific path stem for this annotation."""

        fname = f"{self.stem_chr}{chrnum}"
        if mkdir:
            fs.makedir_for_file(fname)
        return fname

    def annot_filename(self, chrnum: int) -> str:
        return self.filestem(chrnum) + ".annot.gz"

    def sannot_filename(self, chrnum: int, mkdir: bool = False) -> str:
        return self.filestem(chrnum, mkdir) + ".sannot.gz"

    def info_filename(self, chrnum: int) -> str:
        return self.filestem(chrnum) + ".info"

    def ldscores_filename(self, chrnum: int) -> str:
        return self.filestem(chrnum) + ".l2.ldscore.gz"

    def rv_filename(self, chrnum: int) -> str:
        return self.filestem(chrnum) + ".RV.gz"

    def RV_filename(self, chrnum: int, full: bool = False) -> str:
        del full
        return self.rv_filename(chrnum)

    def info_df(self, chrs: int | list[int] | range = range(1, 23)) -> pd.DataFrame:
        """Load per-annotation summary info for one or more chromosomes."""

        if isinstance(chrs, int):
            return pd.read_csv(self.info_filename(chrs), sep="\t", index_col=0)
        return reduce(lambda x, y: x + y, [self.info_df(chrom) for chrom in chrs])

    @memo.memoized
    def annot_df(self, chrnum: int) -> pd.DataFrame:
        """Load a legacy `.annot.gz` dataframe."""

        df = pd.read_csv(self.annot_filename(chrnum), compression="gzip", header=0, sep="\t")
        return df.astype(dtype={name: float for name in self.names(chrnum)})

    @memo.memoized
    def sannot_df(self, chrnum: int) -> pd.DataFrame:
        """Load a signed-annotation dataframe."""

        df = pd.read_csv(self.sannot_filename(chrnum), compression="gzip", header=0, sep="\t")
        return df.astype(dtype={name: float for name in self.names(chrnum)})

    @memo.memoized
    def RV_df(self, chrnum: int) -> pd.DataFrame:
        """Load a postprocessed RV dataframe."""

        return pd.read_csv(self.rv_filename(chrnum), sep="\t")

    @memo.memoized
    def names(self, chrnum: int, RV: bool = False) -> list[str]:
        """Return annotation value column names, excluding SNP metadata columns."""

        filename = self.rv_filename(chrnum) if RV else self.sannot_filename(chrnum)
        temp = pd.read_csv(filename, nrows=1, sep=r"\s+")
        return [column for column in temp.columns.values if column not in _METADATA_COLUMNS]

    def total_sqnorms(self, chrs: int | list[int] | range) -> np.ndarray:
        """Return per-annotation squared norms summed over chromosomes."""

        return self.info_df(chrs)["sqnorm"].values

    def total_sizes(self, chrs: int | list[int] | range) -> np.ndarray:
        """Return per-annotation support sizes summed over chromosomes."""

        return self.info_df(chrs)["supp"].values
