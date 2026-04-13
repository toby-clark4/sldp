import argparse
import gc
import gzip
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

import sldp.annotation as ga
import sldp.config as config
import sldp.dataset as gd
import sldp.fs as fs
import sldp.memo as memo
import sldp.pretty as pretty
import sldp.weights as weights
from sldp.workflow_io import load_ldblocks as _load_ldblocks
from sldp.workflow_io import load_print_snps as _load_print_snps


def _read_sumstats(sumstats_stem: str) -> pd.DataFrame:
    """Load summary statistics and apply the base missing-value filter."""

    print("reading sumstats", sumstats_stem)
    sumstats = pd.read_csv(f"{sumstats_stem}.sumstats.gz", sep="\t")
    sumstats = sumstats[sumstats.Z.notnull() & sumstats.N.notnull()]
    print(
        "{} snps, {}-{} individuals (avg: {})".format(
            len(sumstats),
            np.min(sumstats.N),
            np.max(sumstats.N),
            np.mean(sumstats.N),
        )
    )
    return sumstats


def _filter_sumstats(sumstats: pd.DataFrame, print_snps: pd.DataFrame) -> pd.DataFrame:
    """Drop unsupported variants and restrict to typed SNPs."""

    sumstats["is_monoallelic"] = (sumstats.A1.str.len() == 1) & (sumstats.A2.str.len() == 1)
    n_removed = (~sumstats.is_monoallelic).sum()
    sumstats = sumstats[sumstats.is_monoallelic].drop(columns=["is_monoallelic"])
    print("{} non-monoallelic variants removed".format(n_removed))

    n_dups = sumstats.SNP.duplicated().sum()
    if n_dups > 0:
        sumstats = sumstats[~sumstats.SNP.duplicated(keep="first")]
        print("{} duplicate SNP IDs removed".format(n_dups))

    sumstats = pd.merge(sumstats, print_snps[["SNP"]], on="SNP", how="inner")
    print(len(sumstats), "snps typed")
    return sumstats


def _read_ld_scores(ldscores_chr: str) -> tuple[pd.DataFrame, Callable[[str | Path], int]]:
    """Load LD score tables and return an M-file reader helper."""

    print("reading in ld scores")
    ld_frames = [pd.read_csv(f"{ldscores_chr}{c}.l2.ldscore.gz", sep=r"\s+") for c in range(1, 23)]
    ld_scores = pd.concat([frame for frame in ld_frames if not frame.empty], axis=0)

    def read_m_file(path: str | Path) -> int:
        with Path(path).open() as handle:
            return int(next(handle))

    return ld_scores, read_m_file


def _estimate_h2g(ssld: pd.DataFrame, M: int) -> tuple[float, float, float, float]:
    """Estimate trait heritability from LD scores and Z-scores."""

    ssld_valid = ssld[ssld.L2.notnull()]
    if len(ssld_valid) == 0:
        raise ValueError("No SNPs with valid LD scores found")
    meanchi2 = (ssld_valid.Z**2).mean()
    meanNl2 = (ssld_valid.N * ssld_valid.L2).mean()
    if meanNl2 == 0 or np.isnan(meanNl2):
        raise ValueError("Mean N*L2 is zero or NaN - cannot estimate heritability")
    sigma2g = (meanchi2 - 1) / meanNl2
    h2g = sigma2g * M
    K = M / meanNl2
    return h2g, sigma2g, meanchi2, K


def _write_info_file(dirname: Path, sumstats_stem: str, h2g: float, sigma2g: float, sumstats: pd.DataFrame) -> None:
    """Write trait-level metadata for processed phenotype inputs."""

    print("writing info file")
    info = pd.DataFrame(
        [
            {
                "pheno": sumstats_stem.split("/")[-1],
                "h2g": h2g,
                "sigma2g": sigma2g,
                "Nbar": sumstats.N.mean(),
            }
        ]
    )
    info.to_csv(dirname / "info", sep="\t", index=False)


def _process_chromosome(
    refpanel: gd.Dataset,
    ldblocks: pd.DataFrame,
    chrom: int,
    print_snps: pd.DataFrame,
    sumstats: pd.DataFrame,
    svd_stem: str,
    sigma2g: float,
) -> pd.DataFrame:
    """Process one chromosome into weighted per-SNP phenotype inputs."""

    snps = refpanel.bim_df(chrom)
    snps = pd.merge(snps, print_snps, on="SNP", how="left")
    snps["printsnp"] = snps.printsnp.notnull()
    print(len(snps), "snps in refpanel", len(snps.columns), "columns, including metadata")

    print("reconciling")
    snps = ga.reconciled_to(snps, sumstats, ["Z"], othercolnames=["N"], missing_val=np.nan)
    snps["typed"] = snps.Z.notnull()
    snps["ahat"] = snps.Z / np.sqrt(snps.N)
    snps["Winv_ahat_I"] = np.nan
    snps["R_Winv_ahat_I"] = np.nan
    snps["Winv_ahat_h"] = np.nan
    snps["R_Winv_ahat_h"] = np.nan

    for ldblock, _, meta, ind in refpanel.block_data(ldblocks, chrom, meta=snps):
        if meta is None:
            raise ValueError("phenotype preprocessing requires block metadata")
        svd_r_path = Path(f"{svd_stem}{ldblock.name}.R.npz")
        svd_r2_path = Path(f"{svd_stem}{ldblock.name}.R2.npz")
        if meta.printsnp.sum() == 0 or not svd_r_path.exists():
            print("no svd snps found in this block")
            continue
        print(meta.printsnp.sum(), "svd snps", meta.typed.sum(), "typed snps")
        if meta.typed.sum() == 0:
            print("no typed snps found in this block")
            snps.loc[ind, ["R_Winv_ahat_I", "R_Winv_ahat_h"]] = 0
            continue

        r = np.load(svd_r_path)
        r2 = np.load(svd_r2_path)
        sample_size = meta[meta.typed.values].N.mean()
        meta_svd = meta[meta.printsnp.values]
        snps.loc[ind[meta.printsnp], "Winv_ahat_I"] = weights.invert_weights(
            r,
            r2,
            sigma2g,
            sample_size,
            meta_svd.ahat.values,
            mode="Winv_ahat_I",
        )
        snps.loc[ind[meta.printsnp], "Winv_ahat_h"] = weights.invert_weights(
            r,
            r2,
            sigma2g,
            sample_size,
            meta_svd.ahat.values,
            mode="Winv_ahat_h",
        )

    return snps


def run(args: argparse.Namespace) -> None:
    """Preprocess GWAS summary statistics into weighted per-block inputs."""

    with memo.cache_scope():
        print("initializing...")

        refpanel = gd.Dataset(args.bfile_chr)
        ldblocks = _load_ldblocks(args.ld_blocks)
        print_snps = _load_print_snps(args.print_snps)
        print(len(print_snps), "svd snps")

        ss = _read_sumstats(args.sumstats_stem)
        ss = _filter_sumstats(ss, print_snps)
        ld, read_m_file = _read_ld_scores(args.ldscores_chr)

        if args.no_M_5_50:
            M = sum([read_m_file(f"{args.ldscores_chr}{c}.l2.M") for c in range(1, 23)])
        else:
            M = sum([read_m_file(f"{args.ldscores_chr}{c}.l2.M_5_50") for c in range(1, 23)])
        print(len(ld), "snps with ld scores")
        ssld = pd.merge(ss, ld, on="SNP", how="left")
        print(len(ssld), "hm3 snps with sumstats after merge.")

        h2g, sigma2g, meanchi2, K = _estimate_h2g(ssld, M)
        h2g = max(h2g, 0.03)  # 0.03 is an arbitrarily chosen minimum
        print("mean chi2:", meanchi2)
        print("h2g estimated at:", h2g, "sigma2g =", sigma2g)
        if args.set_h2g:
            print("scaling Z-scores to achieve h2g of", args.set_h2g)
            norm = meanchi2 / (1 + args.set_h2g / K)
            print("dividing all z-scores by", np.sqrt(norm))
            ssld.Z /= np.sqrt(norm)
            h2g, sigma2g, _, _ = _estimate_h2g(ssld, M)
            print("h2g is now", h2g)

        dirname = Path(f"{args.sumstats_stem}.{args.refpanel_name}")
        fs.makedir(dirname)
        if 1 in args.chroms:
            _write_info_file(dirname, args.sumstats_stem, h2g, sigma2g, ss)

        t0 = time.time()
        for c in args.chroms:
            print(time.time() - t0, ": loading chr", c, "of", args.chroms)
            snps = _process_chromosome(refpanel, ldblocks, c, print_snps, ss, args.svd_stem, sigma2g)

            print("writing processed sumstats")
            with gzip.open(dirname / f"{c}.pss.gz", "wt") as f:
                snps.loc[snps.printsnp, ["N", "Winv_ahat_I", "Winv_ahat_h"]].to_csv(f, index=False, sep="\t")

            del snps
            memo.reset()
            gc.collect()

    print("done")


def do(args: argparse.Namespace) -> None:
    """Backward-compatible alias for running phenotype preprocessing."""

    run(args)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for `preprocesspheno`."""

    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument(
        "--sumstats-stem",
        required=True,
        help='path to sumstats.gz files, not including ".sumstats.gz" extension',
    )

    # optional arguments
    parser.add_argument(
        "--refpanel-name",
        default="KG3.95",
        help="suffix added to the directory created for storing output. "
        + "Default is KG3.95, corresponding to 1KG Phase 3 reference panel "
        + "processed with default parameters by preprocessrefpanel.py.",
    )
    parser.add_argument(
        "-no-M-5-50",
        default=False,
        action="store_true",
        help="Dont filter to SNPs with MAF >= 0.05 when estimating heritabilities",
    )
    parser.add_argument(
        "--set-h2g",
        default=None,
        type=float,
        help="Scale Z-scores to achieve this approximate heritability",
    )
    parser.add_argument(
        "--chroms",
        nargs="+",
        default=range(1, 23),
        type=int,
        help="Space-delimited list of chromosomes to analyze. Default is 1..22",
    )

    # configurable arguments
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a json file with values for other parameters. "
        + "Values in this file will be overridden by any values passed "
        + "explicitly via the command line.",
    )
    parser.add_argument(
        "--bfile-chr",
        default=None,
        help="Path to plink bfile of reference panel to use, not including " + "chromosome number. If not supplied, will be read from config file.",
    )
    parser.add_argument(
        "--svd-stem",
        default=None,
        help="Path to directory containing truncated svds of reference panel, by LD "
        + "block, as produced by preprocessrefpanel.py. If not supplied, will be "
        + "read from config file.",
    )
    parser.add_argument(
        "--print-snps",
        default=None,
        help="Path to set of potentially typed SNPs. If not supplied, will be read " + "from config file.",
    )
    parser.add_argument(
        "--ldscores-chr",
        default=None,
        help="Path to LD scores at a smallish set of SNPs (~1M). LD should be computed "
        + "to all potentially causal snps. Used for heritability estimation. "
        + "If not supplied, will be read from config file.",
    )
    parser.add_argument(
        "--ld-blocks",
        default=None,
        help="Path to UCSC bed file containing one bed interval per LD block. If " + "not supplied, will be read from config file.",
    )

    return parser


def main() -> None:
    """Run the `preprocesspheno` command-line entry point."""

    print("=====")
    print(" ".join(sys.argv))
    print("=====")
    args = build_parser().parse_args()
    config.add_default_params(args)
    pretty.print_namespace(args)
    print("=====")

    run(args)


if __name__ == "__main__":
    main()
