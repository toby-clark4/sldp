import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sldp.config as config
import sldp.dataset as gd
import sldp.fs as fs
import sldp.memo as memo
import sldp.pretty as pretty
from sldp.workflow_io import load_ldblocks as _load_ldblocks
from sldp.workflow_io import load_print_snps as _load_print_snps


def _best_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a stable right-hand SVD, falling back to XTX when needed."""

    try:
        u, singular_values, _ = np.linalg.svd(matrix.T)
        singular_values = singular_values**2 / matrix.shape[0]
    except np.linalg.LinAlgError:
        print("\t\tresorting to svd of XTX")
        u, singular_values, _ = np.linalg.svd(matrix.T.dot(matrix))
        singular_values = singular_values / matrix.shape[0]
    return u, singular_values


def _svd_output_path(svd_stem: str | Path, block_name: int, suffix: str) -> Path:
    """Build the output path for a saved block SVD artifact."""

    return Path(f"{svd_stem}{block_name}.{suffix}.npz")


def _prepare_chromosome_snps(refpanel: gd.Dataset, chrom: int, print_snps: pd.DataFrame) -> pd.DataFrame:
    """Load chromosome metadata and mark the printed SNP subset."""

    snps = refpanel.bim_df(chrom)
    snps = pd.merge(snps, print_snps, on="SNP", how="left")
    snps["printsnp"] = snps.printsnp.notnull()
    print(len(snps), "snps in refpanel", len(snps.columns), "columns, including metadata")
    return snps


def _save_block_svds(X_print: np.ndarray, block_name: int, svd_stem: str | Path, spectrum_percent: float, num_print_snps: int) -> None:
    """Compute and save truncated SVDs for R and R2 for one LD block."""

    print("\tcomputing SVD of R_print")
    U_, svs_ = _best_svd(X_print)
    k = np.argmax(np.cumsum(svs_) / svs_.sum() >= spectrum_percent / 100.0)
    print("\treduced rank of", k, "out of", num_print_snps, "printed snps")
    np.savez(_svd_output_path(svd_stem, block_name, "R"), U=U_[:, :k], svs=svs_[:k])

    print("\tcomputing R2_print")
    r2 = X_print.T.dot(X_print.dot(X_print.T)).dot(X_print) / X_print.shape[0] ** 2
    print("\tcomputing SVD of R2_print")
    r2_u, r2_svs, _ = np.linalg.svd(r2)
    k = np.argmax(np.cumsum(r2_svs) / r2_svs.sum() >= spectrum_percent / 100.0)
    print("\treduced rank of", k, "out of", num_print_snps, "printed snps")
    np.savez(_svd_output_path(svd_stem, block_name, "R2"), U=r2_u[:, :k], svs=r2_svs[:k])


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for `preprocessrefpanel`."""

    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument(
        "--spectrum-percent",
        type=float,
        default=95,
        help="Determines how many eigenvectors are kept in the truncated SVD. "
        + "A value of x means that x percent of the eigenspectrum will be kept. "
        + "Default value: 95",
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
        "--svd-stem",
        default=None,
        help="Path to directory in which output files will be stored. " + "If not supplied, will be read from config file.",
    )
    parser.add_argument(
        "--bfile-chr",
        default=None,
        help="Path to plink bfile of reference panel to use, not including " + "chromosome number. If not supplied, will be read from config file.",
    )
    parser.add_argument(
        "--print-snps",
        default=None,
        help="Path to set of potentially typed SNPs. If not supplied, will be read " + "from config file.",
    )
    parser.add_argument(
        "--ld-blocks",
        default=None,
        help="Path to UCSC bed file containing one bed interval per LD block. If " + "not supplied, will be read from config file.",
    )

    return parser


def run(args: argparse.Namespace) -> None:
    """Preprocess a reference panel into truncated per-block SVDs."""

    with memo.cache_scope():
        refpanel = gd.Dataset(args.bfile_chr)
        fs.makedir_for_file(args.svd_stem)

        ldblocks = _load_ldblocks(args.ld_blocks)
        print(len(ldblocks), "loci after removing MHC")
        print_snps = _load_print_snps(args.print_snps)
        print(len(print_snps), "print snps")

        for c in args.chroms:
            print("loading chr", c, "of", args.chroms)
            snps = _prepare_chromosome_snps(refpanel, c, print_snps)

            for ldblock, X, meta, _ in refpanel.block_data(ldblocks, c, meta=snps):
                if X is None or meta is None:
                    raise ValueError("reference panel block processing requires genotypes and metadata")
                if meta.printsnp.sum() == 0:
                    print("no print snps found in this block")
                    continue

                _save_block_svds(
                    X_print=X[:, meta.printsnp.values],
                    block_name=ldblock.name,
                    svd_stem=args.svd_stem,
                    spectrum_percent=args.spectrum_percent,
                    num_print_snps=int(meta.printsnp.sum()),
                )

            del snps
            memo.reset()
            gc.collect()
    print("done")


def main() -> None:
    """Run the `preprocessrefpanel` command-line entry point."""

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
