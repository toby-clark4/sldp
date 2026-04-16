import argparse
import gc
import gzip
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sldp.utils import config, memo, pretty
from sldp.io import (
    annotation as ga,
    dataset as gd,
)
from sldp.io.workflow_io import (
    load_ldblocks as _load_ldblocks,
    load_print_snps as _load_print_snps,
)
from sldp.utils.multiproc import validate_num_proc


RV_CHUNK_WIDTH = 256


def _prepare_chromosome_annotation(
    refpanel: gd.Dataset,
    annot: ga.Annotation,
    chrom: int,
    print_snps: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load and reconcile one annotation on one chromosome."""

    snps = refpanel.bim_df(chrom)
    snps = ga.smart_merge(snps, refpanel.frq_df(chrom)[["SNP", "MAF"]])
    print(len(snps), "snps in refpanel", len(snps.columns), "columns, including metadata")

    print("reading annot", annot.filestem())
    names = annot.names(chrom)
    result_names = [name + ".R" for name in names]
    annot_df = annot.sannot_df(chrom)
    if "SNP" in annot_df.columns:
        print("not a thinannot => doing full reconciliation of snps and allele coding")
        snps = ga.reconciled_to(snps, annot_df, names, missing_val=0)
    else:
        print("detected thinannot, so assuming that annotation is synched to refpanel")
        snps = pd.concat([snps, annot_df[names]], axis=1)

    print("merging in print_snps")
    snps = pd.merge(snps, print_snps, how="left", on="SNP")
    snps["printsnp"] = snps.printsnp.notnull()

    if alpha != -1:
        print("scaling by maf according to alpha=", alpha)
        scale = np.power(2 * snps.MAF.values * (1 - snps.MAF.values), (1.0 + alpha) / 2)
        snps[names] = snps[names].values * scale[:, None]

    snps = pd.concat([snps, pd.DataFrame(np.zeros(snps[names].shape), columns=result_names)], axis=1)
    snps[names] = snps[names].astype(float)
    return snps, names, result_names


def _write_annotation_info(snps: pd.DataFrame, names: list[str], output_path: str) -> None:
    """Write per-annotation summary metrics for one chromosome."""

    print("computing basic statistics and writing")
    annotation_values = snps.loc[:, names].to_numpy(copy=False)
    maf_mask = (snps.MAF >= 0.05).values
    maf_annotation_values = annotation_values[maf_mask]

    info = pd.DataFrame(
        {
            "M": len(snps),
            "M_5_50": maf_mask.sum(),
            "sqnorm": np.sum(annotation_values * annotation_values, axis=0),
            "sqnorm_5_50": np.sum(maf_annotation_values * maf_annotation_values, axis=0),
            "supp": np.count_nonzero(annotation_values, axis=0).astype(float),
            "supp_5_50": np.count_nonzero(maf_annotation_values, axis=0).astype(float),
        },
        index=names,
    )
    info.index.name = "name"
    info.to_csv(output_path, sep="\t")


def _compute_rv_values(
    refpanel: gd.Dataset,
    ldblocks: pd.DataFrame,
    chrom: int,
    snps: pd.DataFrame,
    names: list[str],
    result_names: list[str],
) -> None:
    """Fill per-SNP RV values for one chromosome."""

    result_column_count = len(names)
    for _, X, meta, ind in refpanel.block_data(ldblocks, chrom, meta=snps):
        if X is None or meta is None:
            raise ValueError("annotation RV computation requires genotypes and metadata")
        if meta.printsnp.sum() == 0:
            print("no print-snps in this block")
            continue
        print(meta.printsnp.sum(), "print-snps")
        if (meta[names] == 0).values.all():
            print("annotations are all 0 in this block")
            snps.loc[ind, result_names] = 0
            continue

        mask = meta.printsnp.to_numpy(copy=False)
        annotation_values = meta.loc[:, names].to_numpy(copy=False)
        print_genotypes = X[:, mask]
        rv_values = np.empty((int(mask.sum()), result_column_count), dtype=float)

        for start in range(0, result_column_count, RV_CHUNK_WIDTH):
            stop = min(start + RV_CHUNK_WIDTH, result_column_count)
            annotation_chunk = annotation_values[:, start:stop]
            rv_values[:, start:stop] = print_genotypes.T @ (X @ annotation_chunk) / X.shape[0]

        snps.loc[ind[mask], result_names] = rv_values


def _write_rv_output(snps: pd.DataFrame, names: list[str], result_names: list[str], output_path: Path) -> None:
    """Write the processed RV output for one chromosome."""

    print("writing output")
    with gzip.open(output_path, "wt") as handle:
        snps.loc[snps.printsnp, ["SNP", "A1", "A2"] + names + result_names].to_csv(
            handle,
            index=False,
            sep="\t",
        )


def run(args: argparse.Namespace) -> None:
    """Preprocess signed annotations into LD-profile tables by chromosome."""

    with memo.cache_scope():
        print("initializing...")

        refpanel = gd.Dataset(args.bfile_chr)
        annots = [ga.Annotation(annot) for annot in args.sannot_chr]
        ldblocks = _load_ldblocks(args.ld_blocks)
        print(len(ldblocks), "loci after removing MHC")
        print_snps = _load_print_snps(args.print_snps)
        print(len(print_snps), "print snps")

        for annot in annots:
            t0 = time.time()
            for c in args.chroms:
                print(time.time() - t0, ": loading chr", c, "of", args.chroms)
                snps, names, result_names = _prepare_chromosome_annotation(refpanel, annot, c, print_snps, args.alpha)
                _write_annotation_info(snps, names, annot.info_filename(c))
                _compute_rv_values(refpanel, ldblocks, c, snps, names, result_names)
                _write_rv_output(snps, names, result_names, Path(annot.RV_filename(c)))

                del snps
                memo.reset()
                gc.collect()

    print("done")


def do(args: argparse.Namespace) -> None:
    """Backward-compatible alias for running annotation preprocessing."""

    run(args)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for `preprocessannot`."""

    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument(
        "--sannot-chr",
        nargs="+",
        required=True,
        help="Multiple (space-delimited) paths to sannot.gz files, not including " + "chromosome",
    )

    # optional arguments
    parser.add_argument(
        "--alpha",
        type=float,
        default=-1,
        help="scale annotation values by sqrt(2*maf(1-maf))^{alpha+1}. "
        + "-1 means assume annotation values are already per-normalized-genotype, "
        + "0 means assume they were per allele. Default is -1.",
    )
    parser.add_argument(
        "--chroms",
        nargs="+",
        default=range(1, 23),
        type=int,
        help="Space-delimited list of chromosomes to analyze. Default is 1..22",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of worker processes for chromosome/annotation parallelism. Default is 1 (serial).",
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


def main() -> None:
    """Run the `preprocessannot` command-line entry point."""

    print("=====")
    print(" ".join(sys.argv))
    print("=====")
    args = build_parser().parse_args()
    args = validate_num_proc(args)
    config.add_default_params(args)
    pretty.print_namespace(args)
    print("=====")

    run(args)


if __name__ == "__main__":
    main()
