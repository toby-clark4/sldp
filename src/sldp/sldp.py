import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

import sldp.annotation as ga
import sldp.chunkstats as cs
import sldp.config as config
import sldp.dataset as gd
import sldp.memo as memo
import sldp.pretty as pretty
import sldp.storyteller as storyteller
import sldp.weights as weights


@dataclass(frozen=True)
class AnnotationContext:
    """Annotation metadata needed by the main SLDP regression workflow."""

    annots: list[ga.Annotation]
    background_annots: list[ga.Annotation]
    marginal_name_groups: list[list[str]]
    background_name_groups: list[list[str]]
    marginal_names: list[str]
    background_names: list[str]
    marginal_infos: pd.DataFrame


@dataclass(frozen=True)
class AnnotationResult:
    """Computed statistics for a single marginal annotation."""

    row: dict[str, float | str]
    q: np.ndarray
    r: np.ndarray
    mux: np.ndarray
    muy: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the main `sldp` command."""

    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument("--outfile-stem", required=True, help="Path to an output file stem.")
    pheno = parser.add_mutually_exclusive_group(required=True)
    pheno.add_argument(
        "--pss-chr",
        default=None,
        help="Path to .pss.gz file, without chromosome number or .pss.gz extension. " + "This is the phenotype that SLDP will analyze.",
    )
    pheno.add_argument(
        "--sumstats-stem",
        default=None,
        help='Path to a .sumstats.gz file, not including ".sumstats.gz" extension. '
        + "SLDP will process this into a set of .pss.gz files before running.",
    )
    parser.add_argument(
        "--sannot-chr",
        nargs="+",
        required=True,
        help="One or more (space-delimited) paths to gzipped annot files, without "
        + "chromosome number or .sannot.gz/.RV.gz extension. These are the "
        + "annotations that SLDP will analyze against the phenotype.",
    )

    # optional arguments
    parser.add_argument(
        "--verbose-thresh",
        default=0.0,
        type=float,
        help="Print additional information about each association studied with a "
        + "p-value below this number. (Default is 0.) This includes: "
        + "the covariance in each independent block of genome (.chunks files), "
        + "and the coefficients required to residualize any background "
        + "annotations out of the other annotations being analyzed.",
    )
    parser.add_argument(
        "-fastp",
        default=False,
        action="store_true",
        help="Estimate p-values fast (without permutation)",
    )
    parser.add_argument(
        "-bothp",
        default=False,
        action="store_true",
        help="Print both fastp p-values (as p_fast) and normal p-values. " + "Takes precedence over fastp",
    )
    parser.add_argument(
        "--tell-me-stories",
        default=0.0,
        help="!!Experimental!! For associations with a p-value less than this number, "
        + "print information about loci that may be promising to study. "
        + "This will produce plots of (potentially overlapping) loci where the "
        + "signed LD profile is highly correlated with the GWAS signal in a "
        + "direction consistent with the global effect. "
        + "Default value is 0.",
    )
    parser.add_argument(
        "--story-corr-thresh",
        default=0.8,
        type=float,
        help="The threshold to use for correlation between Rv and alphahat in order " + "for a locus to be considered worthy of a story",
    )
    parser.add_argument(
        "-more-stats",
        default=False,
        action="store_true",
        help="Print additional statistis about q in results file",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=1000000,
        help="number of times to sign flip for empirical p-values. Default is 10^6.",
    )
    parser.add_argument(
        "--jk-blocks",
        type=int,
        default=300,
        help="Number of jackknife blocks to use. Default is 300.",
    )
    parser.add_argument(
        "--weights",
        default="Winv_ahat_h",
        help="which set of regression weights to use. Default is Winv_ahat_h, " + "corresponding to weights described in Reshef et al. 2017.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Seed random number generator to a certain value. Off by default.",
    )
    parser.add_argument(
        "--stat",
        default="sum",
        help="*experimental* Which statistic to use for hypothesis testing. Options " + "are: sum, medrank, or thresh.",
    )
    parser.add_argument(
        "--chi2-thresh",
        default=0,
        type=float,
        help="only use SNPs with a chi2 above this number for the regression",
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
        "--background-sannot-chr",
        nargs="+",
        default=[],
        help="One or more (space-delimited) paths to gzipped annot files, without "
        + "chromosome number or .sannot.gz extension. These are the annotations "
        + "that SLDP will control for.",
    )
    parser.add_argument(
        "--svd-stem",
        default=None,
        help="Path to directory in which output files will be stored. " + "If not supplied, will be read from config file.",
    )
    parser.add_argument(
        "--bfile-reg-chr",
        default=None,
        help="Path to plink bfile of reference panel to use, not including "
        + "chromosome number. This bfile should contain only regression SNPs "
        + "(as opposed to, e.g., all potentially causal SNPs). "
        + "If not supplied, will be read from config file.",
    )
    parser.add_argument(
        "--ld-blocks",
        default=None,
        help="Path to UCSC bed file containing one bed interval per LD block. If " + "not supplied, will be read from config file.",
    )

    return parser


def _build_annotation_context(args: argparse.Namespace) -> AnnotationContext:
    """Load annotation objects, names, and summary metadata for regression."""

    annots = [ga.Annotation(annot) for annot in args.sannot_chr]
    marginal_name_groups = [[name for name in annot.names(22, True) if ".R" in name] for annot in annots]
    background_annots = [ga.Annotation(annot) for annot in args.background_sannot_chr]
    background_name_groups = [[name for name in annot.names(22, True) if ".R" in name] for annot in background_annots]
    marginal_names = sum(marginal_name_groups, [])
    background_names = sum(background_name_groups, [])
    marginal_infos = pd.concat([annot.info_df(args.chroms) for annot in annots], axis=0)

    if len(set(background_names) & set(marginal_names)) > 0:
        raise ValueError("the background annotation names and the marginal annotation names must be disjoint sets")

    return AnnotationContext(
        annots=annots,
        background_annots=background_annots,
        marginal_name_groups=marginal_name_groups,
        background_name_groups=background_name_groups,
        marginal_names=marginal_names,
        background_names=background_names,
        marginal_infos=marginal_infos,
    )


def _load_ldblocks(path: str) -> pd.DataFrame:
    """Load LD blocks and remove blocks overlapping the MHC region."""

    mhc_bp = [25684587, 35455756]
    ldblocks = pd.read_csv(path, sep=r"\s+", header=None, names=["chr", "start", "end"])
    mhcblocks = (ldblocks.chr == "chr6") & (ldblocks.end > mhc_bp[0]) & (ldblocks.start < mhc_bp[1])
    return ldblocks[~mhcblocks]


def _load_trait_info(pss_chr: str) -> tuple[str, float, float]:
    """Load phenotype naming and heritability metadata from processed sumstats."""

    pheno_name = pss_chr.split("/")[-2].replace(".KG3.95", "")
    sumstats_info = pd.read_csv(pss_chr + "info", sep="\t")
    sigma2g = sumstats_info.loc[0].sigma2g
    h2g = sumstats_info.loc[0].h2g
    return pheno_name, sigma2g, h2g


def _collect_block_statistics(
    args: argparse.Namespace,
    refpanel: gd.Dataset,
    ldblocks: pd.DataFrame,
    annotation_context: AnnotationContext,
    sigma2g: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], pd.DataFrame]:
    """Collect per-block numerator and denominator terms across chromosomes."""

    numerators: dict[int, np.ndarray] = {}
    denominators: dict[int, np.ndarray] = {}
    t0 = time.time()
    combined_annotation_names = annotation_context.background_names + annotation_context.marginal_names
    annotation_groups = list(zip(annotation_context.background_annots, annotation_context.background_name_groups)) + list(
        zip(annotation_context.annots, annotation_context.marginal_name_groups)
    )

    for c in args.chroms:
        print(time.time() - t0, ": loading chr", c, "of", args.chroms)

        snps = refpanel.bim_df(c)
        print(len(snps), "snps in refpanel", len(snps.columns), "columns, including metadata")

        print("reading sumstats")
        ss = pd.read_csv(args.pss_chr + str(c) + ".pss.gz", sep="\t")
        print(np.isnan(ss[args.weights]).sum(), "sumstats nans out of", len(ss))
        snps["Winv_ahat"] = ss[args.weights]
        snps["N"] = ss.N
        snps["typed"] = snps.Winv_ahat.notnull()
        if args.chi2_thresh > 0:
            print("applying chi2 threshold of", args.chi2_thresh)
            snps.typed &= ss.Winv_ahat_I**2 * ss.N > args.chi2_thresh
            print(snps.typed.sum(), "typed snps left")

        print("reading annotations")
        for annot, mynames in annotation_groups:
            print(time.time() - t0, ": reading annot", annot.filestem())
            print("adding", mynames)
            snps = pd.concat([snps, annot.RV_df(c)[mynames]], axis=1)
            if (~np.isfinite(snps[mynames].values)).sum() > 0:
                raise ValueError(
                    "There should be no nans in the postprocessed annotation. But there are " + str((~np.isfinite(snps[mynames].values)).sum())
                )

        if (np.array(combined_annotation_names) != snps.columns.values[-len(combined_annotation_names) :]).any():
            raise ValueError("Merged annotations are not in the right order")

        for ldblock, _, meta, _ in refpanel.block_data(ldblocks, c, meta=snps, genos=False, verbose=0):
            if meta.typed.sum() == 0 or not os.path.exists(args.svd_stem + str(ldblock.name) + ".R.npz"):
                ldblocks.loc[ldblock.name, "M_H"] = 0
                continue
            if (meta[combined_annotation_names] == 0).values.all():
                ldblocks.loc[ldblock.name, "M_H"] = 0
                continue

            ldblocks.loc[ldblock.name, "M_H"] = meta.typed.sum()
            ldblocks.loc[ldblock.name, "snpind_begin"] = min(meta.index)
            ldblocks.loc[ldblock.name, "snpind_end"] = max(meta.index) + 1

            meta_t = meta[meta.typed.values]
            N = meta_t.N.mean()
            if args.weights in {"Winv_ahat_h", "Winv_ahat_hlN"}:
                R = np.load(args.svd_stem + str(ldblock.name) + ".R.npz")
                R2 = None
                if len(R["U"]) != len(meta):
                    raise ValueError("regression wgts dimension must match regression snps")
            elif args.weights in {"Winv_ahat_h2", "Winv_ahat"}:
                R = np.load(args.svd_stem + str(ldblock.name) + ".R.npz")
                R2 = np.load(args.svd_stem + str(ldblock.name) + ".R2.npz")
                if len(R["U"]) != len(meta) or len(R2["U"]) != len(meta):
                    raise ValueError("regression wgts dimension must match regression snps")
            else:
                R = None
                R2 = None

            weighted_rv = weights.invert_weights(
                R,
                R2,
                sigma2g,
                N,
                meta[combined_annotation_names].values,
                typed=meta.typed.values,
                mode=args.weights,
            )
            numerators[ldblock.name] = meta_t[combined_annotation_names].T.dot(meta_t.Winv_ahat).values / 1e6
            denominators[ldblock.name] = meta_t[combined_annotation_names].T.dot(weighted_rv[meta.typed.values]).values / 1e6

        memo.reset()

    return numerators, denominators, ldblocks


def _compute_annotation_result(
    args: argparse.Namespace,
    pheno_name: str,
    name: str,
    annotation_context: AnnotationContext,
    index: int,
    h2g: float,
    sigma2g: float,
    chunk_nums: list[np.ndarray],
    chunk_denoms: list[np.ndarray],
    loo_nums: list[np.ndarray],
    loo_denoms: list[np.ndarray],
) -> AnnotationResult:
    """Compute regression statistics for one marginal annotation."""

    background_count = len(annotation_context.background_names)
    sqnorm = annotation_context.marginal_infos.loc[name[:-2], "sqnorm"]
    supp = annotation_context.marginal_infos.loc[name[:-2], "supp"]
    M = annotation_context.marginal_infos.loc[name[:-2], "M"]

    mu = cs.get_est(sum(chunk_nums), sum(chunk_denoms), index, background_count)
    q, r, mux, muy = cs.residualize(chunk_nums, chunk_denoms, background_count, index)
    se = cs.jackknife_se(mu, loo_nums, loo_denoms, index, background_count)

    row: dict[str, float | str] = {
        "pheno": pheno_name,
        "annot": name,
    }

    if args.bothp or not args.fastp:
        p_emp, z_emp = cs.signflip(q, args.T, printmem=True, mode=args.stat)
        row["z"] = z_emp
        row["p"] = p_emp
    if args.bothp or args.fastp:
        z_fast = np.sum(q) / np.linalg.norm(q)
        p_fast = st.chi2.sf(z_fast**2, 1)
        row["z_fast"] = z_fast
        row["p_fast"] = p_fast
    if args.fastp and not args.bothp:
        row["p"] = row["p_fast"]
        row["z"] = row["z_fast"]
        del row["p_fast"]
        del row["z_fast"]

    row["mu"] = mu
    row["se(mu)"] = se
    row["h2g"] = h2g

    if args.more_stats:
        row["qkurtosis"] = st.kurtosis(q)
        row["qstd"] = np.std(q)
        row["p_jk"] = st.chi2.sf((mu / se) ** 2, 1)
        row["sqnorm"] = sqnorm

    row["rf"] = mu * np.sqrt(sqnorm / h2g)
    row["h2v/h2g"] = row["rf"] ** 2 - row["se(mu)"] ** 2 * sqnorm / (M * sigma2g)
    row["h2v"] = row["h2v/h2g"] * h2g
    row["supp(v)/M"] = supp / M

    return AnnotationResult(row=row, q=q, r=r, mux=mux, muy=muy)


def _write_verbose_outputs(
    outfile_stem: str,
    pheno_name: str,
    name: str,
    background_names: list[str],
    chunkinfo: pd.DataFrame,
    result: AnnotationResult,
) -> None:
    """Write per-annotation chunk and coefficient outputs for verbose runs."""

    fname = f"{outfile_stem}.{pheno_name}.{name}"
    print("writing verbose results to", fname)
    verbose_chunkinfo = chunkinfo.copy()
    verbose_chunkinfo["q"] = result.q
    verbose_chunkinfo["r"] = result.r
    verbose_chunkinfo.to_csv(fname + ".chunks", sep="\t", index=False)

    coeffs = pd.DataFrame({"annot": background_names, "mux": result.mux, "muy": result.muy})
    coeffs.to_csv(fname + ".coeffs", sep="\t", index=False)


def run(args: argparse.Namespace) -> None:
    """Execute SLDP regression from a parsed argument namespace."""

    preprocess_sumstats(args)
    preprocess_sannots(args)

    print("initializing...")

    refpanel = gd.Dataset(args.bfile_reg_chr)
    if args.seed is not None:
        np.random.seed(args.seed)
        print("random seed:", args.seed)

    annotation_context = _build_annotation_context(args)
    print("background annotations:", annotation_context.background_names)
    print("marginal annotations:", annotation_context.marginal_names)

    pheno_name, sigma2g, h2g = _load_trait_info(args.pss_chr)
    ldblocks = _load_ldblocks(args.ld_blocks)
    numerators, denominators, ldblocks = _collect_block_statistics(
        args,
        refpanel,
        ldblocks,
        annotation_context,
        sigma2g,
    )

    # get data for jackknifing
    print("jackknifing")
    chunk_nums, chunk_denoms, loo_nums, loo_denoms, chunkinfo = cs.collapse_to_chunks(ldblocks, numerators, denominators, args.jk_blocks)

    # compute final results
    result_rows: list[dict[str, float | str]] = []
    for i, name in enumerate(annotation_context.marginal_names):
        print(i, name)
        annotation_result = _compute_annotation_result(
            args=args,
            pheno_name=pheno_name,
            name=name,
            annotation_context=annotation_context,
            index=i,
            h2g=h2g,
            sigma2g=sigma2g,
            chunk_nums=chunk_nums,
            chunk_denoms=chunk_denoms,
            loo_nums=loo_nums,
            loo_denoms=loo_denoms,
        )
        result_rows.append(annotation_result.row)
        current_p = float(annotation_result.row["p"])

        # print verbose information if required
        if current_p < args.verbose_thresh:
            _write_verbose_outputs(
                outfile_stem=args.outfile_stem,
                pheno_name=pheno_name,
                name=name,
                background_names=annotation_context.background_names,
                chunkinfo=chunkinfo,
                result=annotation_result,
            )

        # nominate interesting loci if desired
        if current_p < args.tell_me_stories:
            storyteller.write(
                args.outfile_stem + "." + name + ".loci",
                args,
                name,
                annotation_context.background_names,
                annotation_result.mux,
                annotation_result.muy,
                float(annotation_result.row["z"]),
                corr_thresh=args.story_corr_thresh,
            )

    results = pd.DataFrame(result_rows)
    results.to_csv(args.outfile_stem + ".gwresults", sep="\t", index=False, na_rep="nan")

    print(results)
    print("writing results to", args.outfile_stem + ".gwresults")
    results.to_csv(args.outfile_stem + ".gwresults", sep="\t", index=False, na_rep="nan")
    print("done")


def main() -> None:
    """Run the main SLDP regression command-line entry point."""

    print("=====")
    print(" ".join(sys.argv))
    print("=====")
    args = build_parser().parse_args()
    config.add_default_params(args)
    pretty.print_namespace(args)
    print("=====")

    run(args)


# preprocess any sumstats that need preprocessing
def preprocess_sumstats(args: argparse.Namespace) -> None:
    import os

    if args.pss_chr is None:
        unprocessed_chroms = [c for c in args.chroms if not os.path.exists(args.sumstats_stem + "." + args.refpanel_name + "/" + str(c) + ".pss.gz")]
        if len(unprocessed_chroms) > 0:
            print(
                "Preprocessing",
                args.sumstats_stem + ".sumstats.gz... at",
                unprocessed_chroms,
            )
            if args.config is None:
                raise ValueError(
                    "automatic pre-processing of a sumstats file requires "
                    + "specification of a config file; otherwise I dont know what "
                    + "parameters to use. If you want, you can preprocess the sumstats "
                    + "without a config file by running preprocesspheno manually"
                )
            print("Using config file", args.config, "and default options")

            # run the command
            import copy
            import sldp.preprocesspheno

            args_ = copy.copy(args)
            args_.no_M_5_50 = False
            args_.set_h2g = None
            args_.chroms = unprocessed_chroms
            sldp.preprocesspheno.run(args_)

        # modify args to reflect existing of pss-chr files
        args.pss_chr = args.sumstats_stem + "." + args.refpanel_name + "/"
        args.sumstats_stem = None
        print("== finished preprocessing sumstats ==")


# preprocess any annotations that need preprocessing
def preprocess_sannots(args: argparse.Namespace) -> None:
    import os

    for sannot in args.sannot_chr:
        unprocessed_chroms = [
            c for c in args.chroms if not (os.path.exists(sannot + str(c) + ".RV.gz") and os.path.exists(sannot + str(c) + ".info"))
        ]
        if len(unprocessed_chroms) > 0:
            print("Preprocessing", sannot, "at chromosomes", unprocessed_chroms)
            if args.config is None:
                raise ValueError(
                    "automatic pre-processing of an annotation "
                    + "requires specification of a config file; otherwise I dont know what "
                    + "parameters to use. If you want, you can preprocess the annotation "
                    + "without a config file by running preprocessannot manually"
                )
            print("Using config file", args.config, "and default options")

            # run preprocessing command
            import copy
            import sldp.preprocessannot

            args_ = copy.copy(args)
            args_.alpha = -1
            args_.sannot_chr = [sannot]
            args_.chroms = unprocessed_chroms
            sldp.preprocessannot.run(args_)

            print("== finished preprocessing annotation", sannot)


if __name__ == "__main__":
    main()
