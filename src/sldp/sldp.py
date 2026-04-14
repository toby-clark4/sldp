import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sldp.annotation as ga
import sldp.chunkstats as cs
import sldp.config as config
import sldp.dataset as gd
import sldp.memo as memo
import sldp.pretty as pretty
import sldp.storyteller as storyteller
import sldp.weights as weights
from sldp._processed_inputs import ensure_processed_inputs as _ensure_processed_inputs_impl
from sldp._processed_inputs import format_missing_message as _format_missing_message_impl
from sldp._processed_inputs import missing_annotation_artifacts as _missing_annotation_artifacts_impl
from sldp._processed_inputs import missing_pheno_artifacts as _missing_pheno_artifacts_impl
from sldp._processed_inputs import preprocess_sannots as _preprocess_sannots_impl
from sldp._processed_inputs import preprocess_sumstats as _preprocess_sumstats_impl
from sldp._processed_inputs import processed_pss_path as _processed_pss_path_impl
from sldp._regression_workflow import AnnotationContext, AnnotationResult
from sldp._regression_workflow import build_annotation_context
from sldp._regression_workflow import collect_block_statistics
from sldp._regression_workflow import compute_annotation_result
from sldp._regression_workflow import load_trait_info
from sldp._regression_workflow import write_verbose_outputs
from sldp.workflow_io import load_ldblocks


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
        "--preprocess",
        default=False,
        action="store_true",
        help="Preprocess missing phenotype or annotation artifacts before running. Only missing outputs are created.",
    )
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
        "--refpanel-name",
        default="KG3.95",
        help="Suffix used to locate or create processed phenotype directories when --sumstats-stem is used. Default is KG3.95.",
    )
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
    return build_annotation_context(args, annotation_module=ga)


def _load_trait_info(pss_chr: str) -> tuple[str, float, float]:
    return load_trait_info(pss_chr)


def _collect_block_statistics(
    args: argparse.Namespace,
    refpanel: gd.Dataset,
    ldblocks: pd.DataFrame,
    annotation_context: AnnotationContext,
    sigma2g: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], pd.DataFrame]:
    """Collect per-block numerator and denominator terms across chromosomes."""

    return collect_block_statistics(
        args,
        refpanel,
        ldblocks,
        annotation_context,
        sigma2g,
        weights_module=weights,
        memo_module=memo,
    )


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
    total_chunk_num: np.ndarray,
    total_chunk_denom: np.ndarray,
    loo_nums: list[np.ndarray],
    loo_denoms: list[np.ndarray],
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> AnnotationResult:
    """Compute regression statistics for one marginal annotation."""

    return compute_annotation_result(
        args,
        pheno_name,
        name,
        annotation_context,
        index,
        h2g,
        sigma2g,
        chunk_nums,
        chunk_denoms,
        total_chunk_num,
        total_chunk_denom,
        loo_nums,
        loo_denoms,
        chunkstats_module=cs,
        rng=rng,
    )


def _write_verbose_outputs(
    outfile_stem: str, pheno_name: str, name: str, background_names: list[str], chunkinfo: pd.DataFrame, result: AnnotationResult
) -> None:
    write_verbose_outputs(outfile_stem, pheno_name, name, background_names, chunkinfo, result)


def _processed_pss_path(args: argparse.Namespace) -> str:
    return _processed_pss_path_impl(args)


def _missing_pheno_artifacts(args: argparse.Namespace) -> list[str]:
    return _missing_pheno_artifacts_impl(args, path_exists=os.path.exists)


def _missing_annotation_artifacts(args: argparse.Namespace) -> dict[str, list[str]]:
    return _missing_annotation_artifacts_impl(args, path_exists=os.path.exists)


def _format_missing_message(header: str, missing: list[str], hint: str | None = None) -> str:
    return _format_missing_message_impl(header, missing, hint)


def _ensure_processed_inputs(args: argparse.Namespace) -> None:
    _ensure_processed_inputs_impl(
        args,
        preprocess_sumstats_fn=preprocess_sumstats,
        preprocess_sannots_fn=preprocess_sannots,
        path_exists=os.path.exists,
    )


def _load_ldblocks(path: str) -> pd.DataFrame:
    return load_ldblocks(path)


def run(args: argparse.Namespace) -> None:
    """Execute SLDP regression from a parsed argument namespace."""

    with memo.cache_scope():
        _ensure_processed_inputs(args)

        print("initializing...")

        refpanel = gd.Dataset(args.bfile_reg_chr)
        outfile_path = Path(args.outfile_stem)
        rng: np.random.Generator | np.random.RandomState | None = None
        if args.seed is not None:
            rng = np.random.RandomState(args.seed)
            print("random seed:", args.seed)

        annotation_context = _build_annotation_context(args)
        print("background annotations:", annotation_context.background_names)
        print("marginal annotations:", annotation_context.marginal_names)

        pheno_name, sigma2g, h2g = _load_trait_info(args.pss_chr)
        ldblocks = load_ldblocks(args.ld_blocks)
        numerators, denominators, ldblocks = _collect_block_statistics(
            args,
            refpanel,
            ldblocks,
            annotation_context,
            sigma2g,
        )

        print("jackknifing")
        chunk_nums, chunk_denoms, loo_nums, loo_denoms, chunkinfo = cs.collapse_to_chunks(ldblocks, numerators, denominators, args.jk_blocks)
        total_chunk_num = np.sum(np.stack(chunk_nums), axis=0)
        total_chunk_denom = np.sum(np.stack(chunk_denoms), axis=0)

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
                total_chunk_num=total_chunk_num,
                total_chunk_denom=total_chunk_denom,
                loo_nums=loo_nums,
                loo_denoms=loo_denoms,
                rng=rng,
            )
            result_rows.append(annotation_result.row)
            current_p = float(annotation_result.row["p"])

            if current_p < args.verbose_thresh:
                _write_verbose_outputs(
                    outfile_stem=args.outfile_stem,
                    pheno_name=pheno_name,
                    name=name,
                    background_names=annotation_context.background_names,
                    chunkinfo=chunkinfo,
                    result=annotation_result,
                )

            if current_p < args.tell_me_stories:
                storyteller.write(
                    str(Path(f"{outfile_path}.{name}.loci")),
                    args,
                    name,
                    annotation_context.background_names,
                    annotation_result.mux,
                    annotation_result.muy,
                    float(annotation_result.row["z"]),
                    corr_thresh=args.story_corr_thresh,
                )

        results = pd.DataFrame(result_rows)
        gwresults_path = outfile_path.with_name(outfile_path.name + ".gwresults")
        print(results)
        print("writing results to", gwresults_path)
        results.to_csv(gwresults_path, sep="\t", index=False, na_rep="nan")
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


def preprocess_sumstats(args: argparse.Namespace) -> None:
    _preprocess_sumstats_impl(args, path_exists=os.path.exists)


def preprocess_sannots(args: argparse.Namespace) -> None:
    _preprocess_sannots_impl(args, path_exists=os.path.exists)


if __name__ == "__main__":
    main()
