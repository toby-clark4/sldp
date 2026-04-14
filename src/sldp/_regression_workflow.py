from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

import sldp.annotation as ga
import sldp.chunkstats as cs
import sldp.dataset as gd
import sldp.memo as memo
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


def build_annotation_context(args: argparse.Namespace, annotation_module=ga) -> AnnotationContext:
    """Load annotation objects, names, and summary metadata for regression."""

    annots = [annotation_module.Annotation(annot) for annot in args.sannot_chr]
    marginal_name_groups = [[name for name in annot.names(22, True) if ".R" in name] for annot in annots]
    background_annots = [annotation_module.Annotation(annot) for annot in args.background_sannot_chr]
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


def load_trait_info(pss_chr: str) -> tuple[str, float, float]:
    """Load phenotype naming and heritability metadata from processed sumstats."""

    pss_path = Path(pss_chr)
    pheno_name = pss_path.name.replace(".KG3.95", "")
    sumstats_info = pd.read_csv(pss_path / "info", sep="\t")
    sigma2g = sumstats_info.loc[0].sigma2g
    h2g = sumstats_info.loc[0].h2g
    return pheno_name, sigma2g, h2g


def collect_block_statistics(
    args: argparse.Namespace,
    refpanel: gd.Dataset,
    ldblocks: pd.DataFrame,
    annotation_context: AnnotationContext,
    sigma2g: float,
    *,
    weights_module=weights,
    memo_module=memo,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], pd.DataFrame]:
    """Collect per-block numerator and denominator terms across chromosomes."""

    numerators: dict[int, np.ndarray] = {}
    denominators: dict[int, np.ndarray] = {}
    t0 = time.time()
    combined_annotation_names = annotation_context.background_names + annotation_context.marginal_names
    pss_path = Path(args.pss_chr)
    svd_stem = Path(args.svd_stem)
    annotation_groups = list(zip(annotation_context.background_annots, annotation_context.background_name_groups)) + list(
        zip(annotation_context.annots, annotation_context.marginal_name_groups)
    )

    for c in args.chroms:
        print(time.time() - t0, ": loading chr", c, "of", args.chroms)

        snps = refpanel.bim_df(c)
        print(len(snps), "snps in refpanel", len(snps.columns), "columns, including metadata")

        print("reading sumstats")
        ss = pd.read_csv(pss_path / f"{c}.pss.gz", sep="\t")
        print(np.isnan(ss[args.weights]).sum(), "sumstats nans out of", len(ss))
        snps["Winv_ahat"] = ss[args.weights]
        snps["N"] = ss.N
        snps["typed"] = snps.Winv_ahat.notnull()
        if args.chi2_thresh > 0:
            print("applying chi2 threshold of", args.chi2_thresh)
            snps.typed &= ss.Winv_ahat_I**2 * ss.N > args.chi2_thresh
            print(snps.typed.sum(), "typed snps left")

        print("reading annotations")
        annotation_frames: list[pd.DataFrame] = []
        for annot, mynames in annotation_groups:
            print(time.time() - t0, ": reading annot", annot.filestem())
            print("adding", mynames)
            annotation_frame = annot.RV_df(c)[mynames]
            annotation_values = annotation_frame.to_numpy(copy=False)
            if (~np.isfinite(annotation_values)).sum() > 0:
                raise ValueError(
                    "There should be no nans in the postprocessed annotation. But there are " + str((~np.isfinite(annotation_values)).sum())
                )
            annotation_frames.append(annotation_frame)

        if annotation_frames:
            snps = pd.concat([snps, *annotation_frames], axis=1)

        if (np.array(combined_annotation_names) != snps.columns.values[-len(combined_annotation_names) :]).any():
            raise ValueError("Merged annotations are not in the right order")

        current_block_names: set[int] = set()
        for ldblock_name, numerator, denominator, typed_count, snpind_begin, snpind_end in _iter_block_statistics(
            refpanel.block_data(ldblocks, c, meta=snps, genos=False, verbose=0),
            svd_stem=svd_stem,
            weights_mode=args.weights,
            sigma2g=sigma2g,
            combined_annotation_names=combined_annotation_names,
            weights_module=weights_module,
        ):
            current_block_names.add(ldblock_name)
            ldblocks.loc[ldblock_name, "M_H"] = typed_count
            ldblocks.loc[ldblock_name, "snpind_begin"] = snpind_begin
            ldblocks.loc[ldblock_name, "snpind_end"] = snpind_end
            numerators[ldblock_name] = numerator
            denominators[ldblock_name] = denominator

        empty_blocks = set(ldblocks[ldblocks.chr == f"chr{c}"].index) - current_block_names
        if empty_blocks:
            ldblocks.loc[list(empty_blocks), "M_H"] = ldblocks.loc[list(empty_blocks), "M_H"].fillna(0)

        memo_module.reset()

    return numerators, denominators, ldblocks


def _iter_block_statistics(
    block_data: Iterator[tuple[pd.Series, np.ndarray | None, pd.DataFrame | None, pd.Index]],
    *,
    svd_stem: Path,
    weights_mode: str,
    sigma2g: float,
    combined_annotation_names: list[str],
    weights_module=weights,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, int, int, int]]:
    """Yield per-block regression terms for non-empty LD blocks."""

    for ldblock, _, meta, _ in block_data:
        if meta is None:
            raise ValueError("main SLDP regression requires block metadata")
        result = _compute_block_statistics(
            ldblock_name=int(ldblock.name),
            meta=meta,
            svd_stem=svd_stem,
            weights_mode=weights_mode,
            sigma2g=sigma2g,
            combined_annotation_names=combined_annotation_names,
            weights_module=weights_module,
        )
        if result is not None:
            yield result


def _compute_block_statistics(
    *,
    ldblock_name: int,
    meta: pd.DataFrame,
    svd_stem: Path,
    weights_mode: str,
    sigma2g: float,
    combined_annotation_names: list[str],
    weights_module=weights,
) -> tuple[int, np.ndarray, np.ndarray, int, int, int] | None:
    """Compute per-block numerator and denominator terms."""

    r_path = svd_stem / f"{ldblock_name}.R.npz"
    r2_path = svd_stem / f"{ldblock_name}.R2.npz"
    typed_count = int(meta.typed.sum())
    if typed_count == 0 or not r_path.exists():
        return None
    annotation_values = meta.loc[:, combined_annotation_names].to_numpy(copy=False)
    if (annotation_values == 0).all():
        return None

    typed_mask = meta.typed.to_numpy(copy=False)
    typed_annotation_values = annotation_values[typed_mask]
    sample_size = meta.loc[typed_mask, "N"].mean()
    r, r2 = _load_regression_weights(weights_mode, r_path, r2_path, len(meta))
    weighted_rv = weights_module.invert_weights(
        r,
        r2,
        sigma2g,
        sample_size,
        annotation_values,
        typed=typed_mask,
        mode=weights_mode,
    )
    numerator = typed_annotation_values.T @ meta.loc[typed_mask, "Winv_ahat"].to_numpy(copy=False) / 1e6
    denominator = typed_annotation_values.T @ weighted_rv[typed_mask] / 1e6
    return ldblock_name, numerator, denominator, typed_count, min(meta.index), max(meta.index) + 1


def _load_regression_weights(
    weights_mode: str,
    r_path: Path,
    r2_path: Path,
    expected_len: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Load the regression-weight SVD files needed for a block."""

    if weights_mode in {"Winv_ahat_h", "Winv_ahat_hlN"}:
        r = np.load(r_path)
        if len(r["U"]) != expected_len:
            raise ValueError("regression wgts dimension must match regression snps")
        return r, None
    if weights_mode in {"Winv_ahat_h2", "Winv_ahat"}:
        r = np.load(r_path)
        r2 = np.load(r2_path)
        if len(r["U"]) != expected_len or len(r2["U"]) != expected_len:
            raise ValueError("regression wgts dimension must match regression snps")
        return r, r2
    return None, None


def compute_annotation_result(
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
    *,
    chunkstats_module=cs,
    scipy_stats_module=st,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> AnnotationResult:
    """Compute regression statistics for one marginal annotation."""

    background_count = len(annotation_context.background_names)
    sqnorm = annotation_context.marginal_infos.loc[name[:-2], "sqnorm"]
    supp = annotation_context.marginal_infos.loc[name[:-2], "supp"]
    total_snps = annotation_context.marginal_infos.loc[name[:-2], "M"]

    mu = chunkstats_module.get_est(total_chunk_num, total_chunk_denom, index, background_count)
    q, r, mux, muy = chunkstats_module.residualize(
        chunk_nums,
        chunk_denoms,
        background_count,
        index,
        total_num=total_chunk_num,
        total_denom=total_chunk_denom,
    )
    se = chunkstats_module.jackknife_se(mu, loo_nums, loo_denoms, index, background_count)

    row: dict[str, float | str] = {
        "pheno": pheno_name,
        "annot": name,
    }

    if args.bothp or not args.fastp:
        signflip_result = chunkstats_module.signflip(q, args.T, printmem=True, mode=args.stat, rng=rng)
        if signflip_result is None:
            raise ValueError(f"Unsupported signflip mode: {args.stat}")
        p_emp, z_emp = signflip_result
        row["z"] = z_emp
        row["p"] = p_emp
    if args.bothp or args.fastp:
        z_fast = np.sum(q) / np.linalg.norm(q)
        p_fast = scipy_stats_module.chi2.sf(z_fast**2, 1)
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
        row["qkurtosis"] = scipy_stats_module.kurtosis(q)
        row["qstd"] = np.std(q)
        row["p_jk"] = scipy_stats_module.chi2.sf((mu / se) ** 2, 1)
        row["sqnorm"] = sqnorm

    rf = mu * np.sqrt(sqnorm / h2g)
    h2v_over_h2g = rf**2 - se**2 * sqnorm / (total_snps * sigma2g)
    row["rf"] = rf
    row["h2v/h2g"] = h2v_over_h2g
    row["h2v"] = h2v_over_h2g * h2g
    row["supp(v)/M"] = supp / total_snps

    return AnnotationResult(row=row, q=q, r=r, mux=mux, muy=muy)


def write_verbose_outputs(
    outfile_stem: str,
    pheno_name: str,
    name: str,
    background_names: list[str],
    chunkinfo: pd.DataFrame,
    result: AnnotationResult,
) -> None:
    """Write per-annotation chunk and coefficient outputs for verbose runs."""

    fname = Path(f"{outfile_stem}.{pheno_name}.{name}")
    print("writing verbose results to", fname)
    verbose_chunkinfo = chunkinfo.copy()
    verbose_chunkinfo["q"] = result.q
    verbose_chunkinfo["r"] = result.r
    verbose_chunkinfo.to_csv(f"{fname}.chunks", sep="\t", index=False)

    coeffs = pd.DataFrame({"annot": background_names, "mux": result.mux, "muy": result.muy})
    coeffs.to_csv(f"{fname}.coeffs", sep="\t", index=False)
