from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path


def processed_pss_path(args: argparse.Namespace) -> str:
    """Return the processed phenotype directory used by the main regression."""

    if args.pss_chr is not None:
        return args.pss_chr
    if args.sumstats_stem is None:
        raise ValueError("Either --pss-chr or --sumstats-stem must be supplied")
    return f"{Path(args.sumstats_stem)}.{args.refpanel_name}/"


def missing_pheno_artifacts(args: argparse.Namespace, *, path_exists=os.path.exists) -> list[str]:
    """List missing processed phenotype artifacts required for the requested run."""

    pss_path = Path(processed_pss_path(args))
    info_path = str(pss_path / "info")
    missing = [info_path] if not path_exists(info_path) else []
    missing.extend(str(pss_path / f"{chrom}.pss.gz") for chrom in args.chroms if not path_exists(str(pss_path / f"{chrom}.pss.gz")))
    return missing


def missing_annotation_artifacts(args: argparse.Namespace, *, path_exists=os.path.exists) -> dict[str, list[str]]:
    """List missing processed annotation artifacts keyed by annotation stem."""

    missing: dict[str, list[str]] = {}
    for sannot in args.sannot_chr + args.background_sannot_chr:
        stem = Path(sannot)
        missing_paths: list[str] = []
        for chrom in args.chroms:
            rv_path = Path(f"{stem}{chrom}.RV.gz")
            info_path = Path(f"{stem}{chrom}.info")
            rv_path_str = str(rv_path)
            info_path_str = str(info_path)
            if not path_exists(rv_path_str):
                missing_paths.append(rv_path_str)
            if not path_exists(info_path_str):
                missing_paths.append(info_path_str)
        if missing_paths:
            missing[sannot] = missing_paths
    return missing


def format_missing_message(header: str, missing: list[str], hint: str | None = None) -> str:
    """Build a compact missing-artifact error message."""

    shown = ", ".join(missing[:3])
    suffix = "" if len(missing) <= 3 else f", and {len(missing) - 3} more"
    message = f"{header}: {shown}{suffix}."
    if hint is not None:
        message += f" {hint}"
    return message


def ensure_processed_inputs(
    args: argparse.Namespace,
    *,
    preprocess_sumstats_fn,
    preprocess_sannots_fn,
    path_exists=os.path.exists,
) -> None:
    """Validate processed inputs or optionally preprocess only missing artifacts."""

    missing_pheno = missing_pheno_artifacts(args, path_exists=path_exists)
    missing_annots = missing_annotation_artifacts(args, path_exists=path_exists)
    if not missing_pheno and not missing_annots:
        if args.pss_chr is None:
            args.pss_chr = processed_pss_path(args)
            args.sumstats_stem = None
        return

    if not args.preprocess:
        if missing_pheno:
            raise ValueError(
                format_missing_message(
                    "Missing processed phenotype artifacts",
                    missing_pheno,
                    "Rerun with --preprocess and --config to build only the missing files.",
                )
            )
        first_annot = next(iter(missing_annots.values()))
        raise ValueError(
            format_missing_message(
                "Missing processed annotation artifacts",
                first_annot,
                "Rerun with --preprocess and --config to build only the missing files.",
            )
        )

    if missing_pheno and args.pss_chr is not None and args.sumstats_stem is None:
        raise ValueError(
            format_missing_message(
                "Missing processed phenotype artifacts",
                missing_pheno,
                "Cannot rebuild them from --pss-chr alone. Rerun with --sumstats-stem, --preprocess, and --config.",
            )
        )

    preprocess_sumstats_fn(args)
    preprocess_sannots_fn(args)

    remaining_pheno = missing_pheno_artifacts(args, path_exists=path_exists)
    remaining_annots = missing_annotation_artifacts(args, path_exists=path_exists)
    if remaining_pheno:
        raise ValueError(format_missing_message("Processed phenotype artifacts are still missing after preprocessing", remaining_pheno))
    if remaining_annots:
        first_annot = next(iter(remaining_annots.values()))
        raise ValueError(format_missing_message("Processed annotation artifacts are still missing after preprocessing", first_annot))


def preprocess_sumstats(args: argparse.Namespace, *, path_exists=os.path.exists) -> None:
    """Preprocess missing phenotype artifacts and rewrite args to use `--pss-chr`."""

    if args.pss_chr is None:
        processed_dir = Path(f"{Path(args.sumstats_stem)}.{args.refpanel_name}")
        unprocessed_chroms = [c for c in args.chroms if not path_exists(str(processed_dir / f"{c}.pss.gz"))]
        if len(unprocessed_chroms) > 0:
            print("Preprocessing", args.sumstats_stem + ".sumstats.gz... at", unprocessed_chroms)
            if args.config is None:
                raise ValueError(
                    "automatic pre-processing of a sumstats file requires "
                    + "specification of a config file; otherwise I dont know what "
                    + "parameters to use. If you want, you can preprocess the sumstats "
                    + "without a config file by running preprocesspheno manually"
                )
            print("Using config file", args.config, "and default options")

            import sldp.preprocesspheno

            args_ = copy.copy(args)
            args_.no_M_5_50 = False
            args_.set_h2g = None
            args_.chroms = unprocessed_chroms
            sldp.preprocesspheno.run(args_)

        args.pss_chr = f"{processed_dir}/"
        args.sumstats_stem = None
        print("== finished preprocessing sumstats ==")


def preprocess_sannots(args: argparse.Namespace, *, path_exists=os.path.exists) -> None:
    """Preprocess missing annotation artifacts in-place on the provided args."""

    for sannot in args.sannot_chr:
        stem = Path(sannot)
        unprocessed_chroms = [
            c for c in args.chroms if not (path_exists(str(Path(f"{stem}{c}.RV.gz"))) and path_exists(str(Path(f"{stem}{c}.info"))))
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

            import sldp.preprocessannot

            args_ = copy.copy(args)
            args_.alpha = -1
            args_.sannot_chr = [sannot]
            args_.chroms = unprocessed_chroms
            sldp.preprocessannot.run(args_)

            print("== finished preprocessing annotation", sannot)
