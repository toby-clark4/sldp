from __future__ import annotations

from pathlib import Path
import shutil

import gzip
import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed, SnpData


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase1_tiny"
DATA_ROOT = FIXTURE_ROOT / "data"
GENERATED_ROOT = FIXTURE_ROOT / "generated"
SVD_ROOT = GENERATED_ROOT / "svd"
RESULTS_ROOT = GENERATED_ROOT / "results"
REF_ROOT = DATA_ROOT / "refpanel"
REF_REG_ROOT = DATA_ROOT / "refpanel_reg"
ANNOT_ROOT = DATA_ROOT / "annot"
SUMSTATS_ROOT = DATA_ROOT / "sumstats"


def ensure_dirs() -> None:
    if FIXTURE_ROOT.exists():
        shutil.rmtree(FIXTURE_ROOT)
    for path in [
        FIXTURE_ROOT,
        DATA_ROOT,
        GENERATED_ROOT,
        SVD_ROOT,
        RESULTS_ROOT,
        REF_ROOT,
        REF_REG_ROOT,
        ANNOT_ROOT,
        SUMSTATS_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_gzip_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        df.to_csv(handle, sep="\t", index=False)


def make_chr1_reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CHR": [1, 1, 1, 1, 1, 1],
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6"],
            "CM": [0.0] * 6,
            "BP": [100, 200, 300, 400, 500, 600],
            "A1": ["A", "A", "C", "G", "A", "A"],
            "A2": ["C", "G", "T", "T", "C", "G"],
        }
    )


def make_chr2_reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CHR": [2, 2, 2, 2],
            "SNP": ["rs7", "rs8", "rs9", "rs10"],
            "CM": [0.0] * 4,
            "BP": [100, 200, 300, 400],
            "A1": ["C", "A", "G", "A"],
            "A2": ["T", "C", "T", "G"],
        }
    )


def genotype_matrix(chr_num: int) -> np.ndarray:
    if chr_num == 1:
        return np.array(
            [
                [0, 1, 2, 0, 1, 2],
                [1, 1, 1, 0, 2, 1],
                [2, 0, 1, 1, 1, 0],
                [0, 2, 0, 2, 0, 1],
            ],
            dtype=np.float64,
        )
    return np.array(
        [
            [0, 1, 2, 0],
            [1, 0, 1, 1],
            [2, 1, 0, 2],
            [0, 2, 1, 0],
        ],
        dtype=np.float64,
    )


def print_snp_set() -> set[str]:
    return {"rs1", "rs2", "rs4", "rs5", "rs7", "rs9", "rs10"}


def write_plink_chromosome(
    chr_num: int, bim: pd.DataFrame, matrix: np.ndarray, root: Path, stem_prefix: str
) -> None:
    iid = np.array([[f"F{i + 1}", f"I{i + 1}"] for i in range(4)], dtype=str)
    sid = bim["SNP"].to_numpy(dtype=str)
    pos = bim[["CHR", "CM", "BP"]].to_numpy(dtype=np.float64)
    snpdata = SnpData(iid=iid, sid=sid, val=matrix, pos=pos)
    stem = root / f"{stem_prefix}.{chr_num}"
    Bed.write(str(stem), snpdata, count_A1=False)
    bim.to_csv(
        root / f"{stem_prefix}.{chr_num}.bim", sep="\t", index=False, header=False
    )

    maf = matrix.mean(axis=0) / 2.0
    frq = pd.DataFrame(
        {
            "CHR": bim["CHR"],
            "SNP": bim["SNP"],
            "A1": bim["A1"],
            "A2": bim["A2"],
            "MAF": np.minimum(maf, 1.0 - maf),
            "NCHROBS": [8] * len(bim),
        }
    )
    frq.to_csv(root / f"{stem_prefix}.{chr_num}.frq", sep=" ", index=False)


def write_reference_panel() -> None:
    for chr_num, bim in [(1, make_chr1_reference()), (2, make_chr2_reference())]:
        matrix = genotype_matrix(chr_num)
        write_plink_chromosome(chr_num, bim, matrix, REF_ROOT, "toy_ref")

        keep = bim["SNP"].isin(print_snp_set()).to_numpy()
        reg_bim = bim.loc[keep].reset_index(drop=True)
        reg_matrix = matrix[:, keep]
        write_plink_chromosome(
            chr_num, reg_bim, reg_matrix, REF_REG_ROOT, "toy_ref_reg"
        )


def write_ld_blocks() -> None:
    lines = [
        "chr1\t50\t350\n",
        "chr1\t350\t700\n",
        "chr2\t50\t250\n",
        "chr2\t250\t500\n",
    ]
    write_text(DATA_ROOT / "ld_blocks.bed", "".join(lines))


def write_print_snps() -> None:
    snps = sorted(print_snp_set(), key=lambda snp: int(snp[2:]))
    write_text(DATA_ROOT / "print_snps.txt", "\n".join(snps) + "\n")


def write_ldscores() -> None:
    for chr_num, bim in [(1, make_chr1_reference()), (2, make_chr2_reference())]:
        df = pd.DataFrame(
            {
                "CHR": bim["CHR"],
                "SNP": bim["SNP"],
                "BP": bim["BP"],
                "L2": np.linspace(1.0, 2.0, len(bim)),
            }
        )
        write_gzip_tsv(DATA_ROOT / f"ldscores.{chr_num}.l2.ldscore.gz", df)
        m_value = f"{len(bim)}\n"
        write_text(DATA_ROOT / f"ldscores.{chr_num}.l2.M", m_value)
        write_text(DATA_ROOT / f"ldscores.{chr_num}.l2.M_5_50", m_value)

    empty = pd.DataFrame(
        {
            "CHR": pd.Series(dtype=int),
            "SNP": pd.Series(dtype=str),
            "BP": pd.Series(dtype=int),
            "L2": pd.Series(dtype=float),
        }
    )
    for chr_num in range(3, 23):
        write_gzip_tsv(DATA_ROOT / f"ldscores.{chr_num}.l2.ldscore.gz", empty)
        write_text(DATA_ROOT / f"ldscores.{chr_num}.l2.M", "0\n")
        write_text(DATA_ROOT / f"ldscores.{chr_num}.l2.M_5_50", "0\n")


def write_sumstats() -> None:
    df = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs4", "rs5", "rs7", "rs9", "rs10"],
            "A1": ["A", "A", "G", "A", "C", "G", "A"],
            "A2": ["C", "G", "T", "C", "T", "T", "G"],
            "Z": [2.0, -1.5, 0.5, 3.0, -2.2, 1.1, 0.7],
            "N": [1000, 1000, 900, 900, 800, 800, 800],
        }
    )
    write_gzip_tsv(SUMSTATS_ROOT / "toy.sumstats.gz", df)


def write_annotations() -> None:
    chr1 = make_chr1_reference().copy()
    chr1["annot_signal"] = [1.0, 0.5, 0.0, -0.5, 1.5, 0.0]
    chr1_background = make_chr1_reference().copy()
    chr1_background["annot_background"] = [0.2, 0.1, 0.0, 0.3, 0.4, 0.0]
    chr2 = make_chr2_reference().copy()
    chr2["annot_signal"] = [0.2, -0.1, 1.2, 0.0]
    chr2_background = make_chr2_reference().copy()
    chr2_background["annot_background"] = [0.0, 0.2, 0.3, 0.1]

    for chr_num, df in [(1, chr1), (2, chr2)]:
        write_gzip_tsv(ANNOT_ROOT / f"toy_annot.{chr_num}.sannot.gz", df)

    for chr_num, df in [(1, chr1_background), (2, chr2_background)]:
        write_gzip_tsv(ANNOT_ROOT / f"toy_background.{chr_num}.sannot.gz", df)

    placeholder_sannot = chr1.iloc[[0]].copy()
    placeholder_sannot["annot_signal"] = 0.0
    write_gzip_tsv(ANNOT_ROOT / "toy_annot.22.sannot.gz", placeholder_sannot)

    placeholder_rv = placeholder_sannot[["SNP", "A1", "A2", "annot_signal"]].copy()
    placeholder_rv["annot_signal.R"] = 0.0
    write_gzip_tsv(ANNOT_ROOT / "toy_annot.22.RV.gz", placeholder_rv)

    placeholder_background = chr1_background.iloc[[0]].copy()
    placeholder_background["annot_background"] = 0.0
    write_gzip_tsv(ANNOT_ROOT / "toy_background.22.sannot.gz", placeholder_background)

    placeholder_background_rv = placeholder_background[
        ["SNP", "A1", "A2", "annot_background"]
    ].copy()
    placeholder_background_rv["annot_background.R"] = 0.0
    write_gzip_tsv(ANNOT_ROOT / "toy_background.22.RV.gz", placeholder_background_rv)


def write_config() -> None:
    config = """{
  \"bfile-chr\": \"tests/fixtures/phase1_tiny/data/refpanel/toy_ref.\",
  \"bfile-reg-chr\": \"tests/fixtures/phase1_tiny/data/refpanel_reg/toy_ref_reg.\",
  \"svd-stem\": \"tests/fixtures/phase1_tiny/generated/svd/\",
  \"print-snps\": \"tests/fixtures/phase1_tiny/data/print_snps.txt\",
  \"ldscores-chr\": \"tests/fixtures/phase1_tiny/data/ldscores.\",
  \"ld-blocks\": \"tests/fixtures/phase1_tiny/data/ld_blocks.bed\"
}
"""
    write_text(FIXTURE_ROOT / "fixture_config.json", config)


def write_readme() -> None:
    content = """# Phase 1 Tiny Fixture

This fixture is generated by `scripts/generate_phase1_fixture.py`.

## Contents

- `data/refpanel/toy_ref.<chr>.bed/.bim/.fam/.frq`
- `data/refpanel_reg/toy_ref_reg.<chr>.bed/.bim/.fam/.frq`
- `data/ld_blocks.bed`
- `data/print_snps.txt`
- `data/ldscores.<chr>.l2.ldscore.gz`
- `data/ldscores.<chr>.l2.M`
- `data/ldscores.<chr>.l2.M_5_50`
- `data/annot/toy_annot.<chr>.sannot.gz`
- `data/annot/toy_background.<chr>.sannot.gz`
- `data/sumstats/toy.sumstats.gz`
- `fixture_config.json`

## Baseline Workflow

Run the installed baseline package in the `sldp` conda environment against this fixture to produce regression artifacts.

Expected early commands:

```bash
PYTHONPATH=src conda run -n sldp python scripts/generate_phase1_fixture.py
conda run -n sldp preprocessrefpanel --config tests/fixtures/phase1_tiny/fixture_config.json --chroms 1 2
conda run -n sldp preprocessannot --config tests/fixtures/phase1_tiny/fixture_config.json --sannot-chr tests/fixtures/phase1_tiny/data/annot/toy_annot. tests/fixtures/phase1_tiny/data/annot/toy_background. --chroms 1 2
conda run -n sldp preprocesspheno --config tests/fixtures/phase1_tiny/fixture_config.json --sumstats-stem tests/fixtures/phase1_tiny/data/sumstats/toy --chroms 1 2
conda run -n sldp sldp --config tests/fixtures/phase1_tiny/fixture_config.json --outfile-stem tests/fixtures/phase1_tiny/generated/results/toy --pss-chr tests/fixtures/phase1_tiny/data/sumstats/toy.KG3.95/ --sannot-chr tests/fixtures/phase1_tiny/data/annot/toy_annot. --background-sannot-chr tests/fixtures/phase1_tiny/data/annot/toy_background. --chroms 1 2 -fastp
```
"""
    write_text(FIXTURE_ROOT / "README.md", content)


def main() -> None:
    ensure_dirs()
    write_reference_panel()
    write_ld_blocks()
    write_print_snps()
    write_ldscores()
    write_sumstats()
    write_annotations()
    write_config()
    write_readme()
    print(f"Wrote fixture under {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
