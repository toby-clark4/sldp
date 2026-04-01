from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from sldp.annotation import Annotation, reconciled_to, smart_merge


class TestSmartMerge:
    def test_smart_merge_concats_when_snps_are_already_aligned(self) -> None:
        left = pd.DataFrame({"SNP": ["rs1", "rs2"], "A": [1, 2]})
        right = pd.DataFrame({"SNP": ["rs1", "rs2"], "B": [3, 4]})

        result = smart_merge(left, right)

        assert result.to_dict(orient="list") == {
            "SNP": ["rs1", "rs2"],
            "A": [1, 2],
            "B": [3, 4],
        }


class TestReconciledTo:
    def test_reconciled_to_flips_signed_values_for_ref_flip(self) -> None:
        ref = pd.DataFrame({"SNP": ["rs1", "rs2"], "A1": ["A", "C"], "A2": ["C", "T"]})
        df = pd.DataFrame(
            {
                "SNP": ["rs1", "rs2"],
                "A1": ["C", "C"],
                "A2": ["A", "T"],
                "score": [2.0, 3.0],
            }
        )

        result = reconciled_to(ref, df, ["score"])

        np.testing.assert_allclose(result["score"].values, np.array([-2.0, 3.0]))


class TestAnnotation:
    def test_annotation_reads_names_and_info(self, tmp_path: Path) -> None:
        stem = tmp_path / "toy."
        sannot = pd.DataFrame(
            {
                "SNP": ["rs1"],
                "CHR": [1],
                "BP": [1],
                "A1": ["A"],
                "A2": ["C"],
                "annot_signal": [1.5],
            }
        )
        info = pd.DataFrame(
            {
                "name": ["annot_signal"],
                "sqnorm": [2.25],
                "supp": [1],
                "M": [1],
                "M_5_50": [1],
                "sqnorm_5_50": [2.25],
                "supp_5_50": [1],
            }
        ).set_index("name")

        with gzip.open(f"{stem}1.sannot.gz", "wt", encoding="utf-8") as handle:
            sannot.to_csv(handle, sep="\t", index=False)
        info.to_csv(f"{stem}1.info", sep="\t")

        annot = Annotation(str(stem))

        assert annot.names(1) == ["annot_signal"]
        assert annot.info_df(1).loc["annot_signal", "sqnorm"] == 2.25
