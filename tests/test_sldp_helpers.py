from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sldp.sldp as main_sldp


class TestBuildAnnotationContext:
    def test_build_annotation_context_collects_names_and_info(self, monkeypatch) -> None:
        class DummyAnnotation:
            def __init__(self, stem: str) -> None:
                self.stem = stem

            def names(self, chrnum: int, rv: bool = False) -> list[str]:
                del chrnum, rv
                if self.stem == "marginal":
                    return ["annot_signal.R", "annot_signal"]
                return ["annot_background.R"]

            def info_df(self, chroms: list[int]) -> pd.DataFrame:
                del chroms
                return pd.DataFrame({"sqnorm": [1.0], "supp": [2], "M": [3]}, index=["annot_signal"])

        monkeypatch.setattr(main_sldp.ga, "Annotation", DummyAnnotation)
        args = argparse.Namespace(sannot_chr=["marginal"], background_sannot_chr=["background"], chroms=[1, 2])

        context = main_sldp._build_annotation_context(args)

        assert context.marginal_names == ["annot_signal.R"]
        assert context.background_names == ["annot_background.R"]
        assert context.marginal_infos.loc["annot_signal", "sqnorm"] == 1.0


class TestLoadTraitInfo:
    def test_load_trait_info_reads_processed_sumstats_metadata(self, tmp_path: Path) -> None:
        info_dir = tmp_path / "toy.KG3.95"
        info_dir.mkdir()
        pd.DataFrame([{"sigma2g": 0.25, "h2g": 0.5}]).to_csv(info_dir / "info", sep="\t", index=False)

        pheno_name, sigma2g, h2g = main_sldp._load_trait_info(f"{info_dir}/")

        assert pheno_name == "toy"
        assert sigma2g == 0.25
        assert h2g == 0.5


class TestComputeAnnotationResult:
    def test_compute_annotation_result_returns_expected_row_fields(self, monkeypatch) -> None:
        context = main_sldp.AnnotationContext(
            annots=[],
            background_annots=[],
            marginal_name_groups=[["annot_signal.R"]],
            background_name_groups=[["annot_background.R"]],
            marginal_names=["annot_signal.R"],
            background_names=["annot_background.R"],
            marginal_infos=pd.DataFrame(
                {"sqnorm": [4.0], "supp": [2.0], "M": [10.0]},
                index=["annot_signal"],
            ),
        )

        monkeypatch.setattr(main_sldp.cs, "get_est", lambda *args, **kwargs: 2.0)
        monkeypatch.setattr(
            main_sldp.cs,
            "residualize",
            lambda *args, **kwargs: (
                np.array([1.0, 2.0]),
                np.array([3.0, 4.0]),
                np.array([0.5]),
                np.array([0.25]),
            ),
        )
        monkeypatch.setattr(main_sldp.cs, "jackknife_se", lambda *args, **kwargs: 0.5)
        monkeypatch.setattr(main_sldp.cs, "signflip", lambda *args, **kwargs: (0.1, 1.7))

        args = argparse.Namespace(T=100, stat="sum", bothp=True, fastp=False, more_stats=True)
        result = main_sldp._compute_annotation_result(
            args=args,
            pheno_name="toy",
            name="annot_signal.R",
            annotation_context=context,
            index=0,
            h2g=0.5,
            sigma2g=0.25,
            chunk_nums=[np.array([1.0])],
            chunk_denoms=[np.array([[1.0]])],
            loo_nums=[np.array([1.0])],
            loo_denoms=[np.array([[1.0]])],
        )

        assert result.row["pheno"] == "toy"
        assert result.row["annot"] == "annot_signal.R"
        assert result.row["mu"] == 2.0
        assert result.row["se(mu)"] == 0.5
        assert result.row["p"] == 0.1
        assert result.row["z"] == 1.7
        assert "p_fast" in result.row
        assert result.row["supp(v)/M"] == 0.2
        np.testing.assert_array_equal(result.q, np.array([1.0, 2.0]))
