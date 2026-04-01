from __future__ import annotations

import argparse
from pathlib import Path

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
