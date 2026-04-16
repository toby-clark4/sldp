from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sldp.core import regression


class TestBuildAnnotationContext:
    def test_build_annotation_context_collects_names_and_info(self) -> None:
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

        annotation_module = type("AnnotationModule", (), {"Annotation": DummyAnnotation})
        args = argparse.Namespace(sannot_chr=["marginal"], background_sannot_chr=["background"], chroms=[1, 2])

        context = regression.build_annotation_context(args, annotation_module=annotation_module)

        assert context.marginal_names == ["annot_signal.R"]
        assert context.background_names == ["annot_background.R"]
        assert context.marginal_infos.loc["annot_signal", "sqnorm"] == 1.0

    def test_build_annotation_context_rejects_overlapping_annotation_names(self) -> None:
        class DummyAnnotation:
            def __init__(self, stem: str) -> None:
                self.stem = stem

            def names(self, chrnum: int, rv: bool = False) -> list[str]:
                del chrnum, rv, self
                return ["shared.R"]

            def info_df(self, chroms: list[int]) -> pd.DataFrame:
                del chroms
                return pd.DataFrame({"sqnorm": [1.0], "supp": [2], "M": [3]}, index=["shared"])

        annotation_module = type("AnnotationModule", (), {"Annotation": DummyAnnotation})
        args = argparse.Namespace(sannot_chr=["marginal"], background_sannot_chr=["background"], chroms=[1])

        try:
            regression.build_annotation_context(args, annotation_module=annotation_module)
        except ValueError as exc:
            assert "must be disjoint sets" in str(exc)
        else:
            raise AssertionError("expected overlapping annotation names to raise ValueError")


class TestLoadTraitInfo:
    def test_load_trait_info_reads_processed_sumstats_metadata(self, tmp_path: Path) -> None:
        info_dir = tmp_path / "toy.KG3.95"
        info_dir.mkdir()
        pd.DataFrame([{"sigma2g": 0.25, "h2g": 0.5}]).to_csv(info_dir / "info", sep="\t", index=False)

        pheno_name, sigma2g, h2g = regression.load_trait_info(f"{info_dir}/")

        assert pheno_name == "toy"
        assert sigma2g == 0.25
        assert h2g == 0.5


class TestComputeAnnotationResult:
    def test_compute_annotation_result_returns_expected_row_fields(self) -> None:
        context = regression.AnnotationContext(
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

        chunkstats_module = type(
            "ChunkstatsModule",
            (),
            {
                "get_est": staticmethod(lambda *args, **kwargs: 2.0),
                "residualize": staticmethod(
                    lambda *args, **kwargs: (
                        np.array([1.0, 2.0]),
                        np.array([3.0, 4.0]),
                        np.array([0.5]),
                        np.array([0.25]),
                    )
                ),
                "jackknife_se": staticmethod(lambda *args, **kwargs: 0.5),
                "signflip": staticmethod(lambda *args, **kwargs: (0.1, 1.7)),
            },
        )

        args = argparse.Namespace(T=100, stat="sum", bothp=True, fastp=False, more_stats=True)
        result = regression.compute_annotation_result(
            args=args,
            pheno_name="toy",
            name="annot_signal.R",
            annotation_context=context,
            index=0,
            h2g=0.5,
            sigma2g=0.25,
            chunk_nums=[np.array([1.0])],
            chunk_denoms=[np.array([[1.0]])],
            total_chunk_num=np.array([1.0]),
            total_chunk_denom=np.array([[1.0]]),
            loo_nums=[np.array([1.0])],
            loo_denoms=[np.array([[1.0]])],
            chunkstats_module=chunkstats_module,
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

    def test_compute_annotation_result_fastp_only_uses_fast_columns(self) -> None:
        context = regression.AnnotationContext(
            annots=[],
            background_annots=[],
            marginal_name_groups=[["annot_signal.R"]],
            background_name_groups=[],
            marginal_names=["annot_signal.R"],
            background_names=[],
            marginal_infos=pd.DataFrame({"sqnorm": [4.0], "supp": [2.0], "M": [10.0]}, index=["annot_signal"]),
        )
        chunkstats_module = type(
            "ChunkstatsModule",
            (),
            {
                "get_est": staticmethod(lambda *args, **kwargs: 1.5),
                "residualize": staticmethod(lambda *args, **kwargs: (np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([]), np.array([]))),
                "jackknife_se": staticmethod(lambda *args, **kwargs: 0.25),
            },
        )

        args = argparse.Namespace(T=100, stat="sum", bothp=False, fastp=True, more_stats=False)
        result = regression.compute_annotation_result(
            args=args,
            pheno_name="toy",
            name="annot_signal.R",
            annotation_context=context,
            index=0,
            h2g=0.5,
            sigma2g=0.25,
            chunk_nums=[np.array([1.0])],
            chunk_denoms=[np.array([[1.0]])],
            total_chunk_num=np.array([1.0]),
            total_chunk_denom=np.array([[1.0]]),
            loo_nums=[np.array([1.0])],
            loo_denoms=[np.array([[1.0]])],
            chunkstats_module=chunkstats_module,
        )

        assert "p_fast" not in result.row
        assert "z_fast" not in result.row
        assert "p" in result.row
        assert "z" in result.row

    def test_compute_annotation_result_rejects_invalid_signflip_mode(self) -> None:
        context = regression.AnnotationContext(
            annots=[],
            background_annots=[],
            marginal_name_groups=[["annot_signal.R"]],
            background_name_groups=[],
            marginal_names=["annot_signal.R"],
            background_names=[],
            marginal_infos=pd.DataFrame({"sqnorm": [4.0], "supp": [2.0], "M": [10.0]}, index=["annot_signal"]),
        )
        chunkstats_module = type(
            "ChunkstatsModule",
            (),
            {
                "get_est": staticmethod(lambda *args, **kwargs: 1.0),
                "residualize": staticmethod(lambda *args, **kwargs: (np.array([1.0]), np.array([1.0]), np.array([]), np.array([]))),
                "jackknife_se": staticmethod(lambda *args, **kwargs: 0.5),
                "signflip": staticmethod(lambda *args, **kwargs: None),
            },
        )

        args = argparse.Namespace(T=100, stat="bad", bothp=True, fastp=False, more_stats=False)
        try:
            regression.compute_annotation_result(
                args=args,
                pheno_name="toy",
                name="annot_signal.R",
                annotation_context=context,
                index=0,
                h2g=0.5,
                sigma2g=0.25,
                chunk_nums=[np.array([1.0])],
                chunk_denoms=[np.array([[1.0]])],
                total_chunk_num=np.array([1.0]),
                total_chunk_denom=np.array([[1.0]]),
                loo_nums=[np.array([1.0])],
                loo_denoms=[np.array([[1.0]])],
                chunkstats_module=chunkstats_module,
            )
        except ValueError as exc:
            assert "Unsupported signflip mode" in str(exc)
        else:
            raise AssertionError("expected invalid signflip mode to raise ValueError")

    def test_compute_annotation_result_passes_explicit_rng_to_signflip(self) -> None:
        context = regression.AnnotationContext(
            annots=[],
            background_annots=[],
            marginal_name_groups=[["annot_signal.R"]],
            background_name_groups=[],
            marginal_names=["annot_signal.R"],
            background_names=[],
            marginal_infos=pd.DataFrame({"sqnorm": [4.0], "supp": [2.0], "M": [10.0]}, index=["annot_signal"]),
        )
        seen: dict[str, object] = {}
        rng = np.random.default_rng(123)

        def fake_signflip(*args, **kwargs):
            seen["rng"] = kwargs["rng"]
            return 0.1, 1.7

        chunkstats_module = type(
            "ChunkstatsModule",
            (),
            {
                "get_est": staticmethod(lambda *args, **kwargs: 1.0),
                "residualize": staticmethod(lambda *args, **kwargs: (np.array([1.0]), np.array([1.0]), np.array([]), np.array([]))),
                "jackknife_se": staticmethod(lambda *args, **kwargs: 0.5),
                "signflip": staticmethod(fake_signflip),
            },
        )

        args = argparse.Namespace(T=100, stat="sum", bothp=True, fastp=False, more_stats=False)
        result = regression.compute_annotation_result(
            args=args,
            pheno_name="toy",
            name="annot_signal.R",
            annotation_context=context,
            index=0,
            h2g=0.5,
            sigma2g=0.25,
            chunk_nums=[np.array([1.0])],
            chunk_denoms=[np.array([[1.0]])],
            total_chunk_num=np.array([1.0]),
            total_chunk_denom=np.array([[1.0]]),
            loo_nums=[np.array([1.0])],
            loo_denoms=[np.array([[1.0]])],
            chunkstats_module=chunkstats_module,
            rng=rng,
        )

        assert seen["rng"] is rng
        assert result.row["p"] == 0.1


class TestWriteVerboseOutputs:
    def test_write_verbose_outputs_creates_chunk_and_coeff_files(self, tmp_path: Path) -> None:
        result = regression.AnnotationResult(
            row={"pheno": "toy", "annot": "annot_signal.R", "p": 0.1, "z": 1.2},
            q=np.array([1.0, 2.0]),
            r=np.array([3.0, 4.0]),
            mux=np.array([0.5]),
            muy=np.array([0.25]),
        )
        chunkinfo = pd.DataFrame({"ldblock_begin": [0, 1]})

        regression.write_verbose_outputs(str(tmp_path / "out"), "toy", "annot_signal.R", ["bg.R"], chunkinfo, result)

        chunks = pd.read_csv(tmp_path / "out.toy.annot_signal.R.chunks", sep="\t")
        coeffs = pd.read_csv(tmp_path / "out.toy.annot_signal.R.coeffs", sep="\t")
        assert chunks["q"].tolist() == [1.0, 2.0]
        assert coeffs["annot"].tolist() == ["bg.R"]
