from __future__ import annotations

import argparse

import sldp.preprocessannot
import sldp.preprocesspheno
import sldp.sldp as main_sldp
from sldp.sldp import preprocess_sannots, preprocess_sumstats


class TestPreprocessSumstats:
    def test_preprocess_sumstats_calls_preprocesspheno_run_for_missing_outputs(self, monkeypatch) -> None:
        calls: list[argparse.Namespace] = []

        def fake_exists(path: str) -> bool:
            return False

        def fake_run(args: argparse.Namespace) -> None:
            calls.append(args)

        monkeypatch.setattr("os.path.exists", fake_exists)
        monkeypatch.setattr(sldp.preprocesspheno, "run", fake_run)

        args = argparse.Namespace(
            pss_chr=None,
            chroms=[1, 2],
            sumstats_stem="sumstats/toy",
            refpanel_name="KG3.95",
            config="cfg.json",
        )

        preprocess_sumstats(args)

        assert len(calls) == 1
        assert calls[0].chroms == [1, 2]
        assert calls[0].sumstats_stem == "sumstats/toy"
        assert calls[0].no_M_5_50 is False
        assert calls[0].set_h2g is None
        assert args.pss_chr == "sumstats/toy.KG3.95/"
        assert args.sumstats_stem is None

    def test_preprocess_sumstats_skips_when_outputs_exist(self, monkeypatch) -> None:
        calls: list[argparse.Namespace] = []

        def fake_exists(path: str) -> bool:
            return path in {"sumstats/toy.KG3.95/info", "sumstats/toy.KG3.95/1.pss.gz", "sumstats/toy.KG3.95/2.pss.gz"}

        def fake_run(args: argparse.Namespace) -> None:
            calls.append(args)

        monkeypatch.setattr("os.path.exists", fake_exists)
        monkeypatch.setattr(sldp.preprocesspheno, "run", fake_run)

        args = argparse.Namespace(
            pss_chr=None,
            chroms=[1, 2],
            sumstats_stem="sumstats/toy",
            refpanel_name="KG3.95",
            config="cfg.json",
        )

        preprocess_sumstats(args)

        assert calls == []
        assert args.pss_chr == "sumstats/toy.KG3.95/"
        assert args.sumstats_stem is None


class TestPreprocessSannots:
    def test_preprocess_sannots_calls_preprocessannot_run_per_missing_annotation(self, monkeypatch) -> None:
        calls: list[argparse.Namespace] = []

        def fake_exists(path: str) -> bool:
            return False

        def fake_run(args: argparse.Namespace) -> None:
            calls.append(args)

        monkeypatch.setattr("os.path.exists", fake_exists)
        monkeypatch.setattr(sldp.preprocessannot, "run", fake_run)

        args = argparse.Namespace(
            sannot_chr=["annot/a.", "annot/b."],
            chroms=[1, 2],
            config="cfg.json",
        )

        preprocess_sannots(args)

        assert len(calls) == 2
        assert calls[0].sannot_chr == ["annot/a."]
        assert calls[1].sannot_chr == ["annot/b."]
        assert calls[0].chroms == [1, 2]
        assert calls[1].chroms == [1, 2]
        assert calls[0].alpha == -1
        assert calls[1].alpha == -1

    def test_preprocess_sannots_only_processes_missing_artifacts(self, monkeypatch) -> None:
        calls: list[argparse.Namespace] = []

        def fake_exists(path: str) -> bool:
            existing = {
                "annot/a.1.RV.gz",
                "annot/a.1.info",
                "annot/a.2.RV.gz",
                "annot/a.2.info",
                "annot/b.2.RV.gz",
                "annot/b.2.info",
            }
            return path in existing

        def fake_run(args: argparse.Namespace) -> None:
            calls.append(args)

        monkeypatch.setattr("os.path.exists", fake_exists)
        monkeypatch.setattr(sldp.preprocessannot, "run", fake_run)

        args = argparse.Namespace(
            sannot_chr=["annot/a.", "annot/b."],
            background_sannot_chr=[],
            chroms=[1, 2],
            config="cfg.json",
        )

        preprocess_sannots(args)

        assert len(calls) == 1
        assert calls[0].sannot_chr == ["annot/b."]
        assert calls[0].chroms == [1]
        assert calls[0].alpha == -1


class TestEnsureProcessedInputs:
    def test_missing_processed_inputs_raise_without_preprocess(self, monkeypatch) -> None:
        monkeypatch.setattr("os.path.exists", lambda path: False)

        args = argparse.Namespace(
            pss_chr=None,
            sumstats_stem="sumstats/toy",
            refpanel_name="KG3.95",
            sannot_chr=["annot/a."],
            background_sannot_chr=[],
            chroms=[1, 2],
            preprocess=False,
            config="cfg.json",
        )

        try:
            main_sldp._ensure_processed_inputs(args)
        except ValueError as exc:
            assert "Missing processed phenotype artifacts" in str(exc)
            assert "--preprocess and --config" in str(exc)
        else:
            raise AssertionError("expected missing processed inputs to raise ValueError")

    def test_missing_processed_annotation_artifacts_raise_without_preprocess(self, monkeypatch) -> None:
        existing = {
            "sumstats/toy.KG3.95/info",
            "sumstats/toy.KG3.95/1.pss.gz",
            "sumstats/toy.KG3.95/2.pss.gz",
        }
        monkeypatch.setattr("os.path.exists", lambda path: path in existing)

        args = argparse.Namespace(
            pss_chr=None,
            sumstats_stem="sumstats/toy",
            refpanel_name="KG3.95",
            sannot_chr=["annot/a."],
            background_sannot_chr=[],
            chroms=[1, 2],
            preprocess=False,
            config="cfg.json",
        )

        try:
            main_sldp._ensure_processed_inputs(args)
        except ValueError as exc:
            assert "Missing processed annotation artifacts" in str(exc)
            assert "annot/a.1.RV.gz" in str(exc)
        else:
            raise AssertionError("expected missing annotation artifacts to raise ValueError")

    def test_preprocess_mode_invokes_preprocess_path_then_revalidates_outputs(self, monkeypatch) -> None:
        calls: list[str] = []
        existing = {
            "annot/a.1.RV.gz",
            "annot/a.1.info",
            "annot/a.2.RV.gz",
            "annot/a.2.info",
        }

        def fake_preprocess_sumstats(args: argparse.Namespace) -> None:
            calls.append("sumstats")
            existing.update({"sumstats/toy.KG3.95/info", "sumstats/toy.KG3.95/1.pss.gz", "sumstats/toy.KG3.95/2.pss.gz"})
            args.pss_chr = "sumstats/toy.KG3.95/"
            args.sumstats_stem = None

        monkeypatch.setattr("os.path.exists", lambda path: path in existing)
        monkeypatch.setattr(main_sldp, "preprocess_sumstats", fake_preprocess_sumstats)
        monkeypatch.setattr(main_sldp, "preprocess_sannots", lambda args: calls.append("sannots"))

        args = argparse.Namespace(
            pss_chr=None,
            sumstats_stem="sumstats/toy",
            refpanel_name="KG3.95",
            sannot_chr=["annot/a."],
            background_sannot_chr=[],
            chroms=[1, 2],
            preprocess=True,
            config="cfg.json",
        )

        main_sldp._ensure_processed_inputs(args)

        assert calls == ["sumstats", "sannots"]

    def test_preprocess_mode_rejects_missing_pss_inputs_without_sumstats_source(self, monkeypatch) -> None:
        monkeypatch.setattr("os.path.exists", lambda path: False)

        args = argparse.Namespace(
            pss_chr="sumstats/toy.KG3.95/",
            sumstats_stem=None,
            refpanel_name="KG3.95",
            sannot_chr=["annot/a."],
            background_sannot_chr=[],
            chroms=[1, 2],
            preprocess=True,
            config="cfg.json",
        )

        try:
            main_sldp._ensure_processed_inputs(args)
        except ValueError as exc:
            assert "Cannot rebuild them from --pss-chr alone" in str(exc)
        else:
            raise AssertionError("expected missing pss inputs without sumstats source to raise ValueError")

    def test_existing_processed_inputs_skip_preprocessing(self, monkeypatch) -> None:
        calls: list[str] = []
        existing = {
            "sumstats/toy.KG3.95/info",
            "sumstats/toy.KG3.95/1.pss.gz",
            "sumstats/toy.KG3.95/2.pss.gz",
            "annot/a.1.RV.gz",
            "annot/a.1.info",
            "annot/a.2.RV.gz",
            "annot/a.2.info",
        }

        monkeypatch.setattr("os.path.exists", lambda path: path in existing)
        monkeypatch.setattr(main_sldp, "preprocess_sumstats", lambda args: calls.append("sumstats"))
        monkeypatch.setattr(main_sldp, "preprocess_sannots", lambda args: calls.append("sannots"))

        args = argparse.Namespace(
            pss_chr=None,
            sumstats_stem="sumstats/toy",
            refpanel_name="KG3.95",
            sannot_chr=["annot/a."],
            background_sannot_chr=[],
            chroms=[1, 2],
            preprocess=True,
            config="cfg.json",
        )

        main_sldp._ensure_processed_inputs(args)

        assert calls == []
        assert args.pss_chr == "sumstats/toy.KG3.95/"
        assert args.sumstats_stem is None


class TestBuildParser:
    def test_build_parser_supports_explicit_preprocess_flag(self) -> None:
        args = main_sldp.build_parser().parse_args(
            [
                "--outfile-stem",
                "out/toy",
                "--pss-chr",
                "sumstats/toy.KG3.95/",
                "--sannot-chr",
                "annot/a.",
                "--preprocess",
            ]
        )

        assert args.preprocess is True
        assert args.refpanel_name == "KG3.95"
