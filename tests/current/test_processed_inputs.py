from __future__ import annotations

import argparse

import sldp.preprocessannot
import sldp.preprocesspheno
from sldp.core import processed_inputs


class TestPreprocessSumstats:
    def test_preprocess_sumstats_calls_preprocesspheno_run_for_missing_outputs(self, monkeypatch) -> None:
        calls: list[argparse.Namespace] = []

        def fake_exists(path: str) -> bool:
            return False

        def fake_run(args: argparse.Namespace) -> None:
            calls.append(args)

        monkeypatch.setattr(sldp.preprocesspheno, "run", fake_run)

        args = argparse.Namespace(
            pss_chr=None,
            chroms=[1, 2],
            sumstats_stem="sumstats/toy",
            refpanel_name="KG3.95",
            config="cfg.json",
        )

        processed_inputs.preprocess_sumstats(args, path_exists=fake_exists)

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

        monkeypatch.setattr(sldp.preprocesspheno, "run", fake_run)

        args = argparse.Namespace(
            pss_chr=None,
            chroms=[1, 2],
            sumstats_stem="sumstats/toy",
            refpanel_name="KG3.95",
            config="cfg.json",
        )

        processed_inputs.preprocess_sumstats(args, path_exists=fake_exists)

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

        monkeypatch.setattr(sldp.preprocessannot, "run", fake_run)

        args = argparse.Namespace(
            sannot_chr=["annot/a.", "annot/b."],
            chroms=[1, 2],
            config="cfg.json",
        )

        processed_inputs.preprocess_sannots(args, path_exists=fake_exists)

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

        monkeypatch.setattr(sldp.preprocessannot, "run", fake_run)

        args = argparse.Namespace(
            sannot_chr=["annot/a.", "annot/b."],
            background_sannot_chr=[],
            chroms=[1, 2],
            config="cfg.json",
        )

        processed_inputs.preprocess_sannots(args, path_exists=fake_exists)

        assert len(calls) == 1
        assert calls[0].sannot_chr == ["annot/b."]
        assert calls[0].chroms == [1]
        assert calls[0].alpha == -1


class TestEnsureProcessedInputs:
    def test_missing_processed_inputs_raise_without_preprocess(self, monkeypatch) -> None:
        fake_exists = lambda path: False

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
            processed_inputs.ensure_processed_inputs(
                args,
                preprocess_sumstats_fn=processed_inputs.preprocess_sumstats,
                preprocess_sannots_fn=processed_inputs.preprocess_sannots,
                path_exists=fake_exists,
            )
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
        fake_exists = lambda path: path in existing

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
            processed_inputs.ensure_processed_inputs(
                args,
                preprocess_sumstats_fn=processed_inputs.preprocess_sumstats,
                preprocess_sannots_fn=processed_inputs.preprocess_sannots,
                path_exists=fake_exists,
            )
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

        def fake_preprocess_sannots(args: argparse.Namespace) -> None:
            del args
            calls.append("sannots")

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

        processed_inputs.ensure_processed_inputs(
            args,
            preprocess_sumstats_fn=fake_preprocess_sumstats,
            preprocess_sannots_fn=fake_preprocess_sannots,
            path_exists=lambda path: path in existing,
        )

        assert calls == ["sumstats", "sannots"]

    def test_preprocess_mode_rejects_missing_pss_inputs_without_sumstats_source(self, monkeypatch) -> None:
        fake_exists = lambda path: False

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
            processed_inputs.ensure_processed_inputs(
                args,
                preprocess_sumstats_fn=processed_inputs.preprocess_sumstats,
                preprocess_sannots_fn=processed_inputs.preprocess_sannots,
                path_exists=fake_exists,
            )
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

        def fake_preprocess_sumstats(args: argparse.Namespace) -> None:
            del args
            calls.append("sumstats")

        def fake_preprocess_sannots(args: argparse.Namespace) -> None:
            del args
            calls.append("sannots")

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

        processed_inputs.ensure_processed_inputs(
            args,
            preprocess_sumstats_fn=fake_preprocess_sumstats,
            preprocess_sannots_fn=fake_preprocess_sannots,
            path_exists=lambda path: path in existing,
        )

        assert calls == []
        assert args.pss_chr == "sumstats/toy.KG3.95/"
        assert args.sumstats_stem is None
