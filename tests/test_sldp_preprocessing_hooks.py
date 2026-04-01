from __future__ import annotations

import argparse

import sldp.preprocessannot
import sldp.preprocesspheno
from sldp.sldp import preprocess_sannots, preprocess_sumstats


class TestPreprocessSumstats:
    def test_preprocess_sumstats_calls_preprocesspheno_run_for_missing_outputs(
        self, monkeypatch
    ) -> None:
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


class TestPreprocessSannots:
    def test_preprocess_sannots_calls_preprocessannot_run_per_missing_annotation(
        self, monkeypatch
    ) -> None:
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
