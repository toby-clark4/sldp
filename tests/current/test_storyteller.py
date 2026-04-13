from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

import sldp.storyteller as storyteller


def _make_story_data(length: int = 120) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bim = pd.DataFrame(
        {
            "CHR": [1] * length,
            "SNP": [f"rs{i}" for i in range(length)],
            "CM": np.arange(length, dtype=float),
            "BP": np.arange(100, 100 + 10 * length, 10),
            "A1": ["A"] * length,
            "A2": ["C"] * length,
        }
    )
    pss = pd.DataFrame(
        {
            "N": [1000.0] * length,
            "Winv_ahat_I": [0.0] * length,
        }
    )
    rv = pd.DataFrame(
        {
            "bg.R": [0.0] * length,
            "annot_signal.R": [0.0] * length,
        }
    )

    signal = np.concatenate(([0.2], np.linspace(0.01, 0.1, 99)))
    pss.loc[:99, "Winv_ahat_I"] = signal
    rv.loc[:99, "annot_signal.R"] = signal
    return bim, pss, rv


class TestStoryteller:
    def test_write_saves_matching_story_window(self, monkeypatch, tmp_path: Path) -> None:
        bim, pss, rv = _make_story_data()
        saved_paths: list[str] = []

        class DummyDataset:
            def __init__(self, bfile_chr: str) -> None:
                self.bfile_chr = bfile_chr

            def bim_df(self, chrom: int) -> pd.DataFrame:
                del chrom
                return bim.copy()

        class DummyAnnotation:
            def __init__(self, stem: str) -> None:
                self.stem = stem

            def names(self, chrnum: int, RV: bool = False) -> list[str]:
                del chrnum, RV
                if self.stem == "focal":
                    return ["annot_signal.R"]
                return ["bg.R"]

            def RV_df(self, chrom: int) -> pd.DataFrame:
                del chrom
                if self.stem == "focal":
                    return rv[["annot_signal.R"]].copy()
                return rv[["bg.R"]].copy()

        monkeypatch.setattr(storyteller.gd, "Dataset", DummyDataset)
        monkeypatch.setattr(storyteller.ga, "Annotation", DummyAnnotation)
        monkeypatch.setattr(storyteller.pd, "read_csv", lambda *args, **kwargs: pss.copy())
        monkeypatch.setattr(storyteller.fs, "makedir_for_file", lambda path: None)
        monkeypatch.setattr(storyteller.plt, "figure", lambda: None)
        monkeypatch.setattr(storyteller.plt, "scatter", lambda *args, **kwargs: None)
        monkeypatch.setattr(storyteller.plt, "title", lambda *args, **kwargs: None)
        monkeypatch.setattr(storyteller.plt, "xlabel", lambda *args, **kwargs: None)
        monkeypatch.setattr(storyteller.plt, "ylabel", lambda *args, **kwargs: None)
        monkeypatch.setattr(storyteller.plt, "savefig", lambda path: saved_paths.append(str(path)))
        monkeypatch.setattr(storyteller.plt, "close", lambda: None)

        args = Namespace(
            bfile_reg_chr="unused",
            sannot_chr=["focal"],
            background_sannot_chr=["background"],
            chroms=[1],
            pss_chr=f"{tmp_path}/pss/",
        )

        storyteller.write(
            folder=str(tmp_path / "stories"),
            args=args,
            name="annot_signal.R",
            background_names=["bg.R"],
            mux=np.array([0.0]),
            muy=np.array([0.0]),
            z=2.0,
            corr_thresh=0.8,
        )

        assert saved_paths == [str(tmp_path / "stories" / "chr1:100-1100.pdf")]

    def test_write_skips_windows_with_wrong_direction(self, monkeypatch, tmp_path: Path) -> None:
        bim, pss, rv = _make_story_data()
        saved_paths: list[str] = []

        class DummyDataset:
            def __init__(self, bfile_chr: str) -> None:
                self.bfile_chr = bfile_chr

            def bim_df(self, chrom: int) -> pd.DataFrame:
                del chrom
                return bim.copy()

        class DummyAnnotation:
            def __init__(self, stem: str) -> None:
                self.stem = stem

            def names(self, chrnum: int, RV: bool = False) -> list[str]:
                del chrnum, RV
                if self.stem == "focal":
                    return ["annot_signal.R"]
                return ["bg.R"]

            def RV_df(self, chrom: int) -> pd.DataFrame:
                del chrom
                if self.stem == "focal":
                    return rv[["annot_signal.R"]].copy()
                return rv[["bg.R"]].copy()

        monkeypatch.setattr(storyteller.gd, "Dataset", DummyDataset)
        monkeypatch.setattr(storyteller.ga, "Annotation", DummyAnnotation)
        monkeypatch.setattr(storyteller.pd, "read_csv", lambda *args, **kwargs: pss.copy())
        monkeypatch.setattr(storyteller.fs, "makedir_for_file", lambda path: None)
        monkeypatch.setattr(storyteller.plt, "figure", lambda: None)
        monkeypatch.setattr(storyteller.plt, "scatter", lambda *args, **kwargs: None)
        monkeypatch.setattr(storyteller.plt, "title", lambda *args, **kwargs: None)
        monkeypatch.setattr(storyteller.plt, "xlabel", lambda *args, **kwargs: None)
        monkeypatch.setattr(storyteller.plt, "ylabel", lambda *args, **kwargs: None)
        monkeypatch.setattr(storyteller.plt, "savefig", lambda path: saved_paths.append(str(path)))
        monkeypatch.setattr(storyteller.plt, "close", lambda: None)

        args = Namespace(
            bfile_reg_chr="unused",
            sannot_chr=["focal"],
            background_sannot_chr=["background"],
            chroms=[1],
            pss_chr=f"{tmp_path}/pss/",
        )

        storyteller.write(
            folder=str(tmp_path / "stories"),
            args=args,
            name="annot_signal.R",
            background_names=["bg.R"],
            mux=np.array([0.0]),
            muy=np.array([0.0]),
            z=-2.0,
            corr_thresh=0.8,
        )

        assert saved_paths == []
