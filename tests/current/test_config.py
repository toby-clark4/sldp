from __future__ import annotations

import argparse
import json
from pathlib import Path

from sldp.config import add_default_params


class TestAddDefaultParams:
    def test_add_default_params_merges_config_and_cli_values(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "svd-stem": "from-config",
                    "chroms": [1, 2],
                    "weights": "Winv_ahat_h",
                }
            ),
            encoding="utf-8",
        )

        args = argparse.Namespace(
            config=str(config_path),
            svd_stem="from-cli",
            chroms=None,
            weights=None,
            outfile_stem="out",
        )

        add_default_params(args)

        assert args.svd_stem == "from-cli"
        assert args.chroms == [1, 2]
        assert args.weights == "Winv_ahat_h"
        assert args.outfile_stem == "out"

    def test_add_default_params_leaves_namespace_unchanged_without_config(self) -> None:
        args = argparse.Namespace(config=None, chroms=[1, 2], outfile_stem="out")

        add_default_params(args)

        assert args.chroms == [1, 2]
        assert args.outfile_stem == "out"
