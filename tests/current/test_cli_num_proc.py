from __future__ import annotations

import argparse

import pytest

from sldp import preprocessannot, preprocesspheno, preprocessrefpanel, sldp
from sldp.utils.multiproc import validate_num_proc


@pytest.mark.parametrize(
    ("module", "required_args"),
    [
        (preprocessannot, ["--sannot-chr", "annot/"]),
        (preprocesspheno, ["--sumstats-stem", "sumstats/toy"]),
        (preprocessrefpanel, []),
        (sldp, ["--outfile-stem", "out", "--pss-chr", "pss/", "--sannot-chr", "annot/"]),
    ],
)
class TestNumProcCli:
    def test_build_parser_sets_default_num_proc_to_one(self, module, required_args: list[str]) -> None:
        args = module.build_parser().parse_args(required_args)

        assert args.num_proc == 1

    def test_build_parser_accepts_num_proc_flag(self, module, required_args: list[str]) -> None:
        args = module.build_parser().parse_args([*required_args, "--num-proc", "4"])

        assert args.num_proc == 4


class TestValidateNumProc:
    def test_validate_num_proc_leaves_positive_value_unchanged(self, capsys) -> None:
        args = argparse.Namespace(num_proc=3)

        result = validate_num_proc(args)

        assert result.num_proc == 3
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("invalid_value", [0, -2])
    def test_validate_num_proc_warns_and_clamps_invalid_values(self, invalid_value: int, capsys) -> None:
        args = argparse.Namespace(num_proc=invalid_value)

        result = validate_num_proc(args)

        assert result.num_proc == 1
        assert "num_proc must be a positive integer" in capsys.readouterr().out
