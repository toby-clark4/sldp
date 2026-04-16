import json
from argparse import Namespace
from pathlib import Path


def add_default_params(args: Namespace) -> None:
    """Fill missing argparse parameters from a JSON config file."""

    if args.config is not None:
        # read in config file and replace '-' with '_' to match argparse behavior
        with Path(args.config).open("r", encoding="utf-8") as handle:
            config = {k.replace("-", "_"): v for k, v in json.load(handle).items()}
        # overwrite with any non-None entries, resolving conflicts in favor of args
        config.update(
            {
                k: v
                for k, v in args.__dict__.items()
                if (v is not None and v != []) or k not in config.keys()
            }
        )
        # replace information in args
        args.__dict__ = config
