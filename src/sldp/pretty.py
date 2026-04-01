from __future__ import annotations

from argparse import Namespace


def print_namespace(ns: Namespace, name_width: int = 25) -> None:
    """Print argparse namespace values in a stable aligned format."""

    for name, value in sorted(vars(ns).items()):
        if not name.startswith("_"):
            print(f"{name:<{name_width}}{value}")
