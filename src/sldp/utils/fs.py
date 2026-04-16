# File management utilities
import os
import resource
from pathlib import Path


def makedir(path_to_dir: str | os.PathLike[str]) -> None:
    """Create a directory if it does not already exist."""

    Path(path_to_dir).mkdir(parents=True, exist_ok=True)


def makedir_for_file(path_to_file: str | os.PathLike[str]) -> None:
    """Create the parent directory required to write a file path."""

    parent = Path(path_to_file).parent
    if str(parent) not in {"", "."}:
        parent.mkdir(parents=True, exist_ok=True)


def mem() -> float:
    """Return the current process peak resident memory in MB."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
