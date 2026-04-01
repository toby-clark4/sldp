from __future__ import annotations

import collections.abc
import functools
from typing import Any


_all_memos: list["memoized"] = []


def reset() -> None:
    """Clear all memoized caches registered through this module."""

    for memo in _all_memos:
        memo.reset()


class memoized:
    """Cache the return value of a function by its positional arguments."""

    def __init__(self, func):
        self.func = func
        self.cache: dict[Any, Any] = {}
        _all_memos.append(self)
        functools.update_wrapper(self, func)

    def reset(self) -> None:
        """Clear the memoized cache for this function."""

        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.abc.Hashable):
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]

        value = self.func(*args)
        self.cache[args] = value
        return value

    def __get__(self, obj, objtype):
        """Support instance methods."""

        return functools.partial(self.__call__, obj)
