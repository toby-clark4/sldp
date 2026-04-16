import collections.abc
import functools
import weakref
from contextlib import contextmanager
from typing import Any


_all_memos: "weakref.WeakSet[memoized]" = weakref.WeakSet()


def reset() -> None:
    """Clear all memoized caches registered through this module."""

    for memo in list(_all_memos):
        memo.reset()


@contextmanager
def cache_scope() -> collections.abc.Iterator[None]:
    """Provide an explicit memoization lifetime for workflow code and tests."""

    try:
        yield
    finally:
        reset()


class memoized:
    """Cache the return value of a function by its positional and keyword arguments."""

    def __init__(self, func: collections.abc.Callable[..., Any]) -> None:
        self.func = func
        self.cache: dict[Any, Any] = {}
        _all_memos.add(self)
        functools.update_wrapper(self, func)

    def reset(self) -> None:
        """Clear the memoized cache for this function."""

        self.cache = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self._cache_key(args, kwargs)
        if key is None:
            return self.func(*args, **kwargs)
        if key in self.cache:
            return self.cache[key]

        value = self.func(*args, **kwargs)
        self.cache[key] = value
        return value

    def __get__(self, obj: Any, objtype: type[Any] | None) -> collections.abc.Callable[..., Any]:
        """Support instance methods."""

        del objtype
        return functools.partial(self.__call__, obj)

    @staticmethod
    def _cache_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...] | None:
        """Build a hashable cache key or return `None` if inputs are not hashable."""

        keyword_items = tuple(sorted(kwargs.items()))
        key = (args, keyword_items)
        if not isinstance(key, collections.abc.Hashable):
            return None
        return key
