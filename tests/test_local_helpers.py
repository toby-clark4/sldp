from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from sldp import fs, memo, pretty


class TestFsHelpers:
    def test_makedir_for_file_creates_parent_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "file.txt"

        fs.makedir_for_file(target)

        assert target.parent.exists()


class TestMemoHelpers:
    def test_reset_clears_registered_memoized_cache(self) -> None:
        calls: list[int] = []

        @memo.memoized
        def square(x: int) -> int:
            calls.append(x)
            return x * x

        assert square(3) == 9
        assert square(3) == 9
        assert calls == [3]

        memo.reset()

        assert square(3) == 9
        assert calls == [3, 3]

    def test_memoized_supports_keyword_arguments(self) -> None:
        calls: list[tuple[int, int]] = []

        @memo.memoized
        def add(x: int, y: int = 0) -> int:
            calls.append((x, y))
            return x + y

        assert add(2, y=3) == 5
        assert add(2, y=3) == 5
        assert calls == [(2, 3)]


class TestPrettyHelpers:
    def test_print_namespace_omits_private_attributes(self, capsys) -> None:
        ns = Namespace(alpha=1, beta="two")
        setattr(ns, "_private", "hidden")

        pretty.print_namespace(ns)

        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" in out
        assert "_private" not in out
