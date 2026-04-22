from __future__ import annotations

from sldp.utils import multiproc


class TestExecuteTasks:
    def test_execute_tasks_runs_serially_in_order_when_num_proc_is_one(self, monkeypatch) -> None:
        called = False

        class UnexpectedExecutor:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs
                nonlocal called
                called = True

        monkeypatch.setattr(multiproc, "ProcessPoolExecutor", UnexpectedExecutor)

        results = multiproc.execute_tasks([1, 2, 3], lambda task: task * 10, 1)

        assert results == [10, 20, 30]
        assert called is False

    def test_execute_tasks_uses_process_pool_when_num_proc_exceeds_one(self, monkeypatch) -> None:
        submitted: list[int] = []
        max_workers_seen: list[int] = []

        class FakeFuture:
            def __init__(self, result: int) -> None:
                self._result = result

            def result(self) -> int:
                return self._result

        class FakeExecutor:
            def __init__(self, *, max_workers: int) -> None:
                max_workers_seen.append(max_workers)

            def __enter__(self) -> FakeExecutor:
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                del exc_type, exc, tb

            def submit(self, worker_fn, task: int) -> FakeFuture:
                submitted.append(task)
                return FakeFuture(worker_fn(task))

        monkeypatch.setattr(multiproc, "ProcessPoolExecutor", FakeExecutor)

        results = multiproc.execute_tasks([1, 2, 3], lambda task: task + 1, 2)

        assert results == [2, 3, 4]
        assert submitted == [1, 2, 3]
        assert max_workers_seen == [2]
