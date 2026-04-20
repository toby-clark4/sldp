import argparse
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor


def validate_num_proc(args: argparse.Namespace) -> argparse.Namespace:
    """Validate that num_proc is a positive integer."""

    if args.num_proc < 1:
        print("Warning: num_proc must be a positive integer. Setting num_proc to 1.")
        args.num_proc = 1
    return args


def execute_tasks(tasks: Iterable[object], worker_fn: Callable[[object], object], num_proc: int) -> list[object]:
    """Execute tasks serially or with a process pool, preserving task order."""

    task_list = list(tasks)
    if num_proc == 1:
        return [worker_fn(task) for task in task_list]

    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = [executor.submit(worker_fn, task) for task in task_list]
        return [future.result() for future in futures]
