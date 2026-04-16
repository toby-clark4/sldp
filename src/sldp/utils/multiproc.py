import argparse

def validate_num_proc(args: argparse.Namespace) -> argparse.Namespace:
    """Validate that num_proc is a positive integer."""
    if args.num_proc < 1:
        print("Warning: num_proc must be a positive integer. Setting num_proc to 1.")
        args.num_proc = 1
    return args