"""Run, train, or test an xverifier experiment."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import sys

from xverifier.runner import run, score_file, train
from xverifier.store import load_model


def main(argv=None):
    """Parse a command and return a process status."""
    parser = argparse.ArgumentParser(prog="xverifier", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run_command = commands.add_parser("run", help="fit every seed and write records")
    run_command.add_argument("experiment")
    run_command.add_argument("--out", required=True)
    run_command.set_defaults(handler=_run)

    train_command = commands.add_parser("train", help="fit one seed and write a model")
    train_command.add_argument("experiment")
    train_command.add_argument("--seed", required=True, type=int)
    train_command.add_argument("--out", required=True)
    train_command.set_defaults(handler=_train)

    test_command = commands.add_parser("test", help="score a row file with a saved model")
    test_command.add_argument("--rows", required=True)
    test_command.add_argument("--model", required=True)
    test_command.set_defaults(handler=_test)

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code is None else int(exc.code)
    try:
        return args.handler(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def format_table(points):
    """Render operating points as a text table."""
    lines = ["error level   cutoff     decided    errors   error rate"]
    for point in points:
        level = f"{float(point['error_level']):.2f}"
        decided = f"{int(point['decided'])}/{int(point['rows'])}"
        if point["review_all"]:
            lines.append(f"{level:<14}{'review all':<14}{decided}")
            continue
        cutoff = f"{float(point['cutoff']):.6f}"
        if point["error_rate"] is None:
            lines.append(f"{level:<14}{cutoff:<11}{decided}")
            continue
        errors = f"{int(point['errors']):>2}"
        rate = f"{float(point['error_rate']):.3f}"
        prefix = f"{level:<14}{cutoff:<11}{decided:<11}{errors}"
        if len(prefix) < 45:
            lines.append(prefix + (" " * (45 - len(prefix))) + rate)
        else:
            lines.append(prefix + "  " + rate)
    return "\n".join(lines)


def _run(args):
    results = run(args.experiment, args.out)
    blocks = []
    show_seed = len(results) > 1
    for seed, result in results.items():
        table = format_table(result["operating_points"])
        blocks.append(f"seed {seed}\n{table}" if show_seed else table)
    print("\n\n".join(blocks))
    return 0


def _train(args):
    train(args.experiment, args.seed, args.out)
    return 0


def _test(args):
    scored = score_file(args.rows, load_model(args.model))
    print(format_table(scored["operating_points"]))
    return 0
