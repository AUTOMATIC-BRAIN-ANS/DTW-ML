"""
Command line of the pipeline:
    python -m ddtw_ml list [--plots] [--version V]
    python -m ddtw_ml run [--plots] [--version V] [--stages K ...] [--from K] [--to K] [--dry-run]
"""

import argparse
import sys

from ddtw_ml.config import ConfigError, load_config
from ddtw_ml.pipeline import PipelineError, describe, run_pipeline, select


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Function to parse the command line.
    :param argv: arguments (default: sys.argv[1:]).
    :return: parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="python -m ddtw_ml", description="ABP-CBFV pipeline under paced breathing.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("list", "show the stages in the order of execution"), ("run", "run the stages")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--plots", action="store_true", help="plots instead of data stages")
        p.add_argument("--version", help="data version (default: [data] version in config.toml)")
    run = sub.choices["run"]
    run.add_argument("--stages", nargs="+", help="only these stages")
    run.add_argument("--from", dest="start", help="first stage (with everything after it)")
    run.add_argument("--to", dest="end", help="last stage (with everything before it)")
    run.add_argument("--dry-run", action="store_true", help="show the inputs and outputs, run nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    kind = "plot" if args.plots else "stage"
    try:
        config = load_config()
        if args.command == "list":
            print("\n".join(describe(config, kind, args.version)))
            return 0
        keys = select(config, kind, args.stages, args.start, args.end)
        run_pipeline(config, keys, kind, args.version, args.dry_run)
    except (ConfigError, PipelineError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
