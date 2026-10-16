"""
Entry point of the pipeline. The whole pipeline runs with one command, as set in settings.toml:
    python main.py
It reads the configuration (config.toml, config.local.toml, settings.toml), sets up the log, creates the enabled
stages and then the enabled plots with their parameters and lets the orchestrator run them in the order of their inputs,
skipping the ones that are done. Optional subcommands; the options of run override settings.toml for one run:
    python main.py list [--plots] [--version V]
    python main.py run [--plots] [--version V] [--stages K ...] [--from K] [--to K] [--dry-run] [--force]
"""

import argparse
import logging
import os
import sys

from ddtw_ml.config import SETTINGS_FILE, Config, ConfigError, Node, Settings, config_file, load_config, load_settings
from ddtw_ml.log import PIPELINE_LOG, PIPELINE_LOGGER, PIPELINE_LOGGERS, LoggerUtils
from ddtw_ml.pipeline import PipelineError, Stage
from ddtw_ml.pipeline.orchestrator import Orchestrator
from ddtw_ml.pipeline.registry import create_stage, select
from ddtw_ml.pipeline.state import describe


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Function to parse the command line.
    :param argv: arguments (default: sys.argv[1:]).
    :return: parsed arguments; without a subcommand, a run as set in settings.toml.
    """
    parser = argparse.ArgumentParser(prog="python main.py", description="ABP-CBFV pipeline under paced breathing. "
                                     "Without a subcommand: run the pipeline as set in settings.toml.")
    parser.set_defaults(command="run", plots=False, version=None, stages=None, start=None, end=None, dry_run=False,
                        force=False)
    sub = parser.add_subparsers(dest="command")
    for name, help_text in (("list", "show the stages in the order of execution"),
                            ("run", "run the stages; the options override settings.toml")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--plots", action="store_true", help="plots instead of data stages (all, enabled or not)")
        p.add_argument("--version", help="data version (default: [data] version in config.toml)")
    run = sub.choices["run"]
    run.add_argument("--stages", nargs="+", help="only these stages, enabled or not (default: the enabled ones)")
    run.add_argument("--from", dest="start", help="first stage (with everything after it)")
    run.add_argument("--to", dest="end", help="last stage (with everything before it)")
    run.add_argument("--dry-run", action="store_true", help="show the inputs and outputs, run nothing")
    run.add_argument("--force", action="store_true", help="run the selected stages even if they are done")
    return parser.parse_args(argv)


def selected_nodes(config: Config, settings: Settings, args: argparse.Namespace) -> list[Node]:
    """
    Function to choose the stages and plots of the run: the enabled ones (stages, then plots), or the ones named in the
    command line.
    :return: stages and plots in the order of execution.
    """
    if args.plots:
        return [config.plots[k] for k in select(config, "plot", args.stages, args.start, args.end)]
    stages = [config.stages[k] for k in select(config, "stage", args.stages, args.start, args.end)]
    if not args.stages:
        stages = [node for node in stages if settings.enabled(node)]
    plots = [config.plots[k] for k in config.order("plot") if settings.enabled(config.plots[k])]
    return stages + plots


def setup_logging(config: Config, settings: Settings, version: str, dry_run: bool) -> logging.Logger:
    """
    Function to set up the log of the run: console, and {data root}/{version}/pipeline.log (appended) unless it is a
    dry run.
    :return: logger of the orchestrator.
    """
    log_path = None
    if not dry_run:
        os.makedirs(config.data_root / version, exist_ok=True)
        log_path = str(config.data_root / version / PIPELINE_LOG)
    LoggerUtils.setup_pipeline(PIPELINE_LOGGERS, log_path, settings.log_level)
    return logging.getLogger(PIPELINE_LOGGER)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        config = load_config()
        if args.command == "list":
            try:
                analysis = load_settings(config).analysis
            except ConfigError:
                analysis = None  # the stages are listed without checking the changes of [analysis]
            print("\n".join(describe(config, "plot" if args.plots else "stage", args.version, analysis)))
            return 0
        settings = load_settings(config)
        version = config.get_version(args.version)
        dry_run = args.dry_run or settings.dry_run
        force = args.force or settings.force
        if version.frozen and not dry_run:
            raise PipelineError(f"Version {version.name} is frozen; set [data] version to another one.")
        # the parameters of everything that runs are checked when it is created, before the first stage starts
        stages: list[Stage] = [create_stage(config, settings, node.key, node.kind, version.name)
                               for node in selected_nodes(config, settings, args)]
        log = setup_logging(config, settings, version.name, dry_run)
        log.info(f"Settings: {config_file().with_name(SETTINGS_FILE)}")
        Orchestrator(config, version.name, log).run(stages, dry_run, force)
    except (ConfigError, PipelineError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
