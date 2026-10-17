"""
Orchestrator of the pipeline: runs the given stages (and plots) in their order, in one data version, skipping the ones
that are done (see state.py) unless the run is forced; a stage whose input was run again in the same run, or later,
or whose parameters of the analysis ([analysis]) changed, runs again. A done stage whose own parameters changed since
its last run is skipped with a warning. The output directory of a stage is cleared before it runs.

It controls the run only: the stages are created by the registry (registry.py) and the logging is set up by the caller
(main.py); the orchestrator logs the run to "pipeline", every stage logs its own work to "stage.<key>" or "plot.<key>".
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path

from ddtw_ml.config import Config, Node
from ddtw_ml.log import PIPELINE_LOGGER
from ddtw_ml.pipeline.stage import PipelineError, Stage, StageReport, output_dir
from ddtw_ml.pipeline.state import analysis_changes, is_done, next_run, read_state, state_key, write_state


def clear_output(config: Config, node: Node, version: str) -> Path | None:
    """
    Function to remove the output directory of a stage (or plot) before it runs, so that no file of an earlier run is
    left in it.
    :param config: configuration.
    :param node: stage or plot.
    :param version: data version.
    :return: removed directory (None if it did not exist).
    :raise PipelineError: if the directory is the raw data, a data or plot root, contains one of them or is an input
        of the stage.
    """
    out = output_dir(config, node, version).resolve()
    if out.is_file():
        out.unlink()
        return out
    if not out.is_dir():
        return None
    protected = [config.raw_dir(), config.data_root, config.plots_root, config.data_root / version,
                 config.plots_root / version] + [config.stage_dir(i, version) for i in node.inputs]
    for path in protected:
        path = Path(path).resolve()
        if out == path or out in path.parents:
            raise PipelineError(f"Output directory of '{node.key}' ({out}) is not removed: it is or contains {path}.")
    shutil.rmtree(out)
    return out


class Orchestrator:
    def __init__(self, config: Config, version: str | None = None, log: logging.Logger | None = None) -> None:
        """
        Constructor of the Orchestrator class.
        :param config: configuration.
        :param version: data version (default: the active one).
        :param log: logger of the run (default: "pipeline").
        """
        self.config = config
        self.version = config.get_version(version)
        self.log = log or logging.getLogger(PIPELINE_LOGGER)

    def run(self, stages: list[Stage], dry_run: bool = False, force: bool = False) -> list[StageReport]:
        """
        Method to run the stages in the given order, skipping the ones that are done.
        :param stages: stages and plots in the order of execution (stages before the plots that show them).
        :param dry_run: only log the inputs and outputs and which stages would be skipped, run nothing.
        :param force: run the stages even if they are done.
        :return: reports of the stages that ran.
        :raise PipelineError: if the version is frozen, a stage belongs to another version or an input is missing.
        """
        config, version, log = self.config, self.version.name, self.log
        if self.version.frozen and not dry_run:
            raise PipelineError(f"Version {version} is frozen; set [data] version to another one.")
        for stage in stages:
            if stage.ctx.version != version:
                raise PipelineError(f"'{stage.name}' is created for version {stage.ctx.version}, not {version}.")
        state = read_state(config, version)
        ran: set[str] = set()
        reports = []
        skipped = []
        log.info(f"##### Version {version}: {len(stages)} to check{' (dry run)' if dry_run else ''}: "
                 f"{', '.join(stage.name for stage in stages)} #####")
        for step, stage in enumerate(stages, start=1):
            node, name = stage.ctx.node, stage.name
            params = stage.ctx.params
            log.info(f"=== [{step}/{len(stages)}] {name} ({node.target}) ===")
            for i in node.inputs:
                path = config.stage_dir(i, version)
                log.info(f"  input  {i}: {path}{'' if path.is_dir() else ' (missing)'}")
            log.info(f"  output {node.key}: {output_dir(config, node, version)}")
            if params:
                log.info(f"  params {params}")
            analysis = stage.ctx.settings.analysis
            entry = state.get(state_key(node))
            if entry is not None and analysis_changes(entry, analysis):
                log.info(f"  parameters of the analysis changed since the last run: "
                         f"{', '.join(analysis_changes(entry, analysis))}")
            if not force and is_done(config, node, version, state, ran, analysis):
                entry = state[state_key(node)]
                if "params" in entry and entry["params"] != params:
                    log.warning(f"  {name} was done with other parameters {entry['params']}; its output does not "
                                f"follow settings.toml (force = true in settings.toml or --force to run it again)")
                log.info(f"=== {name}: skipped, done {entry['finished']} "
                         f"(force = true in settings.toml or --force to run it again) ===")
                skipped.append(name)
                continue
            if node.kind == "stage":
                ran.add(node.key)
            if dry_run:
                log.info(f"=== {name}: would run ===")
                continue
            missing = [i for i in node.inputs if not config.stage_dir(i, version).is_dir()]
            if missing:
                raise PipelineError(f"Missing input of '{name}': {missing}. Run the earlier stages first.")
            # a stage that fails is not done, even if an earlier run of it finished
            state.pop(state_key(node), None)
            write_state(config, version, state)
            cleared = clear_output(config, node, version)
            if cleared is not None:
                log.info(f"  cleared the output of the last run: {cleared}")
            try:
                stage.run()
            except Exception:
                log.error(f"=== {name}: failed (the pipeline stops; see the traceback) ===")
                raise
            report = stage.ctx.report
            log.info(f"=== {name}: {report.processed} processed, {len(report.rejected)} rejected ===")
            state[state_key(node)] = {"finished": datetime.now().isoformat(timespec="microseconds"),
                                      "run": next_run(state), "processed": report.processed, "rejected": len(report.rejected),
                                      "params": params, "analysis": stage.ctx.analysis_used}
            write_state(config, version, state)
            reports.append(report)
        action = "would run" if dry_run else "ran"
        log.info(f"##### Finished version {version}: {len(stages) - len(skipped)} {action}, {len(skipped)} skipped"
                 f"{' (done: ' + ', '.join(skipped) + ')' if skipped else ''} #####")
        return reports
