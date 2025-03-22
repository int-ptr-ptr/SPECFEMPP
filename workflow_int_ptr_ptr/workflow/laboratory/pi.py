import os

import workflow.util.config as config
from workflow.util.task_manager import Manager, Task

from .config_reader import experiment_to_tasks, is_experiment_folder

"""
Manages the laboratory.
"""

ROOT_DIR = config.get("root_dir")
LABORATORY_DIR = os.path.join(ROOT_DIR, "laboratory")
EXPERIMENTS: dict[str, str] | None = None


def get_experiment_list(verbose: bool = False) -> dict[str, str]:
    global EXPERIMENTS
    if EXPERIMENTS is not None:
        return EXPERIMENTS

    experiments = dict()

    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    def recursive_find(folder: str, experiment_heirarchy: str, depth: int):
        header = " │" * depth
        for entry in os.listdir(folder):
            if "." in entry:
                # no experiment folder can have a dot,
                # since that's the name separator (also, skip most files)
                continue
            file = os.path.join(folder, entry)
            if os.path.isdir(file):
                cur_heirarchy = (
                    f"{experiment_heirarchy}.{entry}" if depth > 0 else entry
                )
                if is_experiment_folder(file):
                    log(f"{header} ├ [{entry}] FOUND ({cur_heirarchy})")
                    # this is an experiment. Append it
                    experiments[cur_heirarchy] = file
                else:
                    log(f"{header} ├ {entry} (searching)")
                    recursive_find(file, cur_heirarchy, depth + 1)

    log(f'initializing search for experiments in laboratory "{LABORATORY_DIR}"')
    recursive_find(LABORATORY_DIR, "", 0)
    EXPERIMENTS = experiments
    return experiments


def tasks_in_experiment(name: str, verbose: bool = False) -> list[Task]:
    if name in get_experiment_list(verbose=True):
        return experiment_to_tasks(get_experiment_list()[name], name, verbose=verbose)
    else:
        return []


def full_run_experiment(name: str, use_gui: bool = False, verbose: bool = False):
    if name in get_experiment_list():
        manager = Manager(
            use_gui=use_gui,
            tasks=tasks_in_experiment(name),
            verbose=verbose,
            sequential_groups=["gpu"],
        )
        manager.run()
