import datetime
import json
import os
import re
import time
from typing import Callable

import workflow.simrunner.tasks as simtask
from workflow.util.runjob import SystemCommandJob
from workflow.util.task_manager import Task

EXPERIMENT_CONFIG_FILENAME = "experiment.json"


def file_deps_from_meshfem_parfile(parfile: str) -> tuple[list[str], list[str]]:
    deps = [parfile]
    outs = []
    with open(parfile, "r") as f:
        for line in f:
            if m := re.match(r"database_filename\s*=\s*(.*)\s*$", line):
                outs.append(m.group(1))
            elif m := re.match(r"stations_filename\s*=\s*(.*)\s*$", line):
                outs.append(m.group(1))
            elif m := re.match(r"interfacesfile\s*=\s*(.*)\s*$", line):
                deps.append(m.group(1))
    return deps, outs


def file_deps_from_specfem_parfile(parfile: str) -> tuple[list[str], list[str]]:
    deps = [parfile]
    outs = []
    with open(parfile, "r") as f:
        for line in f:
            if m := re.match(r"\s*directory:\s*(.*)\s*$", line):
                outs.append(m.group(1))
            elif m := re.match(r"\s*mesh-database:\s*(.*)\s*$", line):
                deps.append(m.group(1))
            elif m := re.match(r"\s*stations:\s*(.*)\s*$", line):
                deps.append(m.group(1))
            elif m := re.match(r"\s*sources:\s*(.*)\s*$", line):
                f = m.group(1)
                if f.endswith(".yaml") or f.endswith(".yml"):
                    deps.append(m.group(1))
    return deps, outs


def specfem_clear_output_folders(parfile: str, cwd: str | None = None):
    if cwd is None:
        cwd = os.path.dirname(parfile)
    else:
        parfile = os.path.join(cwd, parfile)
    # folders to clear
    folders = []

    with open(parfile, "r") as f:
        for line in f:
            if m := re.match(r"\s*directory:\s*(.*)\s*$", line):
                folders.append(os.path.join(cwd, m.group(1)))

    for foldername in folders:
        if os.path.exists(foldername):
            for f in os.listdir(foldername):
                os.remove(os.path.join(foldername, f))
        else:
            os.makedirs(foldername)


class ScriptTask(Task):
    def __init__(
        self,
        title: str,
        command: str,
        group: str | None = None,
        cwd: str | None = None,
        dependencies: list["Task"] | None = None,
        on_completion: Callable[[int], None] | None = None,
        on_pre_run: Callable[[], None] | None = None,
    ):
        self.title = title
        name = f"{title} (specfem2d)"
        if group is None:
            group = "unnamed mesher"

        # pass kwargs or use defaults for Specfem2DJob
        job = SystemCommandJob(name, cmd=command, cwd=cwd)
        self.cwd = cwd
        super().__init__(
            job,
            name=name,
            group=group,
            dependencies=dependencies,
            on_completion=on_completion,
            on_pre_run=on_pre_run,
        )


def experiment_to_tasks(
    folder: str,
    experiment_name: str,
    run_necessary_only: bool = True,
    verbose: bool = False,
) -> list[Task]:
    """Converts experiments to task list.

    Args:
        folder (str): folder for the experiment
        experiment_name (str): name of the experiment (used for task names)
        run_necessary_only (bool, optional): as in Makefiles, only runs targets that have
                missing/outdated files when true. Defaults to True.
        verbose (bool, optional): Whether or not to log everything to stdout. Defaults to False.

    Returns:
        list[Task]: the generated list of tasks
    """

    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    res = list()
    try:
        with open(os.path.join(folder, EXPERIMENT_CONFIG_FILENAME), "r") as f:
            data = json.load(f)

        log("Successfully retrieved file contents.")
        if "tasks" not in data:
            raise IOError('array entry "tasks" must exist in config.')

        task_by_id = (
            dict()
        )  # ids in json file, may not correspond to naming: only one at a time
        num_instances = (
            dict()
        )  # number of instances of each kind -- used for default naming

        log("Initializing tasks.")
        for itask, task in enumerate(data["tasks"]):
            log(f"  Reading task {itask}:")
            tid = None
            if "id" in task:
                tid = str(task["id"])
                if tid in task_by_id:
                    del task_by_id[tid]
                    tid = None
                else:
                    task_by_id[tid] = len(res)
            kind = task["type"]
            if kind not in num_instances:
                num_instances[kind] = 0
            num_instances[kind] += 1
            tname = f"{kind} {num_instances[kind]}" if tid is None else tid

            # Not needed in every type, but good to retrieve now:
            parfile = None
            if "parfile" in task:
                parfile = task["parfile"]
            cwd = "."
            if "cwd" in task:
                cwd = task["cwd"]

            if kind == "mesher":
                log(f'    mesher task "{tname}"')
                res.append(
                    simtask.MesherTask(
                        tname, cwd=os.path.join(folder, cwd), meshfem_parfile=parfile
                    )
                )
            elif kind == "specfem2d":
                log(f'    specfem2d task "{tname}"')
                # store pre-run task to init/clear output folders
                t = simtask.Specfem2DTask(
                    tname,
                    cwd=os.path.join(folder, cwd),
                    specfem_parfile=parfile,
                )
                t.on_pre_run = lambda t=t: specfem_clear_output_folders(
                    parfile=t.parfile, cwd=t.cwd
                )
                res.append(t)
            elif kind == "specfemem" or kind == "specfem2d_eventmarcher":
                log(f'    specfem2d_eventmarcher task "{tname}"')
                t = simtask.SpecfemEMTask(
                    tname, cwd=os.path.join(folder, cwd), specfem_parfile=parfile
                )
                t.on_pre_run = lambda t=t: specfem_clear_output_folders(
                    parfile=t.parfile, cwd=t.cwd
                )
                res.append(t)
            elif kind == "script":
                log(f'    script task "{tname}"')
                res.append(
                    ScriptTask(tname, task["command"], cwd=os.path.join(folder, cwd))
                )
            else:
                raise IOError(f"Unknown type: {kind}")

        log("Setting dependencies:")
        task_backlinks = [[] for _ in res]
        for itask, task in enumerate(data["tasks"]):
            log(f"  task {itask}:", end="")
            deps = []
            if "deps" in task:
                deps = task["deps"]

            for dep in deps:
                if isinstance(dep, str):
                    if dep not in task_by_id:
                        raise IOError(
                            f'dependency string "{dep}" given, but cannot resolve this ID'
                        )
                    log(f" {task_by_id[dep]} ({dep}),", end="")
                    dep = task_by_id[dep]
                elif isinstance(dep, int):
                    log(f" {dep},", end="")
                res[itask].dependencies.append(res[dep])
                task_backlinks[dep].append(itask)
            log()

        # prune unneccesary nodes now that dependency lists are not linked by index
        if run_necessary_only:
            log("Pruning unneeded experiments:")
            to_prune = []

            # if intime > outtime, store here, so we dont need to check multiple times
            dont_prune_from_filemod = []

            def prune(itask: int):
                task = res[itask]
                if itask in to_prune:
                    # already pruned
                    return
                if itask in dont_prune_from_filemod:
                    # already checked to not prune
                    return
                if len(task.dependencies) > 0:
                    # unpruned dependency; keep
                    return

                log(f"  checking if we need to prune task {itask} ({task.title})")

                if isinstance(task, simtask.MesherTask):
                    parfile = task.parfile
                    if task.cwd is not None:
                        parfile = os.path.join(task.cwd, parfile)
                    filedeps, fileouts = file_deps_from_meshfem_parfile(parfile)
                elif isinstance(task, simtask.Specfem2DTask) or isinstance(
                    task, simtask.SpecfemEMTask
                ):
                    parfile = task.parfile
                    if task.cwd is not None:
                        parfile = os.path.join(task.cwd, parfile)
                    filedeps, fileouts = file_deps_from_specfem_parfile(parfile)
                else:
                    filedeps = []
                    fileouts = []
                if "files_in" in data["tasks"][itask]:
                    filedeps = data["tasks"][itask]["files_in"]

                if "files_out" in data["tasks"][itask]:
                    fileouts = data["tasks"][itask]["files_out"]

                if len(fileouts) == 0 or len(filedeps) == 0:
                    # always run these.
                    log("    no files on either input or output side. always run these")
                    dont_prune_from_filemod.append(itask)
                    return

                def read_timestamp(file, is_in=False, default_time=time.time()):
                    fullpath = os.path.join(folder, file)
                    log(f"      file {file}")
                    if not os.path.exists(fullpath):
                        timestr = datetime.datetime.fromtimestamp(
                            default_time
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        log(f"        Cannot find. Assuming t = {timestr}")
                        return default_time
                    t = os.path.getmtime(fullpath)
                    timestr = datetime.datetime.fromtimestamp(t).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    log(f"        @ {timestr}")
                    return t

                # last time of input modifications
                log("    dependent files (and their times):")
                intime = max(read_timestamp(dep, is_in=True) for dep in filedeps)

                # first time of output modifications
                log("    output files (and their times):")
                outtime = min(
                    read_timestamp(dep, default_time=intime - 1) for dep in fileouts
                )

                if intime >= outtime:
                    deltat = datetime.timedelta(seconds=intime - outtime)
                    log(
                        f"    not pruning: dep modified time - output modified time: {deltat}"
                    )
                    dont_prune_from_filemod.append(itask)
                else:
                    deltat = datetime.timedelta(seconds=outtime - intime)
                    log(
                        f"    pruning: output modified time - dep modified time: {deltat}"
                    )
                    to_prune.append(itask)
                    for dep in task_backlinks[itask]:
                        res[dep].dependencies.remove(task)
                        prune(dep)

            for i in range(len(res)):
                prune(i)

            to_prune.sort()
            for itask in reversed(to_prune):
                del res[itask]
        return res

    except IOError as e:
        raise IOError(
            f"Failure to read experiment {experiment_name} in folder {folder}."
        ) from e
