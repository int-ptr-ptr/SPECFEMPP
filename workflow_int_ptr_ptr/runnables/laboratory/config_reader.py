from util.task_manager import Task
import simrunner.tasks as simtask
import os
import json

EXPERIMENT_CONFIG_FILENAME = "experiment.json"


def experiment_to_tasks(
    folder: str, experiment_name: str, verbose: bool = False
) -> list[Task]:
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
                res.append(
                    simtask.Specfem2DTask(
                        tname, cwd=os.path.join(folder, cwd), specfem_parfile=parfile
                    )
                )
            elif kind == "specfemem" or kind == "specfem2d_eventmarcher":
                log(f'    specfem2d_eventmarcher task "{tname}"')
                res.append(
                    simtask.SpecfemEMTask(
                        tname, cwd=os.path.join(folder, cwd), specfem_parfile=parfile
                    )
                )
            else:
                raise IOError(f"Unknown type: {kind}")

        log("Setting dependencies:")

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
            log()
        return res

    except IOError as e:
        raise IOError(
            f"Failure to read experiment {experiment_name} in folder {folder}."
        ) from e
