import json
import os
import shutil
import sys
import time

from workflow.laboratory.parfilegen import sf_config
from workflow.simrunner.tasks import SpecfemEMTask
from workflow.util import dump_reader, runjob

workdir = os.path.dirname(__file__)
output_folname = "OUTPUT_FILES"
output_fol = os.path.join(workdir, output_folname)


def get_runsim_task(
    sim_dir: str,
    database_in: str,
    dt: float,
    nstep: int,
    subdivs: int | None,
    steps_per_dump: int | None = None,
):
    if steps_per_dump is None:
        steps_per_dump = max(1, int(0.05 / dt))
    parfile = "specfem_config.yaml"
    # output files
    outdir = os.path.join(workdir, output_folname, sim_dir)

    def set_parfile():
        with open(os.path.join(workdir, parfile), "w") as f:
            f.write(
                sf_config(
                    dt=dt,
                    nstep=nstep,
                    outfolder=output_folname,
                    out_seismo=sim_dir,
                    out_disp=None,
                    stations="OUTPUT_FILES/stations",
                    seismo_step_between_samples=1,
                    database_in=database_in,
                    source_in="source.yaml",
                    domain_sep_subdivision=subdivs,
                )
            )
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        else:
            # clean
            for f in os.listdir(outdir):
                os.remove(os.path.join(outdir, f))

    def compile_series(completestate):
        if completestate == 0:
            series = dump_reader.load_series(os.path.join(outdir, "d"))
            for f in os.listdir(outdir):
                if f.startswith("d") and f.endswith(".dat"):
                    os.remove(os.path.join(outdir, f))
            series.save_to_file(os.path.join(outdir, "dumps.dat"))

    task = SpecfemEMTask(
        sim_dir,
        group="gpu",
        specfem_parfile=parfile,
        cwd=workdir,
        on_pre_run=set_parfile,
        on_completion=compile_series,
        additional_args=f"--lr_periodic --absorb_top --absorb_bottom --dumpfolder {outdir} --dump {steps_per_dump} --kill_boundaries",
    )
    return task, dt, subdivs, sim_dir


DT = 1e-3
NSTEP = int(1.5 / DT)


def call():
    # clean
    for f in os.listdir(output_fol):
        fullpath = os.path.join(output_fol, f)
        if os.path.isdir(fullpath):
            if f.startswith("cont") or f.startswith("dg"):
                shutil.rmtree(fullpath)

    data = None
    with open(os.path.join(output_fol, "meshconf.json"), "r") as f:
        data = json.load(f)

    if data is None:
        print("meshconf.json not found")
        return

    tasks = []
    for mesh in data["meshes"]:
        database = mesh["database_file"]
        nx = mesh["nx"]
        # vp2 = mesh["vp2"]
        vpind = mesh["vp2_ind"]
        suff = f"{nx}_{vpind}"

        def add_simtask(taskret):
            auxdata = {
                "sim": taskret[3],
                "N_size": nx,
                "vp2": mesh["vp2"],
                "vp2_ind": vpind,
                "dt": taskret[1],
                "subdivisions2": taskret[2],
            }
            tasks.append((taskret[0], auxdata))

        add_simtask(get_runsim_task(f"cont_{suff}", database, DT, NSTEP, None))
        add_simtask(get_runsim_task(f"dg1_{suff}", database, DT, NSTEP, 1))
        add_simtask(get_runsim_task(f"dg2_{suff}", database, DT, NSTEP, 2))
        add_simtask(get_runsim_task(f"dg3_{suff}", database, DT, NSTEP, 3))

    task_completes = list()
    failure = None
    for task, auxdata in tasks:
        task.on_pre_run()
        task.job.print_updates = True

        tstart = time.time()

        print(f"starting job {task.name}")
        jid = runjob.queue_job(task.job)
        lines = []
        last_numlines = 0
        while runjob.is_job_running(jid):
            time.sleep(1)
            lines.extend(runjob.consume_queue(jid))
            if len(lines) > last_numlines:
                print(lines[-1])
                last_numlines = len(lines)

        if runjob.complete_job(jid, error_on_still_running=True) > 0:
            print("Job failed:")
            print(*lines, sep="")
            failure = {"job": task.name, "output": "".join(lines)}
            break

        tend = time.time()

        task_completes.append(
            {
                "time": int(
                    round(tend - tstart)
                ),  # seconds, since we use the time.sleep(1) call.
            }
            | auxdata
        )

        task.on_completion(0)
        print("job complete")

    outlog: dict[str, list | dict] = {"tasks_completed": task_completes}
    if failure:
        outlog["failure"] = failure

    with open(os.path.join(output_fol, "run_out.json"), "w") as f:
        json.dump(outlog, f)

    # clean up parfile
    os.remove(os.path.join(workdir, "specfem_config.yaml"))

    if failure:
        with open(os.path.join(output_fol, "run_err.txt"), "w") as f:
            f.write(failure["output"])
        sys.exit(1)


if __name__ == "__main__":
    call()
