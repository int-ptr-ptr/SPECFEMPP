import json
import os
import sys
import time

from workflow.simrunner.tasks import SpecfemEMTask
from workflow.util import runjob

workdir = os.path.dirname(__file__)


def get_runsim_task(
    sim_dir: str, database_in: str, dt: float, nstep: int, subdivs: int | None
):
    parfile = "specfem_config.yaml"

    def set_parfile():
        with open(os.path.join(workdir, parfile), "w") as f:
            f.write(
                sf_config(
                    dt=dt,
                    nstep=nstep,
                    outfolder="OUTPUT_FILES",
                    out_seismo=sim_dir,
                    out_disp=None,
                    stations="OUTPUT_FILES/stations",
                    seismo_step_between_samples=1,
                    database_in=database_in,
                    source_in="source.yaml",
                    domain_sep_subdivision=subdivs,
                )
            )
        # output files
        outdir = os.path.join(workdir, "OUTPUT_FILES", sim_dir)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        else:
            # clean
            for f in os.listdir(outdir):
                os.remove(os.path.join(outdir, f))

    task = SpecfemEMTask(
        sim_dir,
        group="gpu",
        specfem_parfile=parfile,
        cwd=workdir,
        on_pre_run=set_parfile,
        additional_args="--lr_periodic",
    )
    return task


DT = 1e-3
NSTEP = int(1.5 / DT)


def call():
    data = None
    with open(os.path.join(workdir, "meshconf.json"), "r") as f:
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
        tasks.append(get_runsim_task(f"cont{suff}", database, DT, NSTEP, None))
        tasks.append(get_runsim_task(f"dg{suff}", database, DT, NSTEP, 1))

    for task in tasks:
        task.on_pre_run()
        task.job.print_updates = True

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
            sys.exit(1)
        print("job complete")


def sf_config(
    dt: float,
    nstep: float,
    outfolder: str,
    out_seismo: str,
    out_disp: str | None,
    stations: str,
    seismo_step_between_samples: int,
    database_in: str,
    source_in: str,
    domain_sep_subdivision: int | None,
):
    if out_disp is None:
        dispstr = ""
    else:
        dispstr = f"""
          display:
            format: PNG
            directory: {os.path.join(outfolder, out_disp)}
            field: displacement
            simulation-field: forward
            time-interval: 100
"""
    if domain_sep_subdivision is None:
        meshmod = ""
    else:
        meshmod = f"""
  mesh-modifiers:
    subdivisions:
      - material: 2
        x: {domain_sep_subdivision}
        z: {domain_sep_subdivision}
    interface-rules:
      - material1: 1
        material2: 2
        rule: domain-separation
"""

    return f"""parameters:
  header:
    title: FF_snell
    description: |
      Material systems : Acoustic domain (2)
      Interfaces : ...
      Sources : ...
      Boundary conditions : ...

  simulation-setup:
    quadrature:
      quadrature-type: GLL4

    solver:
      time-marching:
        type-of-simulation: forward
        time-scheme:
          type: Newmark
          dt: {dt:e}
          nstep: {nstep}

    simulation-mode:
      forward:
        writer:
          seismogram:
            format: ascii
            directory: {os.path.join(outfolder, out_seismo)}
{dispstr}

  receivers:
    stations: {stations}
    angle: 0.0
    seismogram-type:
      - pressure
    nstep_between_samples: {seismo_step_between_samples}

  run-setup:
    number-of-processors: 1
    number-of-runs: 1

  databases:
    mesh-database: {database_in}

  sources: {source_in}
{meshmod}
"""


if __name__ == "__main__":
    call()
