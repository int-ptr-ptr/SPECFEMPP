import os

import workflow.simrunner.tasks as tasks

workdir = os.path.dirname(__file__)


def get_runsim_task(sim_dir: str, database_in: str):
    title = ""
    dt = 0.01
    nstep = 150
    parfile = "specfem_config.yaml"

    def set_parfile():
        with open(os.path.join(workdir, parfile), "w") as f:
            f.write(
                sf_config(
                    dt=dt,
                    nstep=nstep,
                    outfolder="OUTPUT_FILES",
                    out_seismo=os.path.join(sim_dir),
                    out_disp=None,
                    stations="OUTPUT_FILES/stations",
                    seismo_step_between_samples=1,
                    database_in=database_in,
                    source_in="source.yaml",
                    domain_sep_subdivision=None,
                )
            )

    task = tasks.SpecfemEMTask(
        title, group="gpu", specfem_parfile=parfile, cwd=workdir, on_pre_run=set_parfile
    )
    return task


def call():
    pass


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
