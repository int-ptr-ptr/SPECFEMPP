import os
import re
import shutil
import sys
import time

from experiment import Simulation  # pyright: ignore

from workflow.laboratory.parfilegen import sf_config
from workflow.simrunner.tasks import SpecfemEMTask
from workflow.util import dump_reader, gpulock, runjob

workdir = os.path.dirname(__file__)
output_folname = "OUTPUT_FILES"
output_fol = os.path.join(workdir, output_folname)


def get_runsim_task(
    sim_dir: str,
    database_in: str,
    dt: float,
    nstep: int,
    scheme: str,
    scheme_params: None | dict,
    subdivs: tuple[int, int] | None,
    steps_per_dump: int | None = None,
    use_gpu: bool = False,
    use_new_abs: bool = True,
):
    if steps_per_dump is None:
        steps_per_dump = max(1, int(0.05 / dt))
    parfile = f"specfem_config_{sim_dir}.yaml"
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
                fullpath = os.path.join(outdir, f)
                if os.path.isfile(fullpath):
                    os.remove(fullpath)
                else:
                    shutil.rmtree(fullpath)

    def compile_series(completestate):
        if completestate == 0:
            series = dump_reader.load_series(os.path.join(outdir, "d"))
            for f in os.listdir(outdir):
                if f.startswith("d") and f.endswith(".dat"):
                    os.remove(os.path.join(outdir, f))
            series.save_to_file(os.path.join(outdir, "dumps.dat"))

    if scheme_params is None:
        scheme_params = dict()
    afmode = ""
    if scheme == "symm" or scheme == "sym":
        afmode = " --acoustic_flux 0"
    elif scheme.startswith("upwind"):
        afmode = " --acoustic_flux 2"
    elif scheme.startswith("mid"):
        afmode = " --acoustic_flux 1"
    if "IPS" in scheme_params:
        afmode += f" --flux_jump_penalty {scheme_params['IPS']:.3e}"
    if "TR" in scheme_params:
        afmode += f" --flux_TR {scheme_params['TR']:.3e}"
    if "XR" in scheme_params:
        afmode += f" --flux_XR {scheme_params['XR']:.3e}"
    abs_flags = " --absorb_top --absorb_bottom" if use_new_abs else ""
    task = SpecfemEMTask(
        sim_dir,
        group="gpu",
        specfem_parfile=parfile,
        cwd=workdir,
        on_pre_run=set_parfile,
        on_completion=compile_series,
        additional_args=f"--lr_periodic{abs_flags} --dumpfolder {outdir} --dump {steps_per_dump}"
        + afmode,
    )
    return task, parfile


def NSTEP(dt):
    return int(1.5 / dt)


def run(sim: Simulation, use_new_abs: bool = True):
    simname = sim.simname()

    with gpulock.GPULock(
        utilization=0.65,
        alloted_time=10 * 60,
        request_timeout=1,
        enter_serial_context_on_fail=True,
    ) as context:
        gpu = context.is_gpu
        task, parfile_name = get_runsim_task(
            simname,
            os.path.join(output_fol, sim.recover_mesh().mesh_database_name()),
            dt=sim.dt(),
            nstep=NSTEP(sim.dt()),
            scheme=sim.scheme,
            scheme_params=sim.scheme_params,
            subdivs=sim.subdivisions,
            use_gpu=gpu,
            use_new_abs=use_new_abs,
        )
        task.job.print_updates = True

        task.on_pre_run()
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

    if (retcode := runjob.complete_job(jid, error_on_still_running=True)) > 0:
        print('Job failed. error logged in "run_err.txt".')
        with open(os.path.join(output_fol, simname, "run_err.txt"), "w") as f:
            f.writelines(lines)
        return retcode

    task.on_completion(0)
    os.remove(parfile_name)
    print("job complete")

    return 0


def clean_simwork():
    # for f in os.listdir(output_fol):
    #     fullpath = os.path.join(output_fol, f)
    #     if os.path.isdir(fullpath):
    #         if f.startswith("cont") or f.startswith("dg"):
    #             shutil.rmtree(fullpath)
    for f in os.listdir(workdir):
        if re.match(r"specfem_config_.+\.yaml", f):
            os.remove(os.path.join(workdir, f))


if __name__ == "__main__":
    args = " ".join(sys.argv)
    m = re.search(r"!\s*TASK\s*\[([^\[\]]+)\]\s*!", args)
    if m:
        sim = Simulation.from_str(m.group(1).strip())
        sys.exit(run(sim))

    m = re.search(r"!\s*CLEAN\s*!", args)
    if m:
        clean_simwork()
        sys.exit(0)

    print(f'Failed to parse arguments "{args}"')
    sys.exit(1)
