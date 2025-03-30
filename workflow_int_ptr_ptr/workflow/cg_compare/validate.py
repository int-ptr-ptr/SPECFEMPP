import collections
import os
import re
import shutil
import time
from multiprocessing import Queue
from typing import Callable

import numpy as np

import workflow.util.config as config
import workflow.util.curse_monitor
import workflow.util.dump_reader
import workflow.util.dump_reader_aux
import workflow.util.runjob
import workflow.util.seismo_reader

from . import frame_compare

PRINT_TIME_INTERVAL = 0.1
SWITCH_INTERVAL = 2

SIM_COMPLETE_CODE = "/% SIM_COMPLETE"


class cg_compare_validation:
    def __init__(self, test):
        self.test = test
        self.folder = os.path.join(
            config.get("cg_compare.workspace_folder"), test["name"]
        )
        self.out_folder = os.path.join(
            config.get("cg_compare.output_folder"), test["name"]
        )
        self.job = -1

        self.provenance = workflow.util.dump_reader.dump_series.load_from_file(
            os.path.join(
                self.folder, config.get("cg_compare.workspace_files.provenance_dump")
            )
        )

        self.dumpnum = -1
        self.provind = -1
        self.dumpfol = os.path.dirname(
            os.path.join(
                self.folder, config.get("cg_compare.workspace_files.dump_prefix")
            )
        )
        self.statics = None
        self.errs = dict()
        self.max_err = 0
        self.max_err_dumpnum = 0
        self.lastprint = time.time()

        for of in [config.get("cg_compare.workspace_files.out_seismo")]:
            fol = os.path.join(self.folder, of)
            if not os.path.exists(fol):
                os.makedirs(fol)

        self.dt = float(config.get("cg_compare.dt"))
        self.tmpdir = os.path.join(
            self.folder, config.get("cg_compare.workspace_files.analysis.tmp")
        )
        if not os.path.exists(self.tmpdir):
            os.makedirs(self.tmpdir)
        self.plotnum = 0

        # if the sim thread was completed
        self.simulation_done = False

        # set to true if we timeout on the consume_dumps
        self.was_paused = False
        self.max_num_steps = 1

    def consume_dumps(
        self,
        run_to_completion: bool = True,
        queue_in: "Queue | None" = None,
        queue_out: "Queue | None" = None,
        logfunc: Callable | None = None,
    ):
        t_method_start = time.time()
        if logfunc is None:
            logfunc = print
        while True:
            if queue_in is not None:
                while not queue_in.empty():
                    if queue_in.get() == SIM_COMPLETE_CODE:
                        self.simulation_done = True
            # take all of the integer-named files, retrieve the lowest one > dumpnum
            files_to_check = dict()
            if not os.path.exists(self.dumpfol):
                # the dump folder hasn't even been made yet; hold off.
                if run_to_completion:
                    time.sleep(1)
                    continue
                self.was_paused = False
                break
            for fname in os.listdir(self.dumpfol):
                # did we find the statics file?
                if fname.endswith("statics.dat"):
                    self.statics = workflow.util.dump_reader.read_sfdump(
                        os.path.join(self.dumpfol, fname)
                    )
                    continue
                match = re.search(r"\d+", fname)
                if (
                    match
                    and (index := int(match.group(0))) > self.dumpnum
                    and index in self.provenance.time_indices
                ):
                    files_to_check[index] = (
                        fname,
                        np.where(self.provenance.time_indices == index)[0][0],
                    )
            if (not files_to_check) or (self.statics is None):
                # no more files to check, or we don't have the statics reference yet
                if run_to_completion:
                    if self.simulation_done:
                        break
                    time.sleep(1)
                    continue
                self.was_paused = False
                break
            self.dumpnum = min(files_to_check.keys())
            self.provind = files_to_check[self.dumpnum][1]

            thisdump = workflow.util.dump_reader.read_dump_file(
                os.path.join(self.dumpfol, files_to_check[self.dumpnum][0]),
                statics_data=self.statics,
            )

            prov = self.provenance.get_frame_as_dump_frame(self.provind)

            mapper = workflow.util.dump_reader_aux.field_remapper(
                prov.pts, thisdump.pts
            )
            disp_err = thisdump.displacement - mapper(prov.displacement)
            err = np.linalg.norm(disp_err) / (disp_err.size**0.5)
            self.errs[self.dumpnum] = err

            frame_compare.compare_frames(
                thisdump,
                prov,
                mapper,
                self.dt * self.dumpnum,
                show=False,
                save_filename=os.path.join(self.tmpdir, f"comp{self.plotnum:05d}.png"),
                clear_on_completion=True,
            )

            self.plotnum += 1

            if err > self.max_err:
                self.max_err = err
                self.max_err_dumpnum = self.dumpnum

            t = time.time()
            if t - self.lastprint > PRINT_TIME_INTERVAL:
                self.lastprint = t
                logfunc(
                    f"dump {self.dumpnum}; "
                    f"l2 displacement error maximized at {self.max_err:.6e} "
                    f"(dump {self.max_err_dumpnum})"
                )
                if queue_out is not None:
                    queue_out.put(f"step {self.dumpnum} / {self.max_num_steps}")
                    queue_out.put(f"maxerr {self.max_err:.6e}")
            if (
                not run_to_completion
            ) and time.time() - t_method_start > SWITCH_INTERVAL:
                self.was_paused = True
                break
        if run_to_completion:
            self.finalize(logfunc=logfunc)

    def write_seismos(self):
        workflow.util.seismo_reader.compare_seismos(
            os.path.join(
                self.folder, config.get("cg_compare.workspace_files.out_seismo")
            ),
            os.path.join(
                self.folder, config.get("cg_compare.workspace_files.provenance_seismo")
            ),
            os.path.join(
                self.folder,
                config.get("cg_compare.workspace_files.meshfem_stations_out"),
            ),
            show=False,
            subplot_configuration="individual_rows",
            save_filename=os.path.join(
                self.out_folder, config.get("cg_compare.outputs.seismo")
            ),
        )

    def finalize(self, logfunc=print):
        logfunc(
            "COMPLETE!\n"
            f"   l2 displacement error maximized at {self.max_err:.6e} "
            f"(dump {self.max_err_dumpnum})"
        )
        if os.path.exists(self.out_folder):
            shutil.rmtree(self.out_folder)

        os.makedirs(self.out_folder)

        self.write_seismos()
        logfunc("seismograms written.")
        framerate = config.get(
            "cg_compare.workspace_files.analysis.animation_framerate"
        )
        anim_out_file = config.get("cg_compare.outputs.animation_out")
        os.system(
            f"ffmpeg -pattern_type glob -r {framerate}"
            f' -i "{os.path.join(self.tmpdir, "comp*.png")}" -r {framerate} '
            f"{os.path.join(self.out_folder, anim_out_file)} -y &> /dev/null"
        )
        shutil.rmtree(self.tmpdir)
        logfunc("animation written.")


if __name__ == "__main__":

    def msg_strip_name(msg: str, keep_timestamp: bool = True) -> str:
        if m := re.match(r"\[(.+):\s*(\d+:\d\d)\](\s?)", msg):
            msg = msg[m.end() :]
            if keep_timestamp:
                msg = f"[{m.group(2)}]{m.group(3)}{msg}"
        return msg

    outputs = []
    with workflow.util.curse_monitor.TestMonitor(
        dummy_gui=False, close_with_key=False
    ) as mon:
        tests = config.get("cg_compare.tests")
        compares = dict()
        compare_queues = dict()
        test_disp = dict()
        run_jobs = list()
        compare_jobs = dict()
        test_from_job = dict()
        global_completion_broadcast_task = (
            workflow.util.curse_monitor.TestContainer.Task(
                "All dG-cG comparisons", messages=list()
            )
        )
        num_jobs = 0
        for test in tests:
            num_jobs += 1
            c = cg_compare_validation(test)

            # ====== start the simulation
            if test["class"] == "samemesh" or test["class"] == "subdivmesh":
                args = "%NOC %NOD"
                if test["class"] == "subdivmesh":
                    args += f" -f {config.get('cg_compare.workspace_files.specfem_parfile_subdivs')}"
            else:
                raise ValueError(f"Unknown test class {test['class']}")
            args += " -d " + config.get("cg_compare.dump_test_resolution")
            # args += " --lr_periodic"
            # args += " --kill_boundaries"
            # args += " --absorb_top --absorb_bottom"
            args += " --flux_jump_penalty 20.0"
            # args += " --acoustic_flux 2 --flux_TR 0 --flux_XR 0.01"
            i = workflow.util.runjob.queue_job(
                workflow.util.runjob.SystemCommandJob(
                    name=f"run: {test['name']}",
                    cmd=f"cd {c.folder} && "
                    + config.get("specfem.live.exe")
                    + f" {args}",
                    min_update_interval=0,
                    linebuf_size=100,
                    print_updates=True,
                )
            )
            run_jobs.append(i)
            compares[i] = c
            test_from_job[i] = test
            c.job = i
            c.max_num_steps = config.get("cg_compare.maxsteps")
            # ====== initialize comparison process
            j = workflow.util.runjob.queue_job(
                workflow.util.runjob.CommunicationQueuedFunctionJob(
                    name=f"compare: {test['name']}",
                    func=lambda log, qin, qout: c.consume_dumps(
                        run_to_completion=True,
                        logfunc=log,
                        queue_in=qin,
                        queue_out=qout,
                    ),
                    min_update_interval=0,
                )
            )
            test_disp[i] = workflow.util.curse_monitor.TestContainer(test["name"])
            test_disp[i].tasks = [
                workflow.util.curse_monitor.TestContainer.Task(
                    f"{test['name']} dG simulation",
                    messages=collections.deque(maxlen=100),
                ),
                workflow.util.curse_monitor.TestContainer.Task(
                    f"{test['name']} dG-cG comparison",
                    messages=collections.deque(maxlen=100),
                ),
                global_completion_broadcast_task,
            ]
            mon.add_tab(test_disp[i])
            compare_queues[i] = workflow.util.runjob.get_job_queues(j)
            compare_jobs[i] = j
            mon.manage_inputs()
            mon.redraw_display()

        while compare_jobs:
            mon.manage_inputs()
            mon.redraw_display()
            time.sleep(0.1)

            for i in run_jobs:
                for line in workflow.util.runjob.consume_queue(i):
                    test_disp[i].tasks[0].messages.append(msg_strip_name(line))
                    if m := re.search(r"step\s*(\d+)\s*/\s*(\d+)", line):
                        prog = int(m.group(1)) / int(m.group(2))
                        test_disp[i].tasks[0].progress = prog
                if not workflow.util.runjob.is_job_running(i):
                    test_disp[i].tasks[0].messages.append("[!]: simulation complete.")
                    compare_queues[i]["to_job"].put(SIM_COMPLETE_CODE)
                    run_jobs.remove(i)

            for i, j in list(compare_jobs.items()):
                for line in workflow.util.runjob.consume_queue(j):
                    test_disp[i].tasks[1].messages.append(msg_strip_name(line))

                while not compare_queues[i]["from_job"].empty():
                    line = compare_queues[i]["from_job"].get()
                    if m := re.search(r"step\s*(\d+)\s*/\s*(\d+)", line):
                        prog = int(m.group(1)) / int(m.group(2))
                        test_disp[i].tasks[1].progress = prog
                        test_disp[i].progress = prog
                    if line.startswith("maxerr"):
                        err = line.replace("maxerr", "")
                        test_disp[i].message = f"max error:{err}"

                if not workflow.util.runjob.is_job_running(j):
                    del compare_jobs[i]
                    mon.remove_tab(test_disp[i])
                    global_completion_broadcast_task.messages.append(  # type: ignore
                        f"[{test_from_job[i]['name']}]: {test_disp[i].message}"
                    )
                    global_completion_broadcast_task.progress += 1 / num_jobs
                    outputs.append(
                        f"[{test_from_job[i]['name']}]: {test_disp[i].message}"
                    )
    print("complete. Errors:")
    for m in outputs:
        print(m)
