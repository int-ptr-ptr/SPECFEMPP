from typing import Callable
import util.config as config
import os
import shutil
import subprocess
import time
import numpy as np
import re

import util.runjob
import util.dump_reader


def init_workspace_folder(test):
    folder = os.path.join(config.get("cg_compare.workspace_folder"), test["name"])
    if not os.path.exists(folder):
        os.makedirs(folder)

    if test["class"] == "samemesh" or test["class"] == "subdivmesh":
        is_subdivmesh = test["class"] == "subdivmesh"

        def convert_meshfem_macros(st):
            nx = int(test["meshfem_nx"])
            ny = int(test["meshfem_ny"])
            return (
                st.replace("!!MAT1", test["mat1"])
                .replace("!!MAT2", test["mat2"])
                .replace("!!HNYP1", str((ny // 2) + 1))
                .replace("!!HNY", str(ny // 2))
                .replace("!!NX", str(nx))
                .replace("!!NY", str(ny))
            )

        with open(config.get("cg_compare.config_files.meshfem_parfile"), "r") as f:
            st = f.read()
            with open(
                os.path.join(
                    folder, config.get("cg_compare.workspace_files.meshfem_parfile")
                ),
                "w",
            ) as f:
                f.write(convert_meshfem_macros(st))

        with open(config.get("cg_compare.config_files.topography"), "r") as f:
            st = f.read()
            with open(
                os.path.join(
                    folder, config.get("cg_compare.workspace_files.topography")
                ),
                "w",
            ) as f:
                f.write(convert_meshfem_macros(st))

        shutil.copy(
            config.get("cg_compare.config_files.specfem_parfile"),
            os.path.join(
                folder, config.get("cg_compare.workspace_files.specfem_parfile")
            ),
        )
        if is_subdivmesh:
            shutil.copy(
                config.get("cg_compare.config_files.specfem_parfile_subdivs"),
                os.path.join(
                    folder,
                    config.get("cg_compare.workspace_files.specfem_parfile_subdivs"),
                ),
            )

        shutil.copy(
            config.get("cg_compare.config_files.source"),
            os.path.join(folder, config.get("cg_compare.workspace_files.source")),
        )
        for of in [config.get("cg_compare.workspace_files.out_seismo")]:
            fol = os.path.join(folder, of)
            if not os.path.exists(fol):
                os.makedirs(fol)

        os.chdir(folder)

        def run_xmeshfem(parfile):
            res = subprocess.run(
                [config.get("specfem.live.meshfem"), "-p", parfile],
                stdout=subprocess.PIPE,
            )
            if res.returncode != 0:
                print(
                    f"{test['name']}: xmeshfem2D failed! Output:\n"
                    + res.stdout.decode("utf-8")
                )
                print("Make sure the parameter file is named PAR_FILE.")
                raise RuntimeError

        run_xmeshfem("PAR_FILE")

        return util.runjob.queue_job(
            util.runjob.SystemCommandJob(
                name=f"cg_compare: {test['name']}",
                cmd=f"cd {folder} && "
                + config.get("specfem.live.exe")
                + " %C %NOD -d "
                + config.get("cg_compare.dump_test_resolution"),
                min_update_interval=0,
                linebuf_size=30,
                print_updates=True,
            )
        )
    else:
        raise ValueError(f"Unknown test class '{test['class']}'")


def handle_dump(test, log: Callable[[str], None]):
    folder = os.path.join(config.get("cg_compare.workspace_folder"), test["name"])
    provfol = os.path.join(
        folder, config.get("cg_compare.workspace_files.provenance_fol")
    )

    if os.path.exists(provfol):
        log("Found already existing provenance folder. Deleting.")
        shutil.rmtree(provfol)
        log("Deleted.")

    dump_prefix = os.path.join(
        folder, config.get("cg_compare.workspace_files.dump_prefix")
    )
    log("Compiling dump files into one series file.")
    series = util.dump_reader.load_series(dump_prefix)

    provfile = os.path.join(
        folder, config.get("cg_compare.workspace_files.provenance_dump")
    )
    if not os.path.exists(os.path.dirname(provfile)):
        os.makedirs(os.path.dirname(provfile))

    # make sure the interval is at least dump_test_resolution
    resolution = int(config.get("cg_compare.dump_test_resolution"))
    inds = series.time_indices
    imax = inds.max()
    sub_inds = [inds.argmin()]
    thresh = inds[sub_inds[-1]] + resolution - 1
    while any(inds > thresh):
        sub_inds.append(np.where(inds > thresh, inds, imax + 1).argmin())
        thresh = inds[sub_inds[-1]] + resolution - 1

    series.get_subseries(sub_inds).save_to_file(provfile)
    log("Done")

    if not os.path.exists(provfol):
        os.makedirs(provfol)

    log("Copying other outputs (seismos) into provenance folder.")
    shutil.move(
        os.path.join(folder, config.get("cg_compare.workspace_files.out_seismo")),
        provfol,
    )
    log("Done. Cleaning dump folder")

    shutil.rmtree(os.path.dirname(dump_prefix))
    log("Complete.")


if __name__ == "__main__":
    import util.curse_monitor as disp

    def msg_strip_name(msg: str, keep_timestamp: bool = True) -> str:
        if m := re.match(r"\[(.+):\s*(\d+:\d\d)\](\s?)", msg):
            msg = msg[m.end() :]
            if keep_timestamp:
                msg = f"[{m.group(2)}]{m.group(3)}{msg}"
        return msg

    with disp.TestMonitor(dummy_gui=False, close_with_key=False) as mon:
        disp_tests = dict()
        tests = config.get("cg_compare.tests")
        run_jobs = list()
        clean_jobs = dict()
        test_from_job = dict()
        for test in tests:
            i = init_workspace_folder(test)
            run_jobs.append(i)
            test_from_job[i] = test

            disp_tests[i] = disp.TestContainer(test["name"])
            mon.add_tab(disp_tests[i])
            disp_tests[i].tasks.append(
                disp.TestContainer.Task("specfem provenance (cG) run")
            )

        while run_jobs or clean_jobs:
            mon.manage_inputs()
            mon.redraw_display()
            time.sleep(0.1)

            for i in run_jobs:
                for line in util.runjob.consume_queue(i):
                    disp_tests[i].tasks[0].messages.append(msg_strip_name(line))
                    if mon.dummy_gui:
                        print(line)
                    if m := re.search(r"(\d+)\s*/\s*(\d+)", line):
                        prog = int(m.group(1)) / int(m.group(2))
                        disp_tests[i].tasks[0].progress = prog
                        disp_tests[i].progress = prog
                if not util.runjob.is_job_running(i):
                    run_jobs.remove(i)
                    clean_jobs[i] = util.runjob.queue_job(
                        util.runjob.FunctionJob(
                            name=f"clean workspace: {test_from_job[i]['name']}",
                            func=lambda logfunc: handle_dump(test_from_job[i], logfunc),
                        )
                    )

            for i, j in list(clean_jobs.items()):
                for line in util.runjob.consume_queue(j):
                    disp_tests[i].tasks[0].messages.append(line)
                if not util.runjob.is_job_running(j):
                    del clean_jobs[i]
                    mon.remove_tab(disp_tests[i])
