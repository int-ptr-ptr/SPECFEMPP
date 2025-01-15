import util.config as config
import os
import shutil
import subprocess
import time
import numpy as np

import util.runjob
import util.dump_reader


def init_workspace_folder(test):
    folder = os.path.join(config.get("cg_compare.workspace_folder"), test["name"])
    if not os.path.exists(folder):
        os.makedirs(folder)

    if test["class"] == "doublemesh":
        with open(config.get("cg_compare.config_files.meshfem_parfile"), "r") as f:
            st = f.read()
            with open(
                os.path.join(
                    folder, config.get("cg_compare.workspace_files.meshfem_parfile")
                ),
                "w",
            ) as f:
                f.write(
                    st.replace("!!MAT1", test["mat1"]).replace("!!MAT2", test["mat2"])
                )
        with open(
            config.get("cg_compare.config_files.meshfem_parfile_bottom"), "r"
        ) as f:
            st = f.read()
            with open(
                os.path.join(
                    folder,
                    config.get("cg_compare.workspace_files.meshfem_parfile_bottom"),
                ),
                "w",
            ) as f:
                f.write(
                    st.replace("!!MAT1", test["mat1"]).replace("!!MAT2", test["mat2"])
                )
        with open(config.get("cg_compare.config_files.meshfem_parfile_top"), "r") as f:
            st = f.read()
            with open(
                os.path.join(
                    folder, config.get("cg_compare.workspace_files.meshfem_parfile_top")
                ),
                "w",
            ) as f:
                f.write(
                    st.replace("!!MAT1", test["mat1"]).replace("!!MAT2", test["mat2"])
                )

        shutil.copy(
            config.get("cg_compare.config_files.topography"),
            os.path.join(folder, config.get("cg_compare.workspace_files.topography")),
        )
        shutil.copy(
            config.get("cg_compare.config_files.topography_bottom"),
            os.path.join(
                folder, config.get("cg_compare.workspace_files.topography_bottom")
            ),
        )
        shutil.copy(
            config.get("cg_compare.config_files.topography_top"),
            os.path.join(
                folder, config.get("cg_compare.workspace_files.topography_top")
            ),
        )
        shutil.copy(
            config.get("cg_compare.config_files.specfem_parfile"),
            os.path.join(
                folder, config.get("cg_compare.workspace_files.specfem_parfile")
            ),
        )
        shutil.copy(
            config.get("cg_compare.config_files.specfem_parfile_double"),
            os.path.join(
                folder, config.get("cg_compare.workspace_files.specfem_parfile_double")
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
                [config.get("specfem.reference.meshfem"), "-p", parfile],
                stdout=subprocess.PIPE,
            )
            if res.returncode != 0:
                print(
                    f"{test["name"]}: xmeshfem2D failed! Output:\n"
                    + res.stdout.decode("utf-8")
                )
                print("Make sure the parameter file is named PAR_FILE.")
                raise RuntimeError

        run_xmeshfem("PAR_FILE")
        run_xmeshfem("PAR_FILE_DOUBLE1")
        run_xmeshfem("PAR_FILE_DOUBLE2")

        return util.runjob.queue_job(
            util.runjob.RunJob(
                name=f"cg_compare: {test['name']}",
                cmd=f"cd {folder} && " + config.get("specfem.live.exe") + " %NOD",
                min_update_interval=4,
                linebuf_size=10,
                print_updates=True,
            )
        )
    else:
        raise ValueError(f"Unknown test class '{test["class"]}'")


def handle_dump(test):
    folder = os.path.join(config.get("cg_compare.workspace_folder"), test["name"])
    provfol = os.path.join(
        folder, config.get("cg_compare.workspace_files.provenance_fol")
    )

    if os.path.exists(provfol):
        shutil.rmtree(provfol)

    dump_prefix = os.path.join(
        folder, config.get("cg_compare.workspace_files.dump_prefix")
    )
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

    if not os.path.exists(provfol):
        os.makedirs(provfol)

    shutil.move(
        os.path.join(folder, config.get("cg_compare.workspace_files.out_seismo")),
        provfol,
    )

    shutil.rmtree(os.path.dirname(dump_prefix))


if __name__ == "__main__":
    tests = config.get("cg_compare.tests")
    run_jobs = list()
    test_from_job = dict()
    for test in tests:
        i = init_workspace_folder(test)
        run_jobs.append(i)
        test_from_job[i] = test

    while run_jobs:
        time.sleep(2)

        for i in run_jobs:
            for line in util.runjob.consume_queue(i):
                print(line)
            if not util.runjob.is_job_running(i):
                run_jobs.remove(i)
                handle_dump(test_from_job[i])
