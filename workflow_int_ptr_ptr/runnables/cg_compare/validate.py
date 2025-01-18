import util.config as config
import util.runjob
import util.dump_reader
import util.dump_reader_aux
import util.seismo_reader
import cg_compare.frame_compare

import time
import os
import re
import numpy as np
import shutil

PRINT_TIME_INTERVAL = 5


class cg_compare_validation:
    def __init__(self, test):
        self.test = test
        self.folder = os.path.join(
            config.get("cg_compare.workspace_folder"), test["name"]
        )
        self.job = -1

        self.provenance = util.dump_reader.dump_series.load_from_file(
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

    def consume_dumps(self):
        while True:
            # take all of the integer-named files, retrieve the lowest one > dumpnum
            files_to_check = dict()
            if not os.path.exists(self.dumpfol):
                # the dump folder hasn't even been made yet; hold off.
                break
            for fname in os.listdir(self.dumpfol):
                # did we find the statics file?
                if fname.endswith("statics.dat"):
                    self.statics = util.dump_reader.read_sfdump(
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
                break
            self.dumpnum = min(files_to_check.keys())
            self.provind = files_to_check[self.dumpnum][1]

            thisdump = util.dump_reader.read_dump_file(
                os.path.join(self.dumpfol, files_to_check[self.dumpnum][0]),
                statics_data=self.statics,
            )

            prov = self.provenance.get_frame_as_dump_frame(self.provind)

            mapper = util.dump_reader_aux.field_remapper(prov.pts, thisdump.pts)
            disp_err = thisdump.displacement - mapper(prov.displacement)
            err = np.linalg.norm(disp_err) / (disp_err.size**0.5)
            self.errs[self.dumpnum] = err

            cg_compare.frame_compare.compare_frames(
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
                print(
                    f"[{self.test['name']}]: dump {self.dumpnum}; "
                    f"l2 displacement error maximized at {self.max_err :.6e} "
                    f"(dump {self.max_err_dumpnum})"
                )

    def write_seismos(self):
        util.seismo_reader.compare_seismos(
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
            tlim=(0, 1),
            save_filename=os.path.join(
                self.folder, config.get("cg_compare.workspace_files.analysis.seismo")
            ),
        )

    def finalize(self):
        print(
            f"[{self.test['name']}]: COMPLETE!\n"
            f"   l2 displacement error maximized at {self.max_err:.6e} "
            f"(dump {self.max_err_dumpnum})"
        )
        self.write_seismos()
        print(f"[{self.test['name']}]: seismograms written.")
        framerate = config.get(
            "cg_compare.workspace_files.analysis.animation_framerate"
        )
        anim_out_file = config.get("cg_compare.workspace_files.analysis.animation_out")
        os.system(
            f"ffmpeg -pattern_type glob -r {framerate}"
            f' -i "{os.path.join(self.tmpdir, "comp*.png")}" -r {framerate} '
            f"{os.path.join(self.folder, anim_out_file)} -y"
        )
        shutil.rmtree(self.tmpdir)
        print(f"[{self.test['name']}]: animation written.")


if __name__ == "__main__":
    tests = config.get("cg_compare.tests")
    compares = list()
    run_jobs = list()
    test_from_job = dict()
    for test in tests:
        c = cg_compare_validation(test)
        compares.append(c)
        if test["class"] == "samemesh":
            args = "%NOC %NOD"
        elif test["class"] == "doublemesh":
            args = "%NOC %D"
        else:
            raise ValueError(f'Unknown test class {test["class"]}')
        i = util.runjob.queue_job(
            util.runjob.RunJob(
                name=f"cg_compare: {test['name']}",
                cmd=f"cd {c.folder} && " + config.get("specfem.live.exe") + f" {args}",
                min_update_interval=4,
                linebuf_size=10,
                print_updates=False,
            )
        )
        run_jobs.append(i)
        test_from_job[i] = test
        c.job = i

    while compares:
        time.sleep(2)
        for c in compares:
            c.consume_dumps()
            if c.job not in run_jobs:
                c.finalize()
                compares.remove(c)

        for i in run_jobs:
            for line in util.runjob.consume_queue(i):
                print(line)
            if not util.runjob.is_job_running(i):
                run_jobs.remove(i)
