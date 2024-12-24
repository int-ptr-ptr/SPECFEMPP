import util.config as config
from util.verify_provenance_existence import verify as prov_exist_verify
from util.dump_reader import read_sfdump
from multiprocessing import Process, Queue
import numpy as np
import subprocess
import os
import time
import shutil
import re

STOPKEY = "INITIATE_STOP"
tol = 1e-3

specfem_exe = config.get("specfem.live.exe")


def run_sims(sim, queue):
    def print(st):
        queue.put(f"[RUN {sim['name']}] " + st)

    print("Running live specfem.")
    os.chdir(sim["workspace"])
    res = subprocess.run(specfem_exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print("xmeshfem2D failed! stdout:\n" + res.stdout.decode("utf-8"))
        print("                   stderr:\n" + res.stderr.decode("utf-8"))
        print("Make sure the parameter file is named PAR_FILE.")
        return


def compare_sims(sim, queue_out, queue_in):
    sname = sim["name"]
    tstart = time.time()

    def log(st):
        m, s = divmod(int(round(time.time() - tstart)), 50)
        queue_out.put(f"[COMPARE {sname} - {m:4d}:{s:02d}] " + st)

    should_continue = True
    dumpnum = -1
    dumpfol = os.path.join(sim["workspace"], "dump/simfield")
    tlast = tstart

    maxerr = dict()
    dump_of_maxerr = dict()
    while should_continue:
        while not queue_in.empty():
            v = queue_in.get()
            if isinstance(v, str) and v == STOPKEY:
                should_continue = False

        # take all of the integer-named files, retrieve the lowest one > dumpnum
        files_to_check = dict()
        for fname in os.listdir(dumpfol):
            match = re.search(r"\d+", fname)
            if match and (index := int(match.group(0))) > dumpnum:
                files_to_check[index] = fname
        if not files_to_check:
            # no files, just wait some time
            time.sleep(5)
            continue

        dumpnum = min(files_to_check.keys())
        fname = files_to_check[dumpnum]

        dumptrue = read_sfdump(os.path.join(sim["prov_dir"], "simfield", fname))
        dumplive = read_sfdump(os.path.join(dumpfol, fname))

        # sqrt(frob^2 / num_entries)  ~ RMS
        should_exit = False
        for fieldname in ["acoustic_field", "elastic_field"]:
            if dumptrue[fieldname].size == 0:
                continue
            err = (
                np.linalg.norm(dumptrue[fieldname] - dumplive[fieldname])
                / dumptrue[fieldname].size ** 0.5
            )
            if fieldname not in maxerr or err > maxerr[fieldname]:
                dump_of_maxerr[fieldname] = dumpnum
                maxerr[fieldname] = err
            if err > tol:
                should_exit = True
        tnow = time.time()
        if tnow - tlast > 2:
            tlast = tnow
            log(
                f"dump #{dumpnum} max errors so far: "
                + " ".join(
                    f" {fieldname}: {maxerr[fieldname]:10.6e} @ dump #{dump_of_maxerr[fieldname]}"
                    for fieldname in maxerr
                )
            )

        if should_exit:
            log(fname + f" - errs exceeded tolerance {tol}. Exiting.")
            log(STOPKEY)
            should_continue = False


def compare_dumps():
    prov_exist_verify()

    procs_run = dict()
    procs_compare = dict()
    compare_queues = dict()
    queue = Queue()
    # start sims
    for sim in config.get("tests"):  # type:ignore
        # first: clean dumps
        if os.path.exists(os.path.join(sim["workspace"], "dump")):
            shutil.rmtree(os.path.join(sim["workspace"], "dump"))

        # start sim
        p = Process(target=run_sims, args=(sim, queue))
        p.start()
        procs_run[sim["name"]] = p  # type: ignore

    time.sleep(5)

    # start comparisons

    for sim in config.get("tests"):  # type:ignore
        # flagging queue, start compare procs
        compare_queues[sim["name"]] = Queue()
        p = Process(target=compare_sims, args=(sim, queue, compare_queues[sim["name"]]))
        p.start()
        procs_compare[sim["name"]] = p  # type: ignore

    while procs_run or procs_compare:
        time.sleep(1)
        while not queue.empty():
            msg = queue.get()
            if re.match(r"\[COMPARE\s*(.*[^\s])\s*-\s*(\d*:\d\d)\] " + STOPKEY, msg):
                print("[!] stop issued. Terminating all.")
                for name, p in list(procs_run.items()):
                    p.terminate()
            else:
                print(msg)
        for name, p in list(procs_run.items()):
            if not p.is_alive():
                print(f"[!] run process for {name} has completed!")
                p.join(0.01)
                del procs_run[name]
                compare_queues[name].put(STOPKEY)

        for name, p in list(procs_compare.items()):
            if not p.is_alive():
                print(f"[!] compare process for {name} has completed!")
                p.join(0.01)
                del procs_compare[name]


if __name__ == "__main__":
    compare_dumps()
