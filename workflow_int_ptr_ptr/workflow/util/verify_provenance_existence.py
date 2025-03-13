import collections
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from multiprocessing import Process, Queue

import util.config as config

specfem_exe = config.get("specfem.reference.exe")


def gather_dump(sim, queue):
    def print(st):
        queue.put(f"[{sim['name']}] " + st)

    def longop(cmd, linebuf_size=10, update_interval=2) -> bool:
        """Runs a command through popen (https://docs.python.org/3/library/subprocess.html#subprocess.Popen)
        and returns a success state.
        """
        tstart = time.time()
        tlast = tstart

        def log(st):
            m, s = divmod(int(round(time.time() - tstart)), 50)
            queue.put(f"[{sim['name']}: {m:4d}:{s:02d}] " + st)

        print(f"issuing subprocess command: {cmd}")
        with subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            shell=True,
            bufsize=1,
            universal_newlines=True,
        ) as popen:
            output_queue = collections.deque(maxlen=linebuf_size)
            # while popen.poll():

            # capture outputs
            for line in popen.stdout:  # type: ignore
                # line = line.decode() #type: ignore
                output_queue.append(line)
                t = time.time()
                if t - tlast > update_interval:
                    log(f"Update on subprocess ({cmd}): \n{line}")
                    tlast = t
                time.sleep(0.1)
            retcode = popen.wait()
            time.sleep(update_interval)
        if retcode != 0:
            log(f"subprocess ({cmd}) failed! Output:\n" + "".join(output_queue))
            return False
        return True

    # =============================================
    # ensure workspace is configured.
    if not os.path.exists(sim["workspace"]):
        os.makedirs(sim["workspace"])
    meshfem_parfile = os.path.join(sim["workspace"], "PAR_FILE")
    specfem_parfile = os.path.join(sim["workspace"], "specfem_config.yaml")
    source_parfile = os.path.join(sim["workspace"], "source_config.yaml")
    topo_parfile = os.path.join(sim["workspace"], "topography_file.dat")
    if not os.path.exists(meshfem_parfile):
        shutil.copy(sim["config"]["meshfem_parfile"], meshfem_parfile)
    if not os.path.exists(specfem_parfile):
        shutil.copy(sim["config"]["specfem_parfile"], specfem_parfile)
    if not os.path.exists(source_parfile):
        shutil.copy(sim["config"]["source"], source_parfile)
    if not os.path.exists(topo_parfile):
        shutil.copy(sim["config"]["topography"], topo_parfile)
    sf_outdir = os.path.join(sim["workspace"], "specfem_out/seismo")
    if not os.path.exists(sf_outdir):
        os.makedirs(sf_outdir)

    # =============================================

    print("Running xmeshfem2D.")
    os.chdir(sim["workspace"])
    res = subprocess.run(
        [config.get("specfem.reference.meshfem"), "-p", "PAR_FILE"],
        stdout=subprocess.PIPE,
    )
    if res.returncode != 0:
        print("xmeshfem2D failed! Output:\n" + res.stdout.decode("utf-8"))
        print("Make sure the parameter file is named PAR_FILE.")
        return

    # =============================================
    print("Finished running xmeshfem2D. Starting specfem2D...")
    if not longop(specfem_exe):
        return

    print("Completed specfem run. Migrating dumps.")
    if os.path.exists(sim["prov_dir"]):
        shutil.rmtree(sim["prov_dir"])
    shutil.move(os.path.join(sim["workspace"], "dump"), sim["prov_dir"])


def does_provenance_exist(test):
    provdir = test["prov_dir"]
    return os.path.exists(provdir) and len(os.listdir(provdir)) != 0


def provenance_from_reference(force_rerun: bool = False):
    procs = dict()
    queue = Queue()
    for sim in config.get("tests"):
        if (not force_rerun) and does_provenance_exist(sim):
            continue
        p = Process(target=gather_dump, args=(sim, queue))
        p.start()
        procs[sim["name"]] = p  # type: ignore

    while procs:
        time.sleep(1)
        while not queue.empty():
            print(queue.get())
        for name, p in list(procs.items()):
            if not p.is_alive():
                print(f"[!] process for {name} has completed!")
                p.join(0.01)
                del procs[name]


def retrieve_executable(
    force_rebuild: bool = False,
    force_extract: bool = False,
    force_redownload: bool = False,
):
    """Ensures the existence of the reference executables.

    Args:
        force_rebuild (bool, optional): When True, this and all steps afterwards are done. Defaults to False.
        force_extract (bool, optional): When True, this and all steps afterwards are done. Defaults to False.
        force_redownload (bool, optional): When True, this and all steps afterwards are done. Defaults to False.
    """
    ref_dir = config.get("specfem.reference.dir")
    build_dir = config.get("specfem.reference.build_dir")
    repo_dir = config.get("specfem.reference.local_repo_dir")
    repo_download = config.get("specfem.reference.remote_download_local_filename")

    if force_redownload:
        # forcing redoing means we need to update downstream
        force_extract = True
    if force_extract:
        force_rebuild = True

    if not (
        os.path.exists(build_dir)
        and os.path.exists(config.get("specfem.reference.exe"))
        and os.path.exists(config.get("specfem.reference.meshfem"))
    ):
        # we will need to rebuild the reference executables
        force_rebuild = True

    if force_rebuild and not os.path.exists(repo_dir):
        # we need to rebuild, which requires the repo
        force_extract = True

    if force_extract and not os.path.exists(repo_download):
        force_redownload = True
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    if force_redownload:
        # wget specfem
        print("Downloading specfem...")
        urllib.request.urlretrieve(
            config.get("specfem.reference.repo_remote"),
            filename=config.get("specfem.reference.remote_download_local_filename"),
        )
    else:
        print(
            "Skipping download - either not needed or using existing specfem zip file."
        )

    if force_extract:
        print("Extracting specfem zip...")
        with zipfile.ZipFile(repo_download) as z:
            tmp_dir = os.path.join(ref_dir, "tmp")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)
            z.extractall(tmp_dir)

            shutil.move(os.path.join(tmp_dir, os.listdir(tmp_dir)[0]), repo_dir)
            shutil.rmtree(tmp_dir)
    else:
        print(
            "Skipping extract - either not needed or using already-existing reference repo."
        )

    if force_rebuild:
        print("Building specfem at reference commit.")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        if (
            os.system(
                f"cmake -S '{repo_dir}' -B '{build_dir}' {config.get('specfem.reference.cmake_build_options')}"
            )
            != 0
        ):
            raise RuntimeError("cmake configuration failed!")
        if os.system(f"cmake --build {build_dir} --parallel") != 0:
            raise RuntimeError("cmake build failed!")
    else:
        print("Existing build confirmed.")


def verify():
    needs_to_generate = False
    for sim in config.get("tests"):
        if not does_provenance_exist(sim):
            needs_to_generate = True

    if needs_to_generate:
        print("Provenance generation required.")
        retrieve_executable()

    provenance_from_reference()


if __name__ == "__main__":
    verify()
