import os
import sys
from subprocess import PIPE as subproc_PIPE
from subprocess import run as subproc_run

from experiment import Mesh, output_fol, receivers, workdir  # pyright: ignore

from workflow.laboratory.parfilegen import meshfem_config
from workflow.simrunner.jobs import MesherJob


def save_parfile(
    fname: str,
    nx: int,
    ny: int,
    vp2: float,
    vs1: float = 0,
    vs2: float = 0,
    use_stacey_BCs: bool = True,
    staceyLR: bool = True,
    staceyTB: bool = True,
    outfol: str = "OUTPUT_FILES",
    database_out: str = "mesh_out",
    stations_out="STATIONS",
    topo_file: str = "topo_unit_box.dat",
):
    folder = os.path.dirname(fname)
    # this method is populated at the end of the file for readability
    parfile, topo = meshfem_config(
        nx=nx,
        ny=ny,
        vp1=1,
        vs1=vs1,
        vp2=vp2,
        vs2=vs2,
        stacey=use_stacey_BCs,
        absLR=staceyLR,
        absTB=staceyTB,
        outfol=os.path.join(folder, outfol),
        database_out=database_out,
        stations_out=stations_out,
        topo_in_location=topo_file,
        receivers=receivers,
    )
    with open(fname, "w") as f:
        f.write(parfile)

    with open(os.path.join(folder, topo_file), "w") as f:
        f.write(topo)


def run(mesh: Mesh):
    vp2 = mesh.vp2()
    nx = mesh.N
    parfile = mesh.parfile()
    topo_file = mesh.topofile()

    if not os.path.exists(output_fol):
        os.makedirs(output_fol)

    save_parfile(
        os.path.join(workdir, parfile),
        nx,
        nx,
        vp2,
        database_out=mesh.mesh_database_name(),
        stations_out="stations",
        topo_file=topo_file,
    )
    mesher = MesherJob(parfile, meshfem_parfile=parfile, cwd=str(workdir))

    proc = subproc_run([mesher.exe, "-p", mesher.parfile], stdout=subproc_PIPE)
    if proc.returncode != 0 or not os.path.exists(
        output_fol / mesh.mesh_database_name()
    ):
        print(f"ERROR while running mesher for {str(mesh)}. log:")
        print(proc.stdout.decode("utf-8"))
    else:
        print(f"Completed mesher {str(mesh)}.")
    return proc.returncode


def clean_meshwork():
    # clean directory (vtk files)
    for f in os.listdir(output_fol):
        if f.endswith(".vtk"):
            os.remove(output_fol / f)

    # clear mesh parameter files
    for f in os.listdir(workdir):
        fullpath = os.path.join(workdir, f)
        if f.startswith("Par_File"):
            os.remove(fullpath)
        elif f.startswith("topo"):
            os.remove(fullpath)


if __name__ == "__main__":
    import re

    args = " ".join(sys.argv)
    m = re.search(r"!\s*TASK\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*!", args)
    if m:
        mesh = Mesh(N=int(m.group(1)), vp2_ind=int(m.group(2)))
        sys.exit(run(mesh))

    m = re.search(r"!\s*CLEAN\s*!", args)
    if m:
        clean_meshwork()
        sys.exit(0)

    print(f'Failed to parse arguments "{args}"')
    sys.exit(1)
