import json
import pathlib
from dataclasses import dataclass
from subprocess import PIPE as subproc_PIPE
from subprocess import run as subproc_run
from typing import Literal
from typing import get_args as literal_to_strs

import workflow.util.config as config
from workflow.laboratory.parfilegen import ReceiverSeries

workdir = pathlib.Path(__file__).parent
output_folname = "OUTPUT_FILES"
output_fol = workdir / output_folname

analysis_outfol = pathlib.Path(config.get("output_dir")) / "snell" / "FF"

receivers = [
    ReceiverSeries(nrec=3, xdeb=0.5, zdeb=0.7, xfin=1, zfin=0.7),
    ReceiverSeries(nrec=3, xdeb=1, zdeb=0.45, xfin=0.5, zfin=0.45),
]

cfl_default = 5e-2
_vp2_vals = [0.25, 0.5, 1.0, 2.0, 4.0]

_scheme_types = Literal["cont", "symm"]
_schemes = literal_to_strs(_scheme_types)


@dataclass(eq=True, frozen=True, slots=True)
class Mesh:
    N: int
    vp2_ind: int

    def vp2(self):
        return _vp2_vals[self.vp2_ind]

    def mesh_database_name(self):
        return f"mesh{self.N}_{self.vp2_ind}"

    def parfile(self):
        return f"Par_File{self.N}_{self.vp2_ind}"

    def topofile(self):
        return f"topo_unit_box{self.N}.dat"


@dataclass
class Simulation:
    N: int
    vp2_ind: int
    subdivisions: tuple[int, int] | None
    scheme: _scheme_types

    def vp2(self):
        return _vp2_vals[self.vp2_ind]

    def dt(self):
        numcells = self.get_horiz_numcells()
        return cfl_default / min(numcells[0], numcells[1] * self.vp2())

    def recover_mesh(self) -> Mesh:
        return Mesh(N=self.N, vp2_ind=self.vp2_ind)

    def simname(self):
        if self.subdivisions is None:
            return f"{self.scheme}_{self.N}_{self.vp2_ind}"
        else:
            return f"{self.scheme}{self.subdivisions[0]}-{self.subdivisions[1]}_{self.N}_{self.vp2_ind}"

    def get_subdivisions(self) -> tuple[int, int]:
        if self.subdivisions is None:
            return (1, 1)
        else:
            return self.subdivisions

    def get_horiz_numcells(self) -> tuple[int, int]:
        if self.subdivisions is None:
            return (self.N, self.N)
        else:
            return (self.N * self.subdivisions[0], self.N * self.subdivisions[1])

    def taskname(self):
        return "sim" + self.simname()

    @staticmethod
    def from_str(s: str):
        for scheme in _schemes:
            if s.startswith(scheme):
                try:
                    sub = s[len(scheme) :].split("_")
                    assert len(sub) == 3
                    subdivisions = None
                    if len(sub[0]) != 0:
                        subdivisions = tuple(int(k) for k in sub[0].split("-"))
                        assert len(subdivisions) == 2

                    return Simulation(
                        N=int(sub[1]),
                        vp2_ind=int(sub[2]),
                        subdivisions=subdivisions,
                        scheme=scheme,
                    )
                except Exception:
                    ...

        raise ValueError(f'Cannot parse "{s}"')


_ALL_SIMS: list[Simulation] | None = None
_ALL_MESHES: list[Mesh] | None = None


def get_all_experiments() -> list[Simulation]:
    global _ALL_SIMS
    if _ALL_SIMS is not None:
        return _ALL_SIMS
    N_vals = [20, 40]
    subdivs = [(1, 1), (1, 2), (1, 3)]
    schemes: list[_scheme_types] = ["symm"]

    sims = []

    for N in N_vals:
        for ivp2 in range(len(_vp2_vals)):
            sims.append(
                Simulation(
                    N=N,
                    vp2_ind=ivp2,
                    subdivisions=None,
                    scheme="cont",
                )
            )
            for subdiv in subdivs:
                for scheme in schemes:
                    sims.append(
                        Simulation(
                            N=N,
                            vp2_ind=ivp2,
                            subdivisions=subdiv,
                            scheme=scheme,
                        )
                    )
    _ALL_SIMS = sims
    return sims


def get_all_meshes() -> list[Mesh]:
    global _ALL_MESHES
    if _ALL_MESHES is not None:
        return _ALL_MESHES
    meshes = set()
    for sim in get_all_experiments():
        meshes.add(sim.recover_mesh())

    _ALL_MESHES = list(meshes)
    return _ALL_MESHES


def clean_area():
    pass


def get_all_tasks():
    tasks = []
    for mesh in get_all_meshes():
        tasks.append(
            {
                "id": mesh.mesh_database_name(),
                "type": "script",
                "command": f"python build_meshes.py !TASK[{mesh.N},{mesh.vp2_ind}]!",
                "files_in": [__file__, "build_meshes.py"],
                "files_out": [f"{output_folname}/{mesh.mesh_database_name()}"],
            }
        )

    tasks.append(
        {
            "id": "mesh_cleaner",
            "type": "script",
            "command": "python build_meshes.py !CLEAN!",
            "deps": [mesh.mesh_database_name() for mesh in get_all_meshes()],
            "skip_if_no_deps": "true",
        }
    )

    for sim in get_all_experiments():
        meshdbasename = sim.recover_mesh().mesh_database_name()
        tasks.append(
            {
                "id": sim.taskname(),
                "type": "script",
                "taskgroups": ["gpu"],
                "command": f"python run_sims.py !TASK[{sim.simname()}]!",
                "files_in": [
                    __file__,
                    "run_sims.py",
                    f"{output_folname}/{meshdbasename}",
                ],
                "files_out": [str(output_fol / sim.simname() / "dumps.dat")],
                "deps": [meshdbasename],
            }
        )

    tasks.append(
        {
            "id": "sim_cleaner",
            "type": "script",
            "command": "python run_sims.py !CLEAN!",
            "deps": [sim.taskname() for sim in get_all_experiments()],
            "skip_if_no_deps": "true",
        }
    )

    # get analyze dependencies
    proc = subproc_run(
        ["python", "analyze.py", "deps"],
        cwd=workdir,
        stdout=subproc_PIPE,
        stderr=subproc_PIPE,
    )
    if proc.returncode != 0:
        raise Exception(
            f"Return code {proc.returncode} obtained when retrieving analysis dependencies."
        ) from Exception(proc.stderr)
    for task in json.loads(proc.stdout)["deps"]:
        tasks.append(
            {
                "id": f"analysis: {task['identifier']}",
                "type": "script",
                "command": f"python analyze.py run {task['identifier']}",
                "files_in": [__file__, *task["in"]],
                "files_out": task["out"],
                "deps": task["deps"],
            }
        )

    return {"tasks": tasks}


if __name__ == "__main__":
    print(json.dumps(get_all_tasks()))
