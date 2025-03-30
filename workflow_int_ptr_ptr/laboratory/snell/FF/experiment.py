import json
import math
import pathlib
import re
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

cfl_default = 1e-2
_vp2_vals = [0.25, 0.5, 1.0, 2.0, 4.0]

_scheme_types = Literal["cont", "symm", "upwind", "mid"]
_schemes = literal_to_strs(_scheme_types)


def _str2f(val: str) -> float:
    try:
        return int(val)
    except ValueError:
        ...
    m = re.match(r"(-?\d)(\d*)e(-?\d+)", val)
    if m:
        mant = m.group(2)
        if mant == "":
            mant = "0"
        return (int(m.group(1)) + int(mant) / (10 ** len(mant))) * (
            10 ** int(m.group(3))
        )
    return math.nan


def _f2str(val: float, mantsize: int = 6) -> str:
    if val == 0:
        return "0"
    exponent: int = math.floor(math.log10(abs(val)))
    val *= 10 ** (-exponent)
    assert 1 <= abs(val) < 10

    lead = int(val)
    mant = str(int(abs(val - lead) * (10**mantsize)))
    while len(mant) > 0 and mant[-1] == "0":
        mant = mant[:-1]
    return f"{lead}{mant}e{exponent}"


def _parse_params(scheme: _scheme_types, params: list[str]) -> dict | None:
    if len(params) == 0:
        return None
    if scheme == "cont":
        return None
    elif scheme == "symm":
        return {"IPS": _str2f(params[0])}
    elif scheme == "upwind" or scheme == "mid":
        res = {"IPS": _str2f(params[0])}
        if len(params) > 1:
            res["TR"] = _str2f(params[1])
            if len(params) > 2:
                res["XR"] = _str2f(params[2])
        return res


def _out_params(scheme: _scheme_types, params: dict | None) -> str:
    if params is None or scheme == "cont":
        return ""
    elif scheme == "symm":
        return f"_{_f2str(params['IPS'])}"
    elif scheme == "upwind" or scheme == "mid":
        res = f"_{_f2str(params['IPS'])}"
        if "TR" in params:
            res += f"_{_f2str(params['TR'])}"
            if "XR" in params:
                res += f"_{_f2str(params['XR'])}"
        return res


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
    scheme_params: dict | None = None
    disappearing_information: dict | None = None

    def vp2(self):
        return _vp2_vals[self.vp2_ind]

    def dt(self):
        numcells = self.get_horiz_numcells()
        return cfl_default / min(numcells[0], numcells[1] * self.vp2())

    def recover_mesh(self) -> Mesh:
        return Mesh(N=self.N, vp2_ind=self.vp2_ind)

    def simname(self):
        if self.subdivisions is None:
            return f"{self.scheme}_{self.N}_{self.vp2_ind}" + _out_params(
                self.scheme, self.scheme_params
            )
        else:
            return (
                f"{self.scheme}{self.subdivisions[0]}-{self.subdivisions[1]}_{self.N}_{self.vp2_ind}"
                + _out_params(self.scheme, self.scheme_params)
            )

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
                    assert len(sub) >= 3
                    subdivisions = None
                    if len(sub[0]) != 0:
                        subdivisions = tuple(int(k) for k in sub[0].split("-"))
                        assert len(subdivisions) == 2
                    return Simulation(
                        N=int(sub[1]),
                        vp2_ind=int(sub[2]),
                        subdivisions=subdivisions,
                        scheme=scheme,
                        scheme_params=_parse_params(scheme, sub[3:]),
                    )
                except Exception as e:
                    raise ValueError(f'Cannot parse "{s}"') from e

        raise ValueError(f'Cannot parse "{s}"')

    def get_disappearing_info(self, key, default=None):
        if (
            self.disappearing_information is not None
            and key in self.disappearing_information
        ):
            return self.disappearing_information[key]
        return default

    def get_scheme_param(self, key, default=None):
        if self.scheme_params is not None and key in self.scheme_params:
            return self.scheme_params[key]
        return default


_ALL_SIMS: list[Simulation] | None = None
_ALL_MESHES: list[Mesh] | None = None


def get_all_experiments() -> list[Simulation]:
    global _ALL_SIMS
    if _ALL_SIMS is not None:
        return _ALL_SIMS

    # start with the broad grid
    N_vals = [20, 40]
    subdivs = [(1, 1), (1, 2), (1, 3)]
    maxsubdiv = 3
    schemes: list[tuple[_scheme_types, dict | None]] = [
        ("symm", {"IPS": 20}),
        ("upwind", {"IPS": 20, "TR": 0, "XR": 0}),
        ("mid", {"IPS": 20, "TR": 0, "XR": 0}),
        ("mid", {"IPS": 20, "TR": 0, "XR": -1}),
    ]

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
                            scheme=scheme[0],
                            scheme_params=scheme[1],
                        )
                    )
    # ground truths

    for ivp2 in range(len(_vp2_vals)):
        sims.append(
            Simulation(
                N=max(N_vals) * maxsubdiv,
                vp2_ind=ivp2,
                subdivisions=None,
                scheme="cont",
                disappearing_information={"ground_truth": True},
            )
        )
    # done

    for sim in sims:
        try:
            simdupe = Simulation.from_str(sim.simname())
            assert simdupe.N == sim.N
            assert simdupe.vp2_ind == sim.vp2_ind
            assert simdupe.subdivisions == sim.subdivisions
            assert simdupe.scheme == sim.scheme
            assert (simdupe.scheme_params is None) == (sim.scheme_params is None)
            if sim.scheme_params is not None and simdupe.scheme_params is not None:
                for k, v in sim.scheme_params.items():
                    assert v == simdupe.scheme_params[k]
                for k, v in simdupe.scheme_params.items():
                    assert v == sim.scheme_params[k]
        except AssertionError as e:
            raise AssertionError(
                f"Failed to recover simulation from name {sim.simname()}"
            ) from e
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


def get_all_tasks():
    tasks = []
    prio_overrides = []
    for mesh in get_all_meshes():
        tasks.append(
            {
                "id": mesh.mesh_database_name(),
                "type": "script",
                "command": f"python build_meshes.py !TASK[{mesh.N},{mesh.vp2_ind}]!",
                "files_in": ["build_meshes.py"],
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
    experiment_tasks = dict()
    for sim in get_all_experiments():
        meshdbasename = sim.recover_mesh().mesh_database_name()
        tasks.append(
            {
                "id": sim.taskname(),
                "type": "script",
                "taskgroups": ["gpu"],
                "command": f"python run_sims.py !TASK[{sim.simname()}]!",
                "files_in": [
                    "run_sims.py",
                    f"{output_folname}/{meshdbasename}",
                    "source.yaml",
                ],
                "files_out": [str(output_fol / sim.simname() / "dumps.dat")],
                "deps": [meshdbasename],
                "priority": 10,  # if analysis takes this, task, will bring down
            }
        )
        if sim.get_disappearing_info("ground_truth", False):
            tasks[-1]["priority"] = -1
        experiment_tasks[sim.taskname()] = tasks[-1]

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
                "files_in": [*task["in"]],
                "files_out": task["out"],
                "deps": task["deps"],
            }
        )
        for dep in task["deps"]:
            if dep in experiment_tasks and experiment_tasks[dep]["priority"] > 0:
                experiment_tasks[dep]["priority"] = 0

    return {"tasks": tasks, "priority_overrides": prio_overrides}


if __name__ == "__main__":
    print(json.dumps(get_all_tasks()))
