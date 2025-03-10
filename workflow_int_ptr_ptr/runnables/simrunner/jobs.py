from util.runjob import SystemCommandJob
import util.config as config
import os

_POSSIBLE_BIN_FOLDERS = ["bin", "."]
_POSSIBLE_MESHFEM_NAMES = ["xmeshfem2D"]
_POSSIBLE_SPECFEM2D_NAMES = ["specfem2d"]
_POSSIBLE_SPECFEMEM_NAMES = ["specfem2d_eventmarcher"]


def _isexe(file: str) -> bool:
    return os.path.isfile(file) and os.access(file, os.X_OK)


def _bin_possibilities(exe_name_possibilities):
    sfroot = config.get("specfem.root")
    for fol in os.listdir(sfroot):
        if fol.startswith("build"):
            for binfol in _POSSIBLE_BIN_FOLDERS:
                for exe in exe_name_possibilities:
                    yield os.path.join(sfroot, fol, binfol, exe)


def guess_meshfem_exe() -> str:
    """Tries to locate a meshfem executable.

    Returns:
        str: The located executable
    """
    exe = config.get("specfem.live.meshfem")
    if _isexe(exe):
        return exe

    # try build directories
    for f in _bin_possibilities(_POSSIBLE_MESHFEM_NAMES):
        if _isexe(f):
            return f

    raise FileNotFoundError(
        "Cannot locate a meshfem executable to run. Please specify one."
    )


def guess_specfem2d_exe() -> str:
    """Tries to locate a specfem2d executable.

    Returns:
        str: The located executable
    """
    exe = config.get("specfem.live.cg_exe")
    if _isexe(exe):
        return exe

    # try build directories
    for f in _bin_possibilities(_POSSIBLE_SPECFEM2D_NAMES):
        if _isexe(f):
            return f

    raise FileNotFoundError(
        "Cannot locate a specfem2d executable to run. Please specify one."
    )


def guess_specfemem_exe() -> str:
    """Tries to locate a specfem2d_eventmarcher executable.

    Returns:
        str: The located executable
    """
    exe = config.get("specfem.live.exe")
    if _isexe(exe):
        return exe

    # try build directories
    for f in _bin_possibilities(_POSSIBLE_SPECFEM2D_NAMES):
        if _isexe(f):
            return f

    raise FileNotFoundError(
        "Cannot locate a specfem2d_eventmarcher executable to run. Please specify one."
    )


class MesherJob(SystemCommandJob):
    def __init__(
        self,
        name: str,
        meshfem_exe: str | None = None,
        meshfem_parfile: str = "Par_File",
        cwd: str | None = None,
        min_update_interval: int = 0,
        linebuf_size: int = 10,
    ):
        if meshfem_exe is None:
            meshfem_exe = guess_meshfem_exe()

        self.exe = meshfem_exe
        self.parfile = meshfem_parfile
        cmd = f"{meshfem_exe} -p {meshfem_parfile}"
        super().__init__(
            name,
            cmd,
            min_update_interval=min_update_interval,
            linebuf_size=linebuf_size,
            print_updates=True,
            cwd=cwd,
        )


class Specfem2DJob(SystemCommandJob):
    def __init__(
        self,
        name: str,
        specfem_exe: str | None = None,
        specfem_parfile: str = "Par_File",
        cwd: str | None = None,
        min_update_interval: int = 0,
        linebuf_size: int = 10,
    ):
        if specfem_exe is None:
            specfem_exe = guess_specfem2d_exe()

        self.exe = specfem_exe
        self.parfile = specfem_parfile
        cmd = f"{specfem_exe} -p {specfem_parfile}"
        super().__init__(
            name,
            cmd,
            min_update_interval=min_update_interval,
            linebuf_size=linebuf_size,
            print_updates=True,
            cwd=cwd,
        )


class SpecfemEMJob(SystemCommandJob):
    def __init__(
        self,
        name: str,
        specfem_exe: str | None = None,
        specfem_parfile: str = "Par_File",
        cwd: str | None = None,
        min_update_interval: int = 0,
        linebuf_size: int = 10,
    ):
        if specfem_exe is None:
            specfem_exe = guess_specfemem_exe()

        self.exe = specfem_exe
        self.parfile = specfem_parfile
        cmd = f"{specfem_exe} -p {specfem_parfile}"
        super().__init__(
            name,
            cmd,
            min_update_interval=min_update_interval,
            linebuf_size=linebuf_size,
            print_updates=True,
            cwd=cwd,
        )
