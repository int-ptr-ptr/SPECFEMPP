from typing import Callable

from workflow.util.task_manager import Task

from . import jobs


class MesherTask(Task):
    def __init__(
        self,
        title: str,
        group: str | None = None,
        meshfem_exe: str | None = None,
        meshfem_parfile: str | None = None,
        cwd: str | None = None,
        dependencies: list["Task"] | None = None,
        on_completion: Callable[[int], None] | None = None,
        on_pre_run: Callable[[], None] | None = None,
    ):
        self.title = title
        name = f"{title} (mesher)"
        if group is None:
            group = "unnamed mesher"

        # pass kwargs or use defaults for MesherJob
        kwargs = dict()
        if meshfem_parfile is not None:
            kwargs["meshfem_parfile"] = meshfem_parfile
        job = jobs.MesherJob(name, meshfem_exe=meshfem_exe, cwd=cwd, **kwargs)
        self.exe = job.exe
        self.parfile = job.parfile
        self.cwd = cwd
        super().__init__(
            job,
            name=name,
            group=group,
            dependencies=dependencies,
            on_completion=on_completion,
            on_pre_run=on_pre_run,
        )


class Specfem2DTask(Task):
    def __init__(
        self,
        title: str,
        group: str | None = None,
        specfem_exe: str | None = None,
        specfem_parfile: str | None = None,
        cwd: str | None = None,
        dependencies: list["Task"] | None = None,
        on_completion: Callable[[int], None] | None = None,
        on_pre_run: Callable[[], None] | None = None,
    ):
        self.title = title
        name = f"{title} (specfem2d)"
        if group is None:
            group = "unnamed specfem"

        # pass kwargs or use defaults for Specfem2DJob
        kwargs = dict()
        if specfem_parfile is not None:
            kwargs["specfem_parfile"] = specfem_parfile
        job = jobs.Specfem2DJob(name, specfem_exe=specfem_exe, cwd=cwd, **kwargs)
        self.exe = job.exe
        self.parfile = job.parfile
        self.cwd = cwd
        super().__init__(
            job,
            name=name,
            group=group,
            dependencies=dependencies,
            on_completion=on_completion,
            on_pre_run=on_pre_run,
        )


class SpecfemEMTask(Task):
    def __init__(
        self,
        title: str,
        group: str | None = None,
        specfem_exe: str | None = None,
        specfem_parfile: str | None = None,
        cwd: str | None = None,
        dependencies: list["Task"] | None = None,
        on_completion: Callable[[int], None] | None = None,
        on_pre_run: Callable[[], None] | None = None,
        additional_args: str | None = None,
    ):
        self.title = title
        name = f"{title} (specfemEM)"
        if group is None:
            group = "unnamed specfem"

        # pass kwargs or use defaults for SpecfemEMJob
        kwargs = dict()
        if specfem_parfile is not None:
            kwargs["specfem_parfile"] = specfem_parfile
        if additional_args is not None:
            kwargs["additional_args"] = additional_args
        job = jobs.SpecfemEMJob(name, specfem_exe=specfem_exe, cwd=cwd, **kwargs)
        self.exe = job.exe
        self.parfile = job.parfile
        self.cwd = cwd
        super().__init__(
            job,
            name=name,
            group=group,
            dependencies=dependencies,
            on_completion=on_completion,
            on_pre_run=on_pre_run,
        )
