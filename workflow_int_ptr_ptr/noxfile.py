import nox  # type: ignore
import sys

sys.path.insert(0, "./runnables")
import util.config as config  # type: ignore
import os
import shutil

workspace_root = config.get("root_dir")

nox.options.sessions = ["build"]
nox.options.reuse_existing_virtualenvs = True


@nox.session(python=False)
def reconfigure_build(session):
    session.cd(config.get("specfem.root"))
    build_dir = config.get("specfem.live.build")
    shutil.rmtree(build_dir)
    os.system(
        f"cmake -S . -B {build_dir} {config.get('specfem.live.cmake_build_options')}"
    )


@nox.session(python=False, tags=["test"])
def build(session):
    session.cd(config.get("specfem.root"))
    build_dir = config.get("specfem.live.build")
    if not os.path.exists(build_dir):
        os.system(
            f"cmake -S . -B {build_dir} {config.get('specfem.live.cmake_build_options')}"
        )
    os.system(f"cmake --build {build_dir} --parallel")


@nox.session
def verify_provenance_existence(session):
    session.install("numpy", "matplotlib")

    # check if provenance files
    session.cd(os.path.join(workspace_root, "runnables"))
    session.run("python", "-m", "util.verify_provenance_existence")


@nox.session(tags=["test"])
def test_against_provenance(session):
    session.install("numpy", "matplotlib")
    session.cd(os.path.join(workspace_root, "runnables"))
    session.run("python", "-m", "util.validate_against_dump")
