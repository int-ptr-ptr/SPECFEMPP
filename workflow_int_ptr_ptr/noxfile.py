import sys

import nox  # type: ignore

sys.path.insert(0, "./workflow")
import os
import shutil

import util.config as config  # type: ignore

workspace_root = config.get("root_dir")

nox.options.sessions = ["build"]
nox.options.reuse_existing_virtualenvs = True


@nox.session(python=False)
def reconfigure_build(session):
    session.cd(config.get("specfem.root"))
    build_dir = config.get("specfem.live.build")
    if os.path.exists(build_dir):
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


@nox.session
def init_cgcompare_workspace(session):
    session.install("numpy", "matplotlib")
    session.cd(os.path.join(workspace_root, "runnables"))
    cg_workspace = config.get("cg_compare.workspace_folder")
    if os.path.exists(cg_workspace):
        shutil.rmtree(cg_workspace)
    session.run("python", "-m", "cg_compare.load_workspaces")


@nox.session
def test_cgcompare(session):
    session.install("numpy", "matplotlib")
    session.cd(os.path.join(workspace_root, "runnables"))
    session.run("python", "-m", "cg_compare.validate")


@nox.session(python=False)
def create_workspace(session):
    import gen_workspace_dir  # pyright: ignore

    directory = input("Workspace directory name: ")
    gen_workspace_dir.create_workspace(directory, overwrite=None, verbose=True)
