import nox  # type: ignore


nox.options.sessions = ["test_against_provenance"]


@nox.session
def verify_provenance_existence(session):
    session.install("numpy", "matplotlib")

    # check if provenance files
    session.cd("runnables")
    session.run("python", "-m", "util.verify_provenance_existence")


@nox.session
def test_against_provenance(session):
    session.install("numpy", "matplotlib")
    session.cd("runnables")
    session.run("python", "-m", "util.validate_against_dump")
