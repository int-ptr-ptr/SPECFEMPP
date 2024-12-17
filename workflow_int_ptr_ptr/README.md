# Hanson's Workflow

Make sure to remove this before pushing to the main repo.

## Using Nox

[`nox`](https://nox.thea.codes/en/stable/#) is required to use this test suite. To run a session, use the command
`nox -s <session>` or `nox --session <session>` in the `workflow_int_ptr_ptr` directory, or use the flag `-f <path to noxfile.py>`.

The sessions in the `noxfile` are:

- `reconfigure_build` - Deletes the build directory and rebuilds it.
- `build` - If the build directory does not exist, it is created using `cmake`. `cmake --build` is then run on that directory.
- `verify_provenance_existence` - Ensures that the provenance files exist. If the files do not yet exist, the reference commit 4fe2fd3cec2c54800fb42eac51b3d8fb99f353d is downloaded, built, and run, with dumps placed in the `reference` subdirectory.
- `test_against_provenance` - Runs the current build against the reference provenance dump files. If those files do not exist, the same routine as `verify_provenance_existence` is run. Note that this does not build the active build on its own.

Running without specifying a session runs the `build` session.

Additionally, passing `-t test` runs the sessions tagged with `test`, which are `build` and `test_against_provenance`, in that order.

## Configuration

The config is specified in `config.json`, which is read in using `util.config`. Fields can be accessed using `config.get(member)`,
where `member` is separated by dots. In particular, one can reference the build directory using `config.get("specfem.live.build")`.
