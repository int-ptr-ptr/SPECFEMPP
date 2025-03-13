import os
import re
import shutil

import util.config as config


def copy_with_changes(fsrc: str, fdest: str, replacements: dict[str, str]):
    pattern = re.compile(
        "|".join([re.escape(k) for k in sorted(replacements, key=len, reverse=True)]),
        flags=re.DOTALL,
    )
    with open(fsrc, "r") as fin:
        with open(fdest, "w") as fout:
            fout.write(pattern.sub(lambda x: replacements[x.group(0)], fin.read()))


def create_workspace(
    directory: str, overwrite: bool | None = None, verbose: bool = True
):
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    directory = os.path.abspath(directory)
    # initialize directory
    if not os.path.exists(directory):
        log(f"Building workspace at {directory}")
        os.makedirs(directory)
    else:
        # clear out dir
        ld = os.listdir(directory)
        if len(ld) > 0:
            if overwrite is None:
                ans = input(
                    f"Workspace folder {directory} already exists. Overwrite? [y/N] "
                )
                overwrite = ans.upper() == "Y"
            if not overwrite:
                log(
                    f"Unable to build workspace at {directory}."
                    " `overwrite` flag is set to false, "
                    "but the workspace is not empty."
                )
                return
            log(f"Preparing workspace {directory}:")
            for entry in ld:
                entryrpath = os.path.join(directory, entry)
                log(f"    - removing {entry}")
                if os.path.isfile(entryrpath):
                    os.remove(entryrpath)
                else:
                    shutil.rmtree(entryrpath)

    log(f"Workspace initialization at {directory}")
    repls = {
        "@master_config@": os.path.join(
            config.get("root_dir"), config.FILENAME_MASTER_CONFIG
        ),
        "@workspace@": directory,
        "@workflow_root@": config.get("root_dir"),
    }
    copy_with_changes(
        config.get("workspace_builder.config_override_ref"),
        os.path.join(directory, config.FILENAME_CONFIG_OVERRIDE),
        repls,
    )
    init_env_scripts_dir = config.get("workspace_builder.init_env_scripts")
    for f in os.listdir(init_env_scripts_dir):
        copy_with_changes(
            os.path.join(init_env_scripts_dir, f),
            os.path.join(directory, f"init_env.{f}"),
            repls,
        )
