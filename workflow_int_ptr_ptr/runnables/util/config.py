import json
import sys
import os
import re
from typing import Any

# loaded configuration
config = None


# folders where config file could be
search_folders = []
if "__file__" in globals():
    utildir = os.path.dirname(os.path.realpath(__file__))

    try:
        search_folders.append(os.path.dirname(os.path.dirname(utildir)))
        search_folders.append(os.path.dirname(search_folders[-1]))
        search_folders.append(os.path.dirname(utildir))
        search_folders.append(utildir)
    except Exception:
        ...
search_folders.append(os.getcwd())
try:
    for _ in range(3):
        search_folders.append(os.path.dirname(search_folders[-1]))
except Exception:
    ...

for folder in search_folders:
    try:
        with open(os.path.join(folder, "config.json"), "r") as f:
            config = json.load(f)
            config["root_dir"] = os.path.abspath(folder)
            break
    except Exception:
        ...

if config is None:
    print("Configuration not found!")
    sys.exit(1)


def get(name: str) -> Any:
    # resolve the name; ${A.B.C} should be config[A][B][C]
    found = config
    for resnav in name.split("."):
        if found is None:
            raise ValueError(
                f"config path {name} reached a None entry. Cannot resolve at {resnav}."
            )
        found = found[resnav]  # type: ignore
    return found  # type: ignore


def convert_entries(entry):
    was_changed = False
    is_complete = True
    if isinstance(entry, dict):
        for key, value in entry.items():
            if (out := convert_entries(value))[0] is not None:
                entry[key] = out[0]
                was_changed = True
            is_complete = out[1]
    if isinstance(entry, list):
        for key, value in enumerate(entry):
            if (out := convert_entries(value))[0] is not None:
                entry[key] = out[0]
                was_changed = True
            is_complete = out[1]

    if isinstance(entry, str):
        matches = list(re.finditer(r"\$(?:\{([\.\w]+)\}|(\w+))", entry))
        for m in reversed(matches):
            # first or second group (first is if {} was used)
            replcode = m.group(1)
            if replcode is None:
                replcode = m.group(2)
            try:
                # resolve the name; ${A.B.C} should be config[A][B][C]
                found = get(replcode)
                assert isinstance(found, str)
                if not re.search(r"\$(?:\{([\.\w]+)\}|(\w+))", found):
                    # variable fully resolved, so we can use it
                    entry = entry[: m.start()] + found + entry[m.end() :]
                    was_changed = True
                else:
                    is_complete = False
            except Exception as e:
                raise ValueError(f"Cannot resolve variable {replcode}.") from e

    return entry if was_changed else None, is_complete


try:
    # keep resolving until everything is complete
    is_done = False
    while not is_done:
        c, is_done = convert_entries(config)
        if c is not None:
            config = c
        elif not is_done:
            raise Exception(
                "No change to config resolution, but is not complete. Maybe there is a variable resolution loop somewhere?"
            )
except Exception as e:
    raise ValueError("Cannot process config entries!") from e
