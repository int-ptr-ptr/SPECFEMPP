import json
import sys
import os
import re
from typing import Any, Iterable

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


def get(name: str, search_path: None | Iterable = None) -> Any:
    """Finds the config entry by name.
    Namespaces are separated by dots (.)

    Args:
        name (str): The name of the config entry
        search_path (None | Iterable, optional): A list of entry points. Defaults to None, which
            resolves to [config_root,]. Each entry is checked in order, with the first complete
            resolution returning the value.

    Raises:
        ValueError: If resolution fails

    Returns:
        Any: The entry corresponding to the given name.
    """
    if search_path is None:
        search_path = [config]
    namespaces = name.split(".")
    # resolve the name; ${A.B.C} should be config[A][B][C]
    for search_root in search_path:
        found = search_root
        try:
            for resnav in namespaces:
                if found is None:
                    raise ValueError()
                elif isinstance(found, list):
                    found = found[int(resnav)]  # type: ignore
                else:
                    found = found[resnav]  # type: ignore
            return found  # type: ignore
        except Exception:
            ...
    raise ValueError(f'Unable to resolve name "{name}"')


def convert_entries(entry, parent_stack):
    parent_stack.append(entry)
    was_changed = False
    is_complete = True
    if isinstance(entry, dict):
        for key, value in entry.items():
            if (out := convert_entries(value, parent_stack.copy()))[0] is not None:
                entry[key] = out[0]
                was_changed = True
            is_complete = out[1]
    if isinstance(entry, list):
        for key, value in enumerate(entry):
            if (out := convert_entries(value, parent_stack.copy()))[0] is not None:
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
                found = get(replcode, reversed(parent_stack))
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
        c, is_done = convert_entries(config, [])
        if c is not None:
            config = c
        elif not is_done:
            raise Exception(
                "No change to config resolution, but is not complete. Maybe there is a variable resolution loop somewhere?"
            )
except Exception as e:
    raise ValueError("Cannot process config entries!") from e

if __name__ == "__main__":
    print(get("cg_compare.workspace_files.provenance_seismo"))
