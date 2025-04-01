import json
import os
import re
import sys
from typing import Any, Iterable

FILENAME_MASTER_CONFIG = "config.json"
FILENAME_CONFIG_OVERRIDE = "workspace.conf"
USER_FOLDERNAME = "_user"
DEFAULT_USER_CONFIG = "config_override.json"

_found_config_path = None
_found_user_path = None

# loaded configuration
_config = None


def _override_config(
    target: dict[str, list | dict | str],
    override: dict[str, list | dict | str],
    namespace=None,
):
    for k, v in override.items():
        namespace_new = k if namespace is None else f"{namespace}.{k}"
        if k in target:
            tv = target[k]
            if isinstance(v, list) and isinstance(tv, list):
                if len(v) > 0 and (v[0] == "${...}" or v[0] == "$..."):
                    tv.extend(v[1:])
                else:
                    target[k] = v
            elif isinstance(v, dict) and isinstance(tv, dict):
                _override_config(tv, v, namespace=namespace_new)
            elif isinstance(v, str) and isinstance(tv, str):
                target[k] = v
            else:
                raise ValueError(
                    f"Attempting to override value {namespace_new} with a different type."
                )

        else:
            target[k] = v


def _make_userfol():
    global _found_user_path
    if _found_config_path is not None:
        userfol = os.path.join(os.path.dirname(_found_config_path), USER_FOLDERNAME)
        _found_user_path = userfol
        if not os.path.exists(userfol):
            os.mkdir(userfol)
            with open(os.path.join(userfol, ".gitignore"), "w") as f:
                f.write("*\n")


def _read_default_config_files():
    global _found_config_path
    # folders where config file could be
    search_folders = []
    config = None
    user_config = None
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
            found = False
            file = os.path.join(folder, FILENAME_MASTER_CONFIG)
            if os.path.exists(file):
                with open(file, "r") as f:
                    config = json.load(f)
                    config["root_dir"] = os.path.abspath(folder)
                    _found_config_path = file
                    found = True
            if found:
                _make_userfol()
                file = os.path.join(folder, USER_FOLDERNAME, DEFAULT_USER_CONFIG)
                if os.path.exists(file):
                    with open(file, "r") as f:
                        user_config = json.load(f)
        except Exception:
            ...

    if config is None:
        print("Configuration not found!")
        sys.exit(1)

    if user_config is not None:
        _override_config(config, user_config)
    config["user_dir"] = _found_user_path

    # set after
    return config


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
        search_path = [_config]
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


def _expand_references(entries):
    config = entries
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
        raise RuntimeError("Cannot process config entries!") from e

    return config


def set_user_config(name: str, entry):
    conf = dict()
    try:
        with open(os.path.join(_found_user_path, DEFAULT_USER_CONFIG), "r") as f:  # type: ignore
            conf = json.load(f)
    except IOError:
        ...
    target_type = type(entry)
    # store

    namespaces = name.split(".")
    found = conf
    found_parent = conf
    try:
        for i, resnav in enumerate(namespaces):
            if isinstance(found, list):
                found_parent = found
                rn = int(resnav)

                # this may raise an index error. Let it happen
                found = found[rn]  # type: ignore
            else:
                found_parent = found
                if resnav not in found:
                    # found = entry. They need to be the same type
                    found[resnav] = dict() if i + 1 < len(namespaces) else target_type()
                found = found[resnav]  # type: ignore

        _override_config(found_parent, {namespaces[-1]: entry})  # type: ignore
    except Exception as e:
        raise ValueError(f'Unable to resolve name "{name}"') from e

    with open(os.path.join(_found_user_path, DEFAULT_USER_CONFIG), "w") as f:  # type: ignore
        json.dump(conf, f)


_default_unexpanded_conf = _read_default_config_files()
_config = _expand_references(_default_unexpanded_conf)

if __name__ == "__main__":
    print(get("cg_compare.workspace_files.provenance_seismo"))
    # set_user_config("simrunneTr.build_search_ignores",["build_debug_mpi"])
