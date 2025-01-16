import os
import re
import matplotlib.pyplot as plt
import numpy as np


def compare_seismos(
    test_folder,
    ref_folder,
    stations_file,
    tlim=(None, None),
    show: bool = False,
    save_filename: str | None = None,
    verbose=False,
):
    stations = []
    with open(stations_file, "r") as f:
        while line := f.readline():
            m = re.match(
                r"S(\d+)\s+(\w+)\s+((?:\d*\.?\d+)|(?:\d+\.\d*))\s+((?:\d*\.?\d+)|(?:\d+\.\d*))\s+((?:\d*\.?\d+)|(?:\d+\.\d*))\s+((?:\d*\.?\d+)|(?:\d+\.\d*))",
                line,
            )
            if m:
                sname = "S" + m.group(1) + m.group(2)
                sname_sf2d = m.group(2) + ".S" + m.group(1) + "."
                x = float(m.group(3))
                z = float(m.group(4))
                vx = float(m.group(5))
                vz = float(m.group(6))
                if verbose:
                    print(f"Station found: {sname} @({x},{z}) with vel ({vx},{vz})")
                stations.append((sname, sname_sf2d, x, z, vx, vz))
                continue
            # no match
            if verbose:
                print(
                    f"mesh_stations: line does not match given formats:\n    {repr(line)}"
                )

    STATION_IND_POS_OFFSET = 2
    foldernames = [ref_folder, test_folder]
    colors = ["r", "g"]

    num_stations = len(stations)
    row_suffix_search = [
        "BXX.semd",
        "BXZ.semd",
        "BXX.semv",
        "BXZ.semv",
        "BXX.sema",
        "BXZ.sema",
        "PRE.semp",
    ]
    row_suffixes = []
    for suffix in row_suffix_search:
        for i, station in enumerate(stations):
            stationname_sfpp = f"{station[0]}{suffix}"
            stationname_sf2d = f"{station[1]}{suffix}"
            if os.path.exists(f"{test_folder}/{stationname_sfpp}") or os.path.exists(
                f"{test_folder}/{stationname_sfpp}"
            ):
                row_suffixes.append(suffix)
                break

    figlen = 20
    fig, ax = plt.subplots(
        nrows=len(row_suffixes),
        ncols=num_stations,
        figsize=(figlen, figlen / num_stations * len(row_suffixes)),
        sharex=True,
    )

    data_matrix = [
        [
            [[] for _ in range(len(list(zip(foldernames, colors))))]
            for _ in range(len(row_suffixes))
        ]
        for _ in range(num_stations)
    ]
    dataseq_maxlen = 0

    for i, station in enumerate(stations):
        for j, suffix in enumerate(row_suffixes):
            stationname_sfpp = f"{station[0]}{suffix}"
            stationname_sf2d = f"{station[1]}{suffix}"
            a = ax[j, i]
            for k, folseq in enumerate(zip(foldernames, colors)):
                foldername, color = folseq
                data = None
                try:
                    data = np.genfromtxt(
                        f"{foldername}/{stationname_sfpp}",
                        dtype=float,
                    )
                except FileNotFoundError:
                    ...
                if data is None:
                    try:
                        data = np.genfromtxt(
                            f"{foldername}/{stationname_sf2d}",
                            dtype=float,
                        )
                    except FileNotFoundError:
                        ...
                if data is not None:
                    data_matrix[i][j][k] = data  # type: ignore
                    if data.shape[0] > dataseq_maxlen:
                        dataseq_maxlen = data.shape[0]
                    a.plot(data[:, 0], data[:, 1], color, label="")
            a.set_title(
                f"{stationname_sfpp} ({station[STATION_IND_POS_OFFSET + 0]},{station[STATION_IND_POS_OFFSET + 1]})"
            )
            a.set_xlim(tlim)
    # ax[0,-1].legend()
    fig.suptitle("Seismogram Comparison (dG: green, cG: red)")
    if show:
        plt.show()
    if save_filename is not None:
        fol = os.path.dirname(save_filename)
        if not os.path.exists(fol):
            os.makedirs(fol)
        plt.savefig(save_filename)


if __name__ == "__main__":
    import config

    test = config.get("cg_compare.tests.0")
    folder = os.path.join(config.get("cg_compare.workspace_folder"), test["name"])

    compare_seismos(
        os.path.join(folder, config.get("cg_compare.workspace_files.out_seismo")),
        os.path.join(
            folder, config.get("cg_compare.workspace_files.provenance_seismo")
        ),
        os.path.join(
            folder, config.get("cg_compare.workspace_files.meshfem_stations_out")
        ),
        tlim=(0, 1),
    )
