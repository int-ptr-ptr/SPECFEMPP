import enum
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, overload

import matplotlib.lines as mplines
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Station:
    group: str
    station_id: int
    x: float
    z: float
    vx: float
    vz: float
    station_index: int

    def __getattr__(self, name: Literal["sname", "sname_sf2d"]):
        if name == "sname":
            return f"{self.group}.S{self.station_id:04d}.S2."
        elif name == "sname_sf2d":
            return f"{self.group}.S{self.station_id:04d}."


class SeismoType(enum.Enum):
    NONE = 0, "_*PLACE_HOLDER*_", ""
    DISP_X = 1, "BXX.semd", "$s_x$"
    DISP_Z = 2, "BXZ.semd", "$s_z$"
    VELO_X = 3, "BXX.semv", "$\\dot s_x$"
    VELO_Z = 4, "BXZ.semv", "$\\dot s_z$"
    ACCL_X = 5, "BXX.sema", "$\\ddot s_x$"
    ACCL_Z = 6, "BXZ.sema", "$\\ddot s_z$"
    PRES = 7, "PRE.semp", "$p$"

    @overload
    def __getattr__(self, name: Literal["index"]) -> int: ...
    @overload
    def __getattr__(self, name: Literal["file_suffix"]) -> str: ...
    @overload
    def __getattr__(self, name: Literal["latex_str"]) -> str: ...
    def __getattr__(
        self, name: Literal["index", "file_suffix", "latex_str"]
    ) -> str | int:
        if name == "index":
            return self.value[0]
        elif name == "file_suffix":
            return self.value[1]
        elif name == "latex_str":
            return self.value[2]
        raise AttributeError

    @staticmethod
    def from_index(ind: int) -> "SeismoType":
        for t in SeismoType:
            if ind == t.index:
                return t
        return SeismoType.NONE

    @staticmethod
    def from_file_suffix(filename: str) -> "SeismoType":
        for t in SeismoType:
            if filename.endswith(t.file_suffix):
                return t
        return SeismoType.NONE


@dataclass
class SeismogramCollection:
    _seismos: dict[int, list[np.ndarray | None]] = field(default_factory=dict)
    _num_stations: int = 0
    plot_color = None
    plot_linestyle: Literal["solid", "dashed", "dashdot", "dotted"] = "solid"
    plot_label: str | None = None

    def seistype(self, types: SeismoType | Iterable[SeismoType]):
        if isinstance(types, SeismoType):
            types = [types]

        return SeismogramCollection(
            _seismos={
                t: v
                for t, v in self._seismos.items()
                if SeismoType.from_index(t) in types
            }
        )

    def stations(self, stations: int | Iterable[int]):
        if isinstance(stations, int):
            stations = [stations]

        return SeismogramCollection(
            _seismos={
                t: [s for i, s in enumerate(v) if i in stations]
                for t, v in self._seismos.items()
            }
        )

    @staticmethod
    def empty(num_stations: int):
        return SeismogramCollection(_num_stations=num_stations)

    def set_seismo(self, station: int, fname: str):
        stype = SeismoType.from_file_suffix(fname)
        sind = stype.index
        if sind not in self._seismos:
            self._seismos[sind] = [None] * self._num_stations
        self._seismos[sind][station] = np.genfromtxt(
            fname,
            dtype=float,
        )


class SeismoDump:
    cwd: str | None
    stations_filename: str
    stations: list[Station]
    station_by_group: dict[str, list[Station]]
    seismos: list[SeismogramCollection]
    seismogram_types: set[SeismoType]

    def __init__(
        self, stations_file: str, cwd: str | None = None, verbose: bool = False
    ):
        self.cwd = cwd

        if cwd is not None:
            stations_file = os.path.join(cwd, stations_file)
        self.stations_filename = stations_file

        stations = []
        with open(stations_file, "r") as f:
            while line := f.readline():
                m = re.match(
                    r"S(\d+)\s+(\w+)\s+((?:\d*\.?\d+)|(?:\d+\.\d*))\s+((?:\d*\.?\d+)|(?:\d+\.\d*))\s+((?:\d*\.?\d+)|(?:\d+\.\d*))\s+((?:\d*\.?\d+)|(?:\d+\.\d*))",
                    line,
                )
                if m:
                    sname = m.group(2) + ".S" + m.group(1) + ".S2."
                    x = float(m.group(3))
                    z = float(m.group(4))
                    vx = float(m.group(5))
                    vz = float(m.group(6))
                    if verbose:
                        print(f"Station found: {sname} @({x},{z}) with vel ({vx},{vz})")
                    stations.append(
                        Station(
                            m.group(2),
                            int(m.group(1)),
                            x,
                            z,
                            vx,
                            vz,
                            station_index=len(stations),
                        )
                    )
                    continue
                # no match
                if verbose:
                    print(
                        f"mesh_stations: line does not match given formats:\n    {repr(line)}"
                    )
        self.stations = stations
        self.verbose = verbose
        self.station_by_group = dict()
        for station in stations:
            if station.group not in self.station_by_group:
                self.station_by_group[station.group] = []
            self.station_by_group[station.group].append(station)
        self.seismos = []
        self.seismogram_types = set()

    def load_from_seismodir(
        self,
        fol: str,
        verbose: bool | None = None,
        color=None,
        linestyle: Literal["solid", "dashed", "dashdot", "dotted"] | int = "solid",
        label: str | None = None,
    ):
        if verbose is None:
            verbose = self.verbose

        seismos = SeismogramCollection.empty(len(self.stations))
        seismos.plot_color = color
        seismos.plot_label = label
        if isinstance(linestyle, int):
            lsint = ((linestyle % 4) + 4) % 4
            if lsint == 0:
                seismos.plot_linestyle = "solid"
            elif lsint == 1:
                seismos.plot_linestyle = "dashed"
            elif lsint == 2:
                seismos.plot_linestyle = "dashdot"
            elif lsint == 3:
                seismos.plot_linestyle = "dotted"
        else:
            seismos.plot_linestyle = linestyle
        self.seismos.append(seismos)

        for f in os.listdir(fol):
            if verbose:
                print(f"File {f}: ", end="")

            for station in self.stations:
                if station.sname in f:
                    # we found the station
                    seismos.set_seismo(station.station_index, os.path.join(fol, f))
                    if verbose:
                        print(
                            f"Placed in station {station.station_index} ({station.sname})"
                        )
                    break

                if station.sname_sf2d in f:
                    # we found the station
                    seismos.set_seismo(station.station_index, os.path.join(fol, f))
                    if verbose:
                        print(
                            f"Placed in station {station.station_index} ({station.sname_sf2d})"
                        )
                    break

        self.seismogram_types.update(
            SeismoType.from_index(k) for k in seismos._seismos.keys()
        )

    def plot_onto(
        self,
        axes=None,
        indexing_func: None | Callable = None,
        subplot_configuration: Literal["matrix", "individual_rows"] = "individual_rows",
        seismo_aspect: float | None = None,
        fig_len: float = 10,
        tlim=(None, None),
        show: bool = False,
        save_filename: str | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        plt_title: str | None = None,
        fig_complete_callback: Callable | None = None,
        axtitles_inside: bool = False,
    ):
        """Plots onto the given axes (creates a new set of subplots if axes is None.)

        Args:
            axes (_type_, optional): Axes array to plot onto. Constructs a new set if None. Defaults to None.
            indexing_func (None | Callable, optional): maps pairs (station, seismotype) -> axes_inds. Defaults to None. This value is overridden if axes is None.
            subplot_configuration (Literal["matrix", "individual_rows"], optional): Sets the style if axes is None. Does nothing if axes is already set. Defaults to "individual_rows".
            seismo_aspect (float | None, optional): Sets the seismogram aspect ratio. Defaults to None.
            seismo_aspect (float, optional): Sets the width of the figure. Defaults to 10.
        """
        num_row_suffixes = len(self.seismogram_types)
        if axes is None:
            if subplot_configuration == "matrix":
                if seismo_aspect is None:
                    seismo_aspect = 2
                fig, axes = plt.subplots(
                    nrows=num_row_suffixes,
                    ncols=len(self.stations),
                    figsize=(
                        fig_len,
                        (fig_len / len(self.stations))
                        / seismo_aspect
                        * num_row_suffixes,
                    ),
                    sharex=True,
                )
                get_ax = lambda row_suff_ind, station_ind: (row_suff_ind, station_ind)  # noqa: E731
            elif subplot_configuration == "individual_rows":
                if seismo_aspect is None:
                    seismo_aspect = 5
                fig, axes = plt.subplots(
                    nrows=num_row_suffixes * len(self.stations),
                    ncols=1,
                    figsize=(
                        fig_len,
                        fig_len / seismo_aspect * num_row_suffixes * len(self.stations),
                    ),
                    sharex=True,
                )
                get_ax = (  # noqa: E731
                    lambda row_suff_ind, station_ind: station_ind * num_row_suffixes
                    + row_suff_ind
                )
            else:
                raise ValueError(
                    f"Unknown subplot configuration {subplot_configuration}"
                )
            stype_map = {s.index: i for i, s in enumerate(self.seismogram_types)}
            indexing_func = lambda station, seismotype: get_ax(  # noqa: E731
                stype_map[seismotype.index], station.station_index
            )

        if indexing_func is None:
            raise TypeError("if axes are given, indexing_func must be as well.")

        for icollection, seismocol in enumerate(self.seismos):
            plot_linestyle = seismocol.plot_linestyle
            plot_color = seismocol.plot_color
            if plot_color is None:
                default_colors = ["r", "g", "b", "c", "m", "y"]
                plot_color = default_colors[icollection % len(default_colors)]

            for iseistype, seislist in seismocol._seismos.items():
                seistype = SeismoType.from_index(iseistype)
                for istation, seis in enumerate(seislist):
                    if seis is not None:
                        station = self.stations[istation]
                        a = axes[indexing_func(station, seistype)]
                        a.plot(
                            seis[:, 0],
                            seis[:, 1],
                            color=plot_color,
                            linestyle=plot_linestyle,
                        )
                        if axtitles_inside:
                            a.set_title(
                                f"   {seistype.latex_str} @ ({station.x},{station.z})",
                                y=1,
                                pad=-14,
                                loc="left",
                            )
                        else:
                            a.set_title(
                                f"{seistype.latex_str} @ ({station.x},{station.z})"
                            )
                        a.set_xlim(tlim)
        if plt_title:
            plt.gcf().suptitle(plt_title)
        if legend_kwargs is not None:
            plt.gcf().legend(handles=self.get_legend_handles(), **legend_kwargs)
        if fig_complete_callback is not None:
            fig_complete_callback(axes=axes, indexing_func=indexing_func)
        if show:
            plt.show()
        if save_filename is not None:
            fol = os.path.dirname(save_filename)
            if not os.path.exists(fol):
                os.makedirs(fol)
            plt.savefig(save_filename)

    def get_legend_handles(self):
        handles = []
        for coll in self.seismos:
            if coll.plot_label is not None:
                handles.append(
                    mplines.Line2D(
                        [0],
                        [0],
                        color=coll.plot_color,
                        linestyle=coll.plot_linestyle,
                        label=coll.plot_label,
                    )
                )
        return handles


def compare_seismos(
    test_folder,
    ref_folder,
    stations_file,
    tlim=(None, None),
    show: bool = False,
    save_filename: str | None = None,
    verbose=False,
    subplot_configuration: Literal["matrix", "individual_rows"] = "matrix",
    seismo_aspect: float | None = None,
    fig_len=10,
):
    sd = SeismoDump(stations_file=stations_file, verbose=verbose)
    stations = sd.stations

    suptitle_ = "Seismogram Comparison (dG: green, cG: red)"
    foldernames = [ref_folder, test_folder]
    colors = ["r", "g"]
    if ref_folder is None:
        suptitle_ = "Siesmogram Outputs"
        foldernames = [test_folder]
        colors = ["g"]

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
            stationname_sfpp = f"{station.sname}{suffix}"
            stationname_sf2d = f"{station.sname_sf2d}{suffix}"
            if os.path.exists(f"{test_folder}/{stationname_sfpp}") or os.path.exists(
                f"{test_folder}/{stationname_sfpp}"
            ):
                row_suffixes.append(suffix)
                break

    if subplot_configuration == "matrix":
        if seismo_aspect is None:
            seismo_aspect = 2
        fig, ax = plt.subplots(
            nrows=len(row_suffixes),
            ncols=num_stations,
            figsize=(
                fig_len,
                (fig_len / num_stations) / seismo_aspect * len(row_suffixes),
            ),
            sharex=True,
        )

        def get_ax(row_suff_ind, station_ind):
            return ax[row_suff_ind, station_ind]
    elif subplot_configuration == "individual_rows":
        if seismo_aspect is None:
            seismo_aspect = 5
        fig, ax = plt.subplots(
            nrows=len(row_suffixes) * num_stations,
            ncols=1,
            figsize=(
                fig_len,
                fig_len / seismo_aspect * len(row_suffixes) * num_stations,
            ),
            sharex=True,
        )

        def get_ax(row_suff_ind, station_ind):
            return ax[station_ind * len(row_suffixes) + row_suff_ind]
    else:
        raise ValueError(f"Unknown subplot configuration {subplot_configuration}")

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
            stationname_sfpp = f"{station.sname}{suffix}"
            stationname_sf2d = f"{station.sname_sf2d}{suffix}"
            a = get_ax(j, i)
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
            a.set_title(f"{stationname_sfpp} ({station.x},{station.z})")
            a.set_xlim(tlim)
    # ax[0,-1].legend()
    fig.suptitle(suptitle_)
    if show:
        plt.show()
    if save_filename is not None:
        fol = os.path.dirname(save_filename)
        if not os.path.exists(fol):
            os.makedirs(fol)
        plt.savefig(save_filename)


if __name__ == "__main__":
    import workflow.util.config as config

    test = config.get("cg_compare.tests.0")
    folder = os.path.join(config.get("cg_compare.workspace_folder"), test["name"])

    D = SeismoDump(
        os.path.join(
            folder, config.get("cg_compare.workspace_files.meshfem_stations_out")
        ),
        verbose=True,
    )
    D.load_from_seismodir(
        os.path.join(folder, config.get("cg_compare.workspace_files.out_seismo"))
    )
    D.load_from_seismodir(
        os.path.join(
            folder, config.get("cg_compare.workspace_files.provenance_seismo")
        ),
    )
    D.plot_onto(show=True)
    # compare_seismos(
    #     os.path.join(folder, config.get("cg_compare.workspace_files.out_seismo")),
    #     os.path.join(
    #         folder, config.get("cg_compare.workspace_files.provenance_seismo")
    #     ),
    #     os.path.join(
    #         folder, config.get("cg_compare.workspace_files.meshfem_stations_out")
    #     ),
    #     # tlim=(0, 1),
    #     subplot_configuration="individual_rows",
    #     verbose=True,
    #     show=True,
    # )
