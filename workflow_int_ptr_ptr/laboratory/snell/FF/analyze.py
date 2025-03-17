import json
import os
import pathlib
from typing import Iterable

import matplotlib.animation as mplanim
import matplotlib.patches as mplpatches
import matplotlib.pyplot as plt
import yaml

from workflow.util import config
from workflow.util.dump_reader import dump_series
from workflow.util.seismo_reader import SeismoDump

workdir = pathlib.Path(__file__).parent
output_fol = workdir / "OUTPUT_FILES"

simout_fol = pathlib.Path(config.get("output_dir")) / "snell" / "FF"

runconfig = {}
with (output_fol / "run_out.json").open("r") as f:
    runconfig = json.load(f)

sourceconfig = {}
with (workdir / "source.yaml").open("r") as f:
    sourceconfig = yaml.load(f, Loader=yaml.Loader)

run_sims_by_vp_ind = dict()

for sim in runconfig["tasks_completed"]:
    vp2ind = sim["vp2_ind"]
    if vp2ind not in run_sims_by_vp_ind:
        run_sims_by_vp_ind[vp2ind] = list()

    run_sims_by_vp_ind[vp2ind].append(sim)

first_source = sourceconfig["sources"][0]
first_source = first_source[list(first_source.keys())[0]]
sourceloc = (first_source["x"], first_source["z"])

del first_source


def make_arrival_include_func(seismo: SeismoDump, vp2: float, tmax: float):
    def include_arrivals(**kwargs):
        axes = kwargs["axes"]
        indexing_func = kwargs["indexing_func"]

        for station in seismo.stations:
            x = station.x
            z = station.z
            T_arrivals = []
            if (z - 0.5) * (sourceloc[1] - 0.5) < 0:
                # opposite side
                if z < 0.5:
                    csrc = vp2
                    csta = 1
                else:
                    csrc = 1
                    csta = vp2

                def compute_time(x):
                    xdiff = x - sourceloc[0]

                    # too lazy to do this analytically, so minimize
                    def dist(transx):
                        return ((transx - x) ** 2 + (z - 0.5) ** 2) ** 0.5 / csta + (
                            (transx - sourceloc[0]) ** 2 + (sourceloc[1] - 0.5) ** 2
                        ) ** 0.5 / csrc

                    if abs(xdiff) < 1e-6:
                        return dist((x + sourceloc[0]) / 2)

                    low = min(x, sourceloc[0])
                    high = max(x, sourceloc[0])
                    while (high - low) / abs(xdiff) > 1e-4:
                        c = (high + low) / 2
                        if (dist(c + 1e-6) - dist(c - 1e-6)) / 2e-6 > 0:
                            # dist is increasing at c. min less
                            high = c
                        else:
                            low = c
                    return dist((high + low) / 2)

                num_cycles = 0
                T_arrivals.append(compute_time(x))
                keep_going = True
                while keep_going:
                    num_cycles += 1
                    t = compute_time(x + num_cycles)

                    keep_going = t < tmax
                    if keep_going:
                        T_arrivals.append(t)

                    t = compute_time(x - num_cycles)
                    if t < tmax:
                        T_arrivals.append(t)
                        keep_going = True
            else:
                # same side
                zdiff2 = (z - sourceloc[1]) ** 2
                num_cycles = 0
                xdiff = x - sourceloc[0]
                c = 1 if z < 0.5 else vp2
                T_arrivals.append((zdiff2 + xdiff**2) ** 0.5 / c)
                keep_going = True
                while keep_going:
                    num_cycles += 1
                    t = (zdiff2 + (xdiff + num_cycles) ** 2) ** 0.5 / c

                    keep_going = t < tmax
                    if keep_going:
                        T_arrivals.append(t)

                    t = (zdiff2 + (xdiff - num_cycles) ** 2) ** 0.5 / c
                    if t < tmax:
                        T_arrivals.append(t)
                        keep_going = True
            for stype in seismo.seismogram_types:
                ax = axes[indexing_func(station, stype)]
                for t in T_arrivals:
                    ax.axvline(x=t, color="lightgray")

    return include_arrivals


def compare_sims(vp2_ind: int, show: bool = True):
    seismo = SeismoDump(str(output_fol / "stations"))
    vp2 = 0
    for sim in run_sims_by_vp_ind[vp2_ind]:
        vp2 = sim["vp2"]
        N_size = int(sim["N_size"])
        color_ind = 0 if "cont" in sim["sim"] else int(sim["sim"][2])
        if color_ind == 0:
            label = f"cG {N_size} cell resolution"
        else:
            label = f"dG {N_size * color_ind}:{N_size} cell resolution (Symmetric flux)"

        simfol = output_fol / sim["sim"]
        seismo.load_from_seismodir(
            str(simfol),
            color="rgbc"[color_ind],
            linestyle=N_size // 10,
            label=label,
        )
    tmax = max(
        max(
            max(arr[-1, 0] for arr in stdata if arr is not None)
            for stdata in seis._seismos.values()
        )
        for seis in seismo.seismos
    )

    seismo.plot_onto(
        show=show,
        legend_kwargs={"loc": "lower left"},
        plt_title=f"Seismogram comparison ($(v_p)_2 = {vp2:.2f}$)",
        save_filename=None
        if show
        else str(simout_fol / f"compare_seismo_{vp2_ind}.png"),
        fig_complete_callback=make_arrival_include_func(seismo, vp2, tmax),
    )


def loadsim(sim, frames: Iterable | None = None):
    simname = sim["sim"]
    dt = sim["dt"]
    vp2 = sim["vp2"]
    simfol = output_fol / simname

    data = dump_series.load_from_file(str(simfol / "dumps.dat"))
    seismo = SeismoDump(str(output_fol / "stations"))
    seismo.load_from_seismodir(str(simfol))
    nseismo = len(seismo.stations)

    t0 = min(
        min(
            min(arr[0, 0] for arr in stdata if arr is not None)
            for stdata in seis._seismos.values()
        )
        for seis in seismo.seismos
    )
    tmax = max(
        max(
            max(arr[-1, 0] for arr in stdata if arr is not None)
            for stdata in seis._seismos.values()
        )
        for seis in seismo.seismos
    )

    # initialize figure and axes
    figwidth = 20
    figheight = 10
    fig = plt.figure(figsize=(figwidth, figheight))
    domain_margin = 0.1
    seismo_domain_margin = 0.045
    ax_domain_size = min(
        figheight * (1 - domain_margin * 2),
        figwidth * 0.5,
    )
    ax_domain = fig.add_axes(
        (
            (0.5 - ax_domain_size / figwidth) / 2,
            (1 - ax_domain_size / figheight) / 2,
            ax_domain_size / figwidth,
            ax_domain_size / figheight,
        )
    )

    seismo_margin = 0.1
    seismo_inter_margin = 0.01
    seismo_ax_height = (
        1 - seismo_margin * 2 - seismo_inter_margin * (nseismo - 1)
    ) / nseismo
    axs_seismos = [
        fig.add_axes(
            (
                0.5 + seismo_domain_margin,
                1
                - (i + 1) * (seismo_ax_height + seismo_inter_margin)
                + seismo_inter_margin
                - seismo_margin,
                0.5 - seismo_margin - seismo_domain_margin,
                seismo_ax_height,
            )
        )
        for i in range(nseismo)
    ]
    for ax in axs_seismos[:-1]:
        ax.xaxis.set_ticks(())

    # line between seismo and domain
    seismo_connections = [
        mplpatches.ConnectionPatch(
            xyA=(-seismo_domain_margin, 0.5),
            xyB=(station.x, station.z),
            coordsA="axes fraction",
            coordsB="data",
            axesA=axs_seismos[i],
            axesB=ax_domain,
        )
        for i, station in enumerate(seismo.stations)
    ]

    # time markers
    ax_vlines = [ax.axvline(x=0, color="lightgray") for ax in axs_seismos]

    # fill out seismos
    seismo.plot_onto(
        axes=axs_seismos,
        indexing_func=lambda st, ty: st.station_index,
        fig_complete_callback=make_arrival_include_func(seismo, vp2, tmax),
    )

    # correct title and make sure seismo-domain lines get drawn
    for i, ax in enumerate(axs_seismos):
        ax.add_artist(seismo_connections[i])
        ax.set_title(ax.get_title(), y=1, pad=-14)

    def plotframe(i: int):
        time_index = list(data.time_indices).index(i)
        frame = data.get_frame_as_dump_frame(time_index)
        t = data.time_indices[time_index] * dt + t0
        frame.plot_field(
            frame.X,
            plt_cell_margin=0,
            mode="contour",
            show=False,
            title=f"t={t:.3f}",
            current_axes=ax_domain,
        )
        for vline in ax_vlines:
            vline.set_xdata([t, t])
        return (*ax_vlines,)

    tis = list(data.time_indices)
    tis.sort()
    if frames is None:
        ani = mplanim.FuncAnimation(
            fig=fig, func=lambda i: plotframe(tis[i]), frames=len(tis), interval=100
        )
        # ani.save(filename=simout_fol / (simname + ".gif"), writer="pillow")
        ani.save(filename=simout_fol / (simname + ".mp4"), writer="ffmpeg")
        plt.close()
    else:
        for frame in frames:
            plotframe(tis[frame])
            fig.show()
            plt.pause(1e-3)
            plt.show()

    # plotframe(tis[-1])

    # seismo.plot_onto(show=True)


if not simout_fol.exists():
    os.makedirs(simout_fol)

# compare_sims(1, show=True)

# for vp2_ind in run_sims_by_vp_ind.keys():
#     compare_sims(vp2_ind,show=False)

for run in runconfig["tasks_completed"]:
    if run["sim"] == "dg2_20_1":
        loadsim(run)
        break
