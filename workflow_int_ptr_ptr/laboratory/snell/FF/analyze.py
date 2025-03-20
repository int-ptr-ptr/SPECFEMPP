import argparse
import json
import os
import pathlib
import sys
from math import nan
from typing import Any, Callable, Iterable, Literal, overload

import matplotlib.animation as mplanim
import matplotlib.lines as mplines
import matplotlib.patches as mplpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

from workflow.laboratory.config_reader import should_skip_by_file_deps
from workflow.util import config
from workflow.util.dump_reader import dump_series
from workflow.util.seismo_reader import SeismoDump

workdir = pathlib.Path(__file__).parent
output_fol = workdir / "OUTPUT_FILES"

analysis_outfol = pathlib.Path(config.get("output_dir")) / "snell" / "FF"

runconfig = {}
with (output_fol / "run_out.json").open("r") as f:
    runconfig = json.load(f)

sourceconfig = {}
with (workdir / "source.yaml").open("r") as f:
    sourceconfig = yaml.load(f, Loader=yaml.Loader)

run_sims_by_vp_ind = dict()
run_sims_by_name = dict()

for sim in runconfig["tasks_completed"]:
    vp2ind = sim["vp2_ind"]
    if vp2ind not in run_sims_by_vp_ind:
        run_sims_by_vp_ind[vp2ind] = list()

    run_sims_by_vp_ind[vp2ind].append(sim)

    run_sims_by_name[sim["sim"]] = sim


first_source = sourceconfig["sources"][0]
first_source = first_source[list(first_source.keys())[0]]
sourceloc = (first_source["x"], first_source["z"])

del first_source


@overload
def compute_arrivals(
    seismo: SeismoDump, vp2: float, tmax: float, store_crossovers: Literal[True]
) -> tuple[dict[int, list[float]], dict[int, list[tuple[float, float]]]]: ...
@overload
def compute_arrivals(
    seismo: SeismoDump,
    vp2: float,
    tmax: float,
    store_crossovers: Literal[False] = False,
) -> dict[int, list[float]]: ...
def compute_arrivals(
    seismo: SeismoDump,
    vp2: float,
    tmax: float,
    store_crossovers: Literal[True, False] = False,
):
    arrivals: dict[int, list[float]] = {}
    crossovers: dict[int, list[tuple[float, float]]] = {}
    for station in seismo.stations:
        x = station.x
        z = station.z
        T_arrivals = []
        x_crossovers = []
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
                    c = (x + sourceloc[0]) / 2
                    return dist(c), c

                low = min(x, sourceloc[0])
                high = max(x, sourceloc[0])
                while (high - low) / abs(xdiff) > 1e-4:
                    c = (high + low) / 2
                    if (dist(c + 1e-6) - dist(c - 1e-6)) / 2e-6 > 0:
                        # dist is increasing at c. min less
                        high = c
                    else:
                        low = c
                c = (high + low) / 2
                return dist(c), c

            num_cycles = 0
            t, cross = compute_time(x)
            if t < tmax:
                T_arrivals.append(t)
                x_crossovers.append((x, cross))
            keep_going = True
            while keep_going:
                num_cycles += 1
                t, cross = compute_time(x + num_cycles)
                keep_going = t < tmax
                if keep_going:
                    T_arrivals.append(t)
                    x_crossovers.append((x + num_cycles, cross))

                t, cross = compute_time(x - num_cycles)
                if t < tmax:
                    T_arrivals.append(t)
                    x_crossovers.append((x - num_cycles, cross))
                    keep_going = True
        else:
            # same side
            zdiff2 = (z - sourceloc[1]) ** 2
            num_cycles = 0
            xdiff = x - sourceloc[0]
            c = 1 if z < 0.5 else vp2

            t = (zdiff2 + xdiff**2) ** 0.5 / c
            if t < tmax:
                T_arrivals.append(t)
                x_crossovers.append((x, nan))
            keep_going = True
            while keep_going:
                num_cycles += 1
                t = (zdiff2 + (xdiff + num_cycles) ** 2) ** 0.5 / c

                keep_going = t < tmax
                if keep_going:
                    T_arrivals.append(t)
                    x_crossovers.append((x + num_cycles, nan))

                t = (zdiff2 + (xdiff - num_cycles) ** 2) ** 0.5 / c
                if t < tmax:
                    T_arrivals.append(t)
                    x_crossovers.append((x - num_cycles, nan))
                    keep_going = True

        arrivals[station.station_index] = T_arrivals
        crossovers[station.station_index] = x_crossovers
    if store_crossovers:
        return arrivals, crossovers
    return arrivals


def make_arrival_include_func(seismo: SeismoDump, vp2: float, tmax: float):
    arrivals = compute_arrivals(seismo, vp2, tmax)

    def include_arrivals(**kwargs):
        axes = kwargs["axes"]
        indexing_func = kwargs["indexing_func"]

        for station in seismo.stations:
            T_arrivals = arrivals[station.station_index]
            for stype in seismo.seismogram_types:
                ax = axes[indexing_func(station, stype)]
                for t in T_arrivals:
                    ax.axvline(x=t, color="lightgray")
                    ax.annotate(
                        f"t={t:.3f}",
                        xy=(t, 1),
                        xycoords=("data", "axes fraction"),
                        va="bottom",
                        ha="center",
                    )
            axes[0].legend(
                handles=[
                    mplines.Line2D(
                        [0], [0], color="lightgray", label="Expected arrivals"
                    )
                ],
                loc="lower right",
                bbox_to_anchor=(1, 1),
            )

    return include_arrivals


def compare_sim_filename(vp2_ind: int):
    return str(analysis_outfol / f"compare_seismo_{vp2_ind}.png")


def compare_sims(
    vp2_ind: int, show: bool = True, skip_and_get_filedeps_only: bool = False
) -> Any:
    seismo = SeismoDump(str(output_fol / "stations"))
    fdeps = [__file__, str(output_fol / "stations")]
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
        if skip_and_get_filedeps_only:
            fdeps.append(str(simfol))
        else:
            seismo.load_from_seismodir(
                str(simfol),
                color="rgbc"[color_ind],
                linestyle=N_size // 10,
                label=label,
            )
    outfile = compare_sim_filename(vp2_ind)
    if skip_and_get_filedeps_only:
        return fdeps, [outfile]

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
        plt_title=f"Acoustic-Acoustic Seismogram comparison ($(v_p)_2 = {vp2:.2f}$)",
        save_filename=None if show else outfile,
        fig_complete_callback=make_arrival_include_func(seismo, vp2, tmax),
        axtitles_inside=True,
    )


def loadsim(
    sim, frames: Iterable | None = None, skip_and_get_filedeps_only: bool = False
) -> Any:
    simname = sim["sim"]
    dt = sim["dt"]
    vp2 = sim["vp2"]
    simfol = output_fol / simname

    outfile_mp4 = analysis_outfol / (simname + ".mp4")
    if skip_and_get_filedeps_only:
        return [
            __file__,
            str(simfol / "dumps.dat"),
            str(output_fol / "stations"),
            str(simfol),
        ], [outfile_mp4]

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
    seismo_inter_margin = 0.02
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
    ax_vlines = [ax.axvline(x=0, color="gray") for ax in axs_seismos]

    # fill out seismos
    seismo.plot_onto(
        axes=axs_seismos,
        indexing_func=lambda st, ty: st.station_index,
        fig_complete_callback=make_arrival_include_func(seismo, vp2, tmax),
        axtitles_inside=True,
    )

    # correct title and make sure seismo-domain lines get drawn
    for i, ax in enumerate(axs_seismos):
        ax.add_artist(seismo_connections[i])
        ax.set_title(ax.get_title(), y=1, pad=-14)

    # get arrivals for drawing on field
    arrivals, crossovers = compute_arrivals(seismo, vp2, tmax, store_crossovers=True)
    t_wavestart = 0

    def plotframe(i: int):
        time_index = list(data.time_indices).index(i)
        frame = data.get_frame_as_dump_frame(time_index)
        t = data.time_indices[time_index] * dt + t0
        ax_domain.cla()
        frame.plot_field(
            frame.X,
            plt_cell_margin=0,
            mode="contour",
            show=False,
            title=f"Pressure wavefield $p$ (t={t:.3f})",
            current_axes=ax_domain,
        )
        # arrivals tracing

        def trace_arrivals(do_ratios: bool, lineargs):
            for station in seismo.stations:
                sz = station.z
                for t_arrive, co_ in zip(
                    arrivals[station.station_index], crossovers[station.station_index]
                ):
                    sx, co = co_
                    if (t_arrive < t and do_ratios) or t_arrive - t_wavestart <= 1e-6:
                        continue
                    if co is nan:
                        # no crossover: straight line
                        xdata = np.array([sourceloc[0], sx])
                        ydata = np.array([sourceloc[1], sz])
                        if do_ratios:
                            ratio = max(0, (t - t_wavestart) / (t_arrive - t_wavestart))
                            xdata[0] = (1 - ratio) * xdata[0] + ratio * xdata[1]
                            ydata[0] = (1 - ratio) * ydata[0] + ratio * ydata[1]
                        xoff = 0
                        # 0 < sourceloc < 1, but sx may lie outside.
                        while True:
                            ax_domain.add_line(
                                mplines.Line2D(xdata + xoff, ydata, **lineargs)
                            )
                            if 0 <= sx + xoff and sx + xoff <= 1:
                                break

                            if sx + xoff > 1:
                                xoff -= 1
                            else:
                                xoff += 1
                    else:
                        # crossovers
                        xdata = np.array([sourceloc[0], co, sx])
                        ydata = np.array([sourceloc[1], 0.5, sz])

                        if do_ratios:
                            if sz < 0.5:
                                csrc = vp2
                            else:
                                csrc = 1
                            crossover_time = (
                                (co - sourceloc[0]) ** 2 + (0.5 - sourceloc[1]) ** 2
                            ) ** 0.5 / csrc
                            if (t - t_wavestart) > crossover_time:
                                # we crossed over. kill first data point and trace second
                                if (t_arrive - t_wavestart - crossover_time) < 1e-6:
                                    # hits target immediately after crossover. don't draw
                                    continue
                                ratio = (t - t_wavestart - crossover_time) / (
                                    t_arrive - t_wavestart - crossover_time
                                )
                                xdata = np.array([(1 - ratio) * co + ratio * sx, sx])
                                ydata = np.array([(1 - ratio) * 0.5 + ratio * sz, sz])
                            else:
                                # no crossover yet, trace first data point
                                ratio = max(0, (t - t_wavestart) / crossover_time)
                                xdata[0] = (1 - ratio) * xdata[0] + ratio * xdata[1]
                                ydata[0] = (1 - ratio) * ydata[0] + ratio * ydata[1]

                        xoff = 0
                        # 0 < sourceloc < 1, but sx may lie outside.
                        while True:
                            ax_domain.add_line(
                                mplines.Line2D(xdata + xoff, ydata, **lineargs)
                            )
                            if 0 <= sx + xoff and sx + xoff <= 1:
                                break

                            if sx + xoff > 1:
                                xoff -= 1
                            else:
                                xoff += 1

        trace_arrivals(False, {"color": (1, 0, 0, 0.5), "linestyle": ":"})
        trace_arrivals(True, {"color": (1, 0, 0, 0.8), "linestyle": "-"})

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
        ani.save(filename=outfile_mp4, writer="ffmpeg")
        plt.close()
    else:
        for frame in frames:
            plotframe(tis[frame])
            fig.show()
            plt.pause(1e-3)
            plt.show()

    # plotframe(tis[-1])

    # seismo.plot_onto(show=True)


if not analysis_outfol.exists():
    os.makedirs(analysis_outfol)


def run_standard():
    # select targets (infiles, outfiles, invoke)
    targets: list[tuple[list[str], list[str], Callable[[], None]]] = []
    for vp2_ind in run_sims_by_vp_ind.keys():
        targets.append(
            (
                *compare_sims(vp2_ind, skip_and_get_filedeps_only=True),
                lambda vp2_ind=vp2_ind: compare_sims(vp2_ind, show=False),
            )
        )

    # sims_to_run = ["dg2_20_1"]
    sims_to_run = []
    for simname in sims_to_run:
        sim = run_sims_by_name[simname]
        targets.append(
            (
                *loadsim(sim, skip_and_get_filedeps_only=True),
                lambda sim=sim: loadsim(sim),
            )
        )

    # kill unneeded files
    files_to_keep = []
    for _, outfiles, _ in targets:
        files_to_keep.extend(outfiles)
    for f in analysis_outfol.iterdir():
        if str(f) not in files_to_keep:
            print(f"clearing {f}")
            os.remove(f)

    # run everything
    for infiles, outfiles, invoke in targets:
        if not should_skip_by_file_deps(infiles, outfiles, "", False):
            print(f"Running target for files {outfiles}")
            invoke()
        else:
            print(f"Skipping target for files {outfiles}")

    (output_fol / "analyze_timestamp").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--default", action="store_true")
    args = parser.parse_args()
    if args.default:
        run_standard()
        sys.exit(0)

    # for vp2_ind in run_sims_by_vp_ind.keys():
    #     compare_sims(vp2_ind,show=False)

    loadsim(run_sims_by_name["dg2_20_1"])
