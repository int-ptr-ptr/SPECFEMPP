import json
import os
import pathlib
from typing import Iterable

import matplotlib.animation as mplanim
import matplotlib.patches as mplpatches
import matplotlib.pyplot as plt

from workflow.util import config
from workflow.util.dump_reader import dump_series
from workflow.util.seismo_reader import SeismoDump

workdir = pathlib.Path(__file__).parent
output_fol = workdir / "OUTPUT_FILES"

simout_fol = pathlib.Path(config.get("output_dir")) / "snell" / "FF"

runconfig = {}
with (output_fol / "run_out.json").open("r") as f:
    runconfig = json.load(f)

run_sims_by_vp_ind = dict()

for sim in runconfig["tasks_completed"]:
    vp2ind = sim["vp2_ind"]
    if vp2ind not in run_sims_by_vp_ind:
        run_sims_by_vp_ind[vp2ind] = list()

    run_sims_by_vp_ind[vp2ind].append(sim)


def compare_sims(vp2_ind: int, show: bool = False):
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
    seismo.plot_onto(
        show=show,
        legend_kwargs={"loc": "lower left"},
        plt_title=f"Seismogram comparison ($(v_p)_2 = {vp2:.2f}$)",
        save_filename=None
        if show
        else str(simout_fol / f"compare_seismo_{vp2_ind}.png"),
    )


def loadsim(simname: str, dt: float = 1e-3, frames: Iterable | None = None):
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
    seismo.plot_onto(axes=axs_seismos, indexing_func=lambda st, ty: st.station_index)

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

# for vp2_ind in run_sims_by_vp_ind.keys():
#     compare_sims(vp2_ind)
loadsim("dg2_20_1")
