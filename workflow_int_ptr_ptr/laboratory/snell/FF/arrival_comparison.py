import os
import pathlib
import sys

import analyze  # pyright: ignore
import experiment  # pyright: ignore
import matplotlib.lines as mplines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import workflow.util.seismo_reader as seismo_reader
from workflow.analysis.seismo import seismo_fft
from workflow.util import config

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

WORKDIR = pathlib.Path(__file__).parent
OUTFOL = WORKDIR / "OUTPUT_FILES"
EXPORT_FOL = pathlib.Path(config.get("output_dir")) / "snell" / "FF"


def get_all_sims():
    sims = []
    for f_ in os.listdir(OUTFOL):
        f = OUTFOL / f_
        if f.exists() and (f / "dumps.dat").exists():
            sims.append(experiment.Simulation.from_str(f_))
    return sims


def reader_from_vpind(vpind):
    reader = seismo_reader.SeismoDump(str(OUTFOL / "stations"))
    sims = []
    for sim in get_all_sims():
        if sim.vp2_ind == vpind:
            if sim.disappearing_information is None:
                sim.disappearing_information = dict()

            sim.disappearing_information["seismo_collection_ind"] = (
                reader.load_from_seismodir(str(OUTFOL / sim.simname()))
            )
            sims.append(sim)
    return reader, sims


def display_domain(
    ax, do_xlabel: bool = True, stations=None, vp2=None, insert_labels=False
):
    if stations is None:
        stations = seismo_reader.SeismoDump(str(OUTFOL / "stations")).stations
    ax.plot([0, 1], [0.5, 0.5], ":k")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    kwargs = dict()
    if insert_labels:
        kwargs["label"] = "Source"
    ax.scatter(
        [analyze.sourceloc[0]], [analyze.sourceloc[1]], color="b", marker="*", **kwargs
    )

    kwargs = dict()
    if insert_labels:
        kwargs["label"] = "Station"
    ax.scatter(
        [station.x for station in stations],
        [station.z for station in stations],
        color="c",
        marker="o",
        **kwargs,
    )
    ax.annotate(
        "c = $c_1$ = 1.00",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=8,
    )
    ax.annotate(
        f"c = $c_2$ = {vp2:.2f}" if vp2 is not None else "c = $c_2$",
        xy=(0.01, 1 - 0.01),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=8,
    )
    ax.set_aspect(1)
    ax.set_ylabel("$z/L$ position")
    if do_xlabel:
        ax.set_xlabel("$x/L$ position")


def recover_ground_truth(simlist):
    maxN = 0
    ref_sim = None
    for sim in simlist:
        if sim.scheme == "cont" and sim.N > maxN:
            ref_sim = sim
            maxN = sim.N
    return ref_sim


def compare_at_arrivals(
    vp2ind: int,
    targets: list[tuple[int, float]],
    verbose: bool = False,
    graceful_errors: bool = False,
    save_filename: str | None = None,
    show: bool | None = None,
    close_on_complete: bool = True,
    clear_on_complete: bool = True,
):
    if show is None:
        show = save_filename is None

    vp2 = experiment._vp2_vals[vp2ind]
    reader, sims = reader_from_vpind(vp2ind)
    tmax = max(
        max(
            max(max(arr[:, 0]) for arr in stdata if arr is not None)
            for stdata in seis._seismos.values()
        )
        for seis in reader.seismos
    )
    ground_truth = recover_ground_truth(sims)
    ground_truth_ind = ground_truth.disappearing_information["seismo_collection_ind"]  # type:ignore
    ground_truth_collection = reader.seismos[ground_truth_ind]
    arrivals, crossovers = analyze.compute_arrivals(
        reader, vp2=vp2, tmax=tmax, store_crossovers=True
    )
    do_phase_offsets = False
    do_amp_ratio = False
    do_freq_domain = False
    ncols = 3 + do_freq_domain + do_phase_offsets + do_amp_ratio
    fig, axes = plt.subplots(
        nrows=len(targets),
        ncols=ncols,
        figsize=(15, 3 * len(targets)),
        gridspec_kw={"width_ratios": [0.5] + [1] * (ncols - 1)},
    )
    fig.subplots_adjust(wspace=0.15, left=0.02, right=0.98)
    ax_disp = 0
    ax_wave = 1
    ax_err = 2
    ax_amps = 3 if do_freq_domain else 9999
    ax_amp_err = (3 + do_freq_domain) if do_amp_ratio else 9999
    ax_phase = (3 + do_freq_domain + do_amp_ratio) if do_phase_offsets else 9999

    linestyles = [
        None,
        (0, (1, 5)),
        (0, (1, 1)),
        (0, (2, 2, 1, 2)),
        (0, (2, 2, 2, 2)),
        "-",
    ]
    markers = [None, "1", "+", "v", "s", "."]
    subdiv_markers = {1: "$1$", 2: "$2$", 3: "$3$"}
    colors = {
        "cont": "r",
        "symm": "g",
        "upwind": "b",
        "mid": "c",
        "crossover": "m",
    }
    wchar_bins = np.array([-np.inf])

    for itarget, target in enumerate(targets):
        collections = []
        ground_truth_fft = None

        windowsize_period = 0.5
        station_num = target[0]
        arrival_target = target[1]

        if arrivals[station_num]:
            station = reader.stations[station_num]
            gtdat = ground_truth_collection._seismos[7][station_num]
            arrival_ind = np.argmin(
                np.abs(np.array(arrivals[station_num]) - arrival_target)
            )
            arrival_time = arrivals[station_num][arrival_ind]

            crossover = crossovers[station_num][arrival_ind]
            if np.isnan(crossover[1]):
                xdata = np.array([analyze.sourceloc[0], crossover[0]])
                zdata = np.array([analyze.sourceloc[1], station.z])
            else:
                xdata = np.array([analyze.sourceloc[0], crossover[1], crossover[0]])
                zdata = np.array([analyze.sourceloc[1], 0.5, station.z])
            while True:
                axes[itarget, ax_disp].plot(xdata, zdata, "r")
                if 0 <= xdata[-1] and xdata[-1] <= 1:
                    break
                xdata -= np.sign(xdata[-1])
            display_domain(
                axes[itarget, ax_disp], do_xlabel=False, stations=[station], vp2=vp2
            )

            _lrweight = 0.67
            axes[itarget, ax_disp].annotate(
                f"Station {station_num}\n(t = {arrival_time:.2f})",
                xy=(
                    _lrweight * station.x + (1 - _lrweight) * 0.5,
                    station.z + (0.05 if station.z > 0.5 else -0.05),
                ),
                xycoords="data",
                va="bottom" if station.z > 0.5 else "top",
                ha="center",
                fontsize=8,
            )

            if verbose:
                print(
                    f"Station {station_num} @ ({station.x},{station.z}) with arrivals {arrivals[station_num]}."
                )
                print(f"Closest one to target {arrival_target} is {arrival_time}.")

            cliptimes = (
                arrival_time - windowsize_period / 2,
                arrival_time + windowsize_period / 2,
            )

            # ===========================wave

            for sim in sims:
                # characteristic frequency of domain (takes values 5,10,15,20,30)
                wchar = min(
                    sim.get_horiz_numcells()[0], sim.get_horiz_numcells()[1] * sim.vp2()
                )

                linestyle_closest = np.argmin(np.abs(wchar_bins - wchar))
                if np.abs(wchar_bins[linestyle_closest] - wchar) > 1e-2:
                    if len(wchar_bins) < len(linestyles):
                        wchar_bins = np.sort([*wchar_bins, wchar])

                wcharmin = -np.inf
                wcharmax = np.inf
                if wchar - 1e-6 <= wcharmin or wchar + 1e-6 >= wcharmax:
                    continue

                # print(sim.get_horiz_numcells())

                icoll = sim.disappearing_information["seismo_collection_ind"]
                coll = reader.seismos[icoll]

                sub = coll.clip_times(*cliptimes)
                col2 = seismo_fft(sub)

                if np.any(np.isnan(sub._seismos[7][station_num])):
                    print(sim)
                    continue

                scheme = sim.scheme
                if scheme == "mid":
                    if sim.scheme_params["XR"] < -0.2:
                        scheme = "crossover"
                        scheme_name = "Crossover flux"
                    else:
                        scheme_name = "Midpoint flux"
                elif scheme == "symm":
                    scheme_name = "Symmetric flux"
                elif scheme == "upwind":
                    scheme_name = "Upwind flux"
                else:
                    scheme_name = '"ground truth"'
                col2._sim_params = {  # type:ignore
                    "wchar": wchar,
                    "sim": sim,
                    "sub_coll": sub,
                    "scheme": scheme,
                    "scheme_label": scheme_name,
                    "color": colors[scheme],
                    "subdiv_marker": subdiv_markers[sim.get_subdivisions()[1]],
                }
                collections.append(col2)
                if icoll == ground_truth_ind:
                    ground_truth_fft = col2
            if verbose:
                print(f"characteristic frequency values: {wchar_bins}")
            ground_truth_max_Linf = np.max(
                np.abs(
                    ground_truth_fft._sim_params["sub_coll"]._seismos[7][station_num][  # type:ignore
                        :, 1
                    ]
                )
            )
            for icoll, coll in enumerate(collections):
                sub = coll._sim_params["sub_coll"]
                coll._sim_params["wchar_bin"] = np.argmin(
                    np.abs(wchar_bins - coll._sim_params["wchar"])
                )
                seisdat = sub._seismos[7][station_num]

                seisref = np.interp(seisdat[:, 0], gtdat[:, 0], gtdat[:, 1])
                err = seisdat[:, 1] - seisref
                # print(icoll, wchar, np.linalg.norm(err))

                # plt.scatter([wchar],[np.linalg.norm(err)])

                axes[itarget, ax_err].scatter(
                    [coll._sim_params["wchar"]],
                    [np.linalg.norm(err, ord=2) / np.linalg.norm(seisref, ord=2)],
                    marker=coll._sim_params["subdiv_marker"],
                    c=coll._sim_params["color"],
                    alpha=0.5,
                )
                axes[itarget, ax_wave].plot(
                    seisdat[:, 0],
                    seisdat[:, 1] / ground_truth_max_Linf,
                    linestyle=linestyles[coll._sim_params["wchar_bin"]],
                    color=coll._sim_params["color"],
                    # label = f"{sim.get_horiz_numcells()}, {sim.dt()}"
                )
            # ==========================freqdomain
            ground_truth_freq = ground_truth_fft._seismos[7][station_num]  # type:ignore
            datasize = ground_truth_freq.shape[0] // 2  # type:ignore
            ground_truth_freq = np.roll(ground_truth_freq, datasize, axis=0)  # type:ignore
            ground_truth_amp = np.real(np.abs(ground_truth_freq[:, 1]))
            ground_truth_phase = np.atan2(
                np.imag(ground_truth_freq[:, 1]), np.real(ground_truth_freq[:, 1])
            )
            ground_truth_freq = np.real(ground_truth_freq[:, 0])
            ground_truth_maxamp = np.max(ground_truth_amp)

            freqhighs = []
            for coll in collections:
                dat = coll._seismos[7][station_num]
                datasize = dat.shape[0] // 2
                dat = np.roll(dat, datasize, axis=0)
                amp = np.real(np.abs(dat[:, 1]))
                phase = np.atan2(np.imag(dat[:, 1]), np.real(dat[:, 1]))
                freq = np.real(dat[:, 0])
                if do_freq_domain:
                    axes[itarget, ax_amps].plot(
                        freq,
                        amp / ground_truth_maxamp,
                        linestyle=linestyles[coll._sim_params["wchar_bin"]],
                        color=coll._sim_params["color"],
                    )

                freq_includes = freq[np.logical_and(amp > 1e-1 * max(amp), freq >= 0)]
                freq_includes = freq_includes[
                    freq_includes < 3 * np.median(freq_includes)
                ]
                amp_includes = np.interp(freq_includes, freq, amp)
                amp_integ = np.sum(amp_includes)
                mean = np.sum(amp_includes * freq_includes) / amp_integ
                freqhighs.append(
                    mean
                    + 3
                    * np.sqrt(
                        np.sum(amp_includes * (freq_includes - mean) ** 2) / amp_integ
                    )
                )

                # diffusion
                amp_gt_interp = np.interp(freq, ground_truth_freq, ground_truth_amp)
                freq_exists_filter = amp_gt_interp > 1e-1 * np.max(amp_gt_interp)
                amp_gt_interp = amp_gt_interp[freq_exists_filter]
                amp_err = amp[freq_exists_filter] / amp_gt_interp
                filtered_freqs = freq[freq_exists_filter]
                # axes[itarget, 2].scatter(
                #     filtered_freqs, amp_err, s=coll._sim_params["wchar"],c=coll._sim_params["color"]
                # )

                if do_amp_ratio:
                    axes[itarget, ax_amp_err].plot(
                        filtered_freqs,
                        amp_err,
                        marker=markers[coll._sim_params["wchar_bin"]],
                        linestyle=linestyles[coll._sim_params["wchar_bin"]],
                        color=coll._sim_params["color"],
                    )

                # delay
                phase_error = (
                    np.mod(
                        phase[freq_exists_filter]
                        - np.interp(
                            filtered_freqs, ground_truth_freq, ground_truth_phase
                        )
                        + np.pi / 2,
                        np.pi,
                    )
                    - np.pi / 2
                )

                if do_phase_offsets:
                    # axes[itarget, 3].scatter(
                    #     filtered_freqs, phase_error, s=coll._sim_params["wchar"], c=coll._sim_params["color"]
                    # )
                    axes[itarget, ax_phase].plot(
                        filtered_freqs,
                        phase_error,
                        marker=markers[coll._sim_params["wchar_bin"]],
                        linestyle=linestyles[coll._sim_params["wchar_bin"]],
                        color=coll._sim_params["color"],
                    )

            freq_cap = np.mean(freqhighs)

            axes[itarget, ax_wave].set_ylabel("$\\frac{p(t)}{\\|p_{true}\\|_\\infty}$")

            axes[itarget, ax_err].set_yscale("log")
            # axes[itarget, ax_err].yaxis.set_major_formatter('{x:.2e}')
            # axes[itarget, ax_err].yaxis.set_minor_formatter('{x:.2e}')
            axes[itarget, ax_err].tick_params(axis="y", which="both", labelsize=7)
            axes[itarget, ax_err].set_ylabel(
                "$\\frac{\\|p - p_{true}\\|_2}{\\|p_{true}\\|_2}$"
            )

            gtwchar = ground_truth_fft._sim_params["wchar"]  # type:ignore
            axes[itarget, ax_err].axvline(x=gtwchar, color="lightgray")
            axes[itarget, ax_err].annotate(
                " ground truth \n $f_{char}$ ",
                xy=(gtwchar, 1 - 1e-2),
                xycoords=("data", "axes fraction"),
                va="top",
                ha="right",
                fontsize=10,
            )
            if do_freq_domain:
                axes[itarget, ax_amps].set_xlim(0, freq_cap)
                axes[itarget, ax_amps].set_ylabel(
                    "$\\frac{\\hat p(f)}{\\|\\hat p_{true}\\|_\\infty}$"
                )

            # diff err
            if do_amp_ratio:
                axes[itarget, ax_amp_err].set_xlim(0, freq_cap)
                axes[itarget, ax_amp_err].set_ylabel(
                    "$\\frac{\\hat p(f)}{\\hat p_{true}(f)}$"
                )
                axes[itarget, ax_amp_err].set_yscale("log")
                # axes[itarget, ax_amp_err].yaxis.set_major_formatter('{x:.2f}')
                # axes[itarget, ax_amp_err].yaxis.set_minor_formatter('{x:.2f}')
                axes[itarget, ax_amp_err].tick_params(
                    axis="y", which="both", labelsize=7
                )

            # delay
            if do_phase_offsets:
                axes[itarget, ax_phase].set_xlim(0, freq_cap)
            # ========================== diffusion

        else:
            if verbose:
                print(
                    f"Station {station_num} @ ({station.x},{station.z}) with no arrivals. Cannot analyze."  # type:ignore
                )
            if not graceful_errors:
                raise ValueError("Attempting to analyze a station with no arrivals.")
    axes[0, ax_wave].set_title("Pressure seismogram at arrival")
    axes[-1, ax_wave].set_xlabel("time $tc_1/L$")
    axes[0, ax_err].set_title("$L^2$ error of arrival")
    axes[-1, ax_err].set_xlabel("characteristic frequency $f_{char} L/c_1$")
    if do_freq_domain:
        axes[0, ax_amps].set_title("Frequency domain amplitude at arrival")
        axes[-1, ax_amps].set_xlabel("frequency $fL/c_{1}$")
    axes[-1, ax_err].legend(
        handles=[
            mpatches.Patch(color=colors["cont"], label="cG (no flux)"),
            mpatches.Patch(color=colors["symm"], label="Symmetric Flux"),
            mpatches.Patch(color=colors["upwind"], label="Upwind Flux"),
            mpatches.Patch(color=colors["mid"], label="Midpoint Flux"),
            mpatches.Patch(color=colors["crossover"], label="Crossover Flux"),
            mplines.Line2D(
                [], [], linestyle="", marker="$n$", label="1:n subdivision for medium 2"
            ),
            *[
                mplines.Line2D([], [], linestyle=ls, label="f_{char} = %.1f" % wch)
                for ls, wch in zip(linestyles[1:], wchar_bins[1:])
            ],
        ],
        prop={"size": 6},
    )
    axes[0, ax_disp].set_title("Path of signal")
    axes[-1, ax_disp].set_xlabel("$x/L$ position")

    if do_amp_ratio:
        axes[0, ax_amp_err].set_title("Frequency domain amplituide ratio")
        axes[-1, ax_amp_err].set_xlabel("frequency $fL/c_{1}$")
    if do_phase_offsets:
        axes[0, ax_phase].set_title("Frequency domain phase errors")
        axes[-1, ax_phase].set_xlabel("frequency $fL/c_{1}$")
    fig.suptitle(
        f"Seismogram comparison ($c_2 = {vp2}$) for various arrivals",
        y=0.93,
        va="bottom",
        fontsize=20,
    )

    if save_filename is not None:
        plt.savefig(save_filename)
    if show:
        plt.show()
    if clear_on_complete:
        plt.clf()
    if close_on_complete:
        plt.close()


def parse_targets(targets: list[list[str]]) -> list[tuple[int, float]]:
    return [(int(li[0]), float(li[1])) for li in targets]


if __name__ == "__main__":
    import json
    import re

    args = " ".join(sys.argv)
    m = re.search(r"!\s*TASK\s*\[([^\[\]]+)\]\s*!", args)
    if m:
        if re.search("domain", m.group(1)):
            display_domain(plt.gca(), insert_labels=True)
            plt.title("Domain $\\Omega$ of Snell experiment")
            plt.legend()
            plt.savefig(EXPORT_FOL / "domain.png")
            plt.savefig(EXPORT_FOL / "domain.pdf")
            sys.exit(0)
        data = dict()
        with open(WORKDIR / "arrival_comparison.json", "r") as f:
            data = json.load(f)

        vp2ind = int(m.group(1).strip())
        comp = None
        if vp2ind in data:
            comp = data[vp2ind]
        if comp is None and str(vp2ind) in data:
            comp = data[str(vp2ind)]
        if comp is None:
            print(f"Cannot parse task {m.group(1)}")
            sys.exit(1)
        if "export_prefix" in comp and "targets" in comp:
            save_fname = str(EXPORT_FOL / f"{comp['export_prefix']}.png")
            compare_at_arrivals(
                vp2ind=vp2ind,
                targets=parse_targets(comp["targets"]),
                save_filename=save_fname,
            )
        else:
            print(f"Task {m.group(1)}: no work to do. Exiting.")
        sys.exit(0)

    # m = re.search(r"!\s*CLEAN\s*!", args)
    # if m:
    #     clean_simwork()
    #     sys.exit(0)

    print(f'Failed to parse arguments "{args}"')
    sys.exit(1)
