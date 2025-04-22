import os
import pathlib
import sys
from typing import Literal

import matplotlib.animation as mplanim
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from workflow.theory.eigen.sys1d import (
    alpha,
    genpoly,
    get_residual_1D,
    residual_to_system,
)
from workflow.theory.eigen.sys1d import x as x_var
from workflow.util import config

workdir = pathlib.Path(__file__).parent

analysis_outfol = pathlib.Path(config.get("output_dir")) / "theory"
if not analysis_outfol.exists():
    os.makedirs(analysis_outfol)

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def flux_eigenvalues_plots():
    markers = "+x1234"
    lines = [(i * 2, (3, 5, 3, 2)) for i in range(5)]
    degs = [1, 2, 3, 4, 5]
    alphas = [0, 5, 10, 15, 20]
    alpha_maxeig_sample = np.linspace(0, 20, 201)
    schemes = [
        ("symmetric", "symmetric", 0),
        ("upwind", "upwind", 0),
        ("midpoint", "midpoint", 0),
        ("crossover", "midpoint", 1),
    ]
    fig, ax = plt.subplots(
        nrows=len(schemes), ncols=len(degs), figsize=(5 * len(degs), 5 * len(schemes))
    )

    fig_maxeig, ax_maxeig = plt.subplots(
        nrows=len(schemes), figsize=(10, 5 * len(schemes))
    )

    for ischeme, scheme in enumerate(schemes):
        for ideg, deg in enumerate(degs):
            polystring = r"$\mathcal P^{%d}$" % deg
            system = residual_to_system(
                get_residual_1D(method=scheme[1], deg=deg, w=scheme[2])  # type: ignore
            )

            maxrelam = np.empty(len(alpha_maxeig_sample))
            for ialpha, alphasub in enumerate(alpha_maxeig_sample):
                sys = np.array(
                    system.subs({alpha: alphasub}),
                    dtype=float,
                )
                maxrelam[ialpha] = np.max(np.real(np.linalg.eigvals(sys)))

            ax_maxeig[ischeme].plot(
                alpha_maxeig_sample, maxrelam, linestyle=lines[ideg], label=polystring
            )

            for ialpha, alphasub in enumerate(alphas):
                sys = np.array(
                    system.subs({alpha: alphasub}),
                    dtype=float,
                )
                eigenvals = np.linalg.eigvals(sys)
                # print(scheme[0],deg,alphasub, np.max(np.real(eigenvals)))
                ax[ischeme, ideg].scatter(
                    np.real(eigenvals),
                    np.imag(eigenvals),
                    marker=markers[ialpha],
                    label=(r" = " + str(alphasub)),
                )
            ax[ischeme, ideg].legend()
            ax[ischeme, ideg].set_title(f"{scheme[0]} {polystring} eigenvalues")
            ax[ischeme, ideg].set_xlabel(r"$\operatorname{Re} \lambda$")
            ax[ischeme, ideg].set_ylabel(r"$\operatorname{Im} \lambda$")

        maxstr = r"$\max \{ \operatorname{Re} \lambda \}$"
        ax_maxeig[ischeme].set_title(f"{scheme[0]} {maxstr}")
        ax_maxeig[ischeme].set_xlabel(r"$\alpha$")
        ylim = max(ax_maxeig[ischeme].get_ylim()[1], 1)
        ax_maxeig[ischeme].set_ylim(-0.05 * ylim, ylim)
        ax_maxeig[ischeme].set_ylabel(maxstr)
        ax_maxeig[ischeme].legend()
    fig.suptitle(
        "Eigenvalues of 1D flux-enforced periodic polynomial element",
        fontsize=20,
        va="top",
        y=0.92,
    )
    plt.figure(fig)
    plt.savefig(analysis_outfol / "flux1d_eig.png")
    fig_maxeig.suptitle(
        r"$\max \{ \operatorname{Re} \lambda \}$ for different schemes vs. $\alpha$",
        fontsize=20,
        va="top",
        y=0.92,
    )
    plt.figure(fig_maxeig)

    plt.savefig(analysis_outfol / "flux1d_maxeig.png")


def flux_eigenvalue_anim():
    markers = "+x1234"
    degs = [1, 2, 3, 4, 5]
    schemes = [
        ("symmetric", "symmetric", 0),
        ("upwind", "upwind", 0),
        ("midpoint", "midpoint", 0),
        ("crossover", "midpoint", 1),
    ]

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 9))
    axind = [(0, 0), (0, 1), (1, 0), (1, 1)]

    polystring = [r"$\mathcal P^{%d}$" % deg for deg in degs]

    scatters = [
        [
            ax[axind[i]].scatter([], [], marker=markers[j], label=polystring[j])
            for j in range(len(degs))
        ]
        for i in range(len(schemes))
    ]
    systems = [
        [
            residual_to_system(get_residual_1D(method=scheme[1], deg=deg, w=scheme[2]))  # type:ignore
            for deg in degs
        ]
        for scheme in schemes
    ]

    def get_eigs(ischeme, ideg, alphasub):
        sys = np.array(
            systems[ischeme][ideg].subs({alpha: alphasub}),
            dtype=float,
        )
        eigenvals = np.linalg.eigvals(sys)
        re = np.real(eigenvals)
        im = np.imag(eigenvals)
        return re, im

    for ischeme, scheme in enumerate(schemes):
        ax[axind[ischeme]].set_title(scheme[0])
        ax[axind[ischeme]].set_xlabel(r"$\operatorname{Re} \lambda$")
        ax[axind[ischeme]].set_ylabel(r"$\operatorname{Im} \lambda$")
        for ideg in range(len(degs)):
            xlow = 0
            xhigh = 0
            ylow = 0
            yhigh = 0
            for alpha_ in [0, 0.1, 1, 10, 20]:
                re, im = get_eigs(ischeme, ideg, alpha_)
                xlow = min(xlow, np.min(re))
                xhigh = max(xhigh, np.max(re))

                ylow = min(ylow, np.min(im))
                yhigh = max(yhigh, np.max(im))

            xmarg = max((xhigh - xlow) * 0.05, 0.01)
            ymarg = max((yhigh - ylow) * 0.05, 0.01)
            ax[axind[ischeme]].set_xlim(xlow - xmarg, xhigh + xmarg)
            ax[axind[ischeme]].set_ylim(ylow - ymarg, yhigh + ymarg)

    ax[axind[0]].legend(bbox_to_anchor=(0, 0), loc="lower left")

    def update(frame):
        for ischeme in range(len(schemes)):
            for ideg in range(len(degs)):
                re, im = get_eigs(ischeme, ideg, frame)

                scatters[ischeme][ideg].set_offsets(np.stack([re, im], axis=1))

        fig.suptitle(
            f"Eigenvalues of 1D flux-enforced polynomial element ($\\alpha$ = {frame:.1f})"
        )
        return sum(scatters, start=[])

    ani = mplanim.FuncAnimation(
        fig, update, frames=np.linspace(0, 20, 100), interval=1000 // 30
    )

    ani.save(analysis_outfol / "flux1d_anim.mp4")


def plot_eig(
    vec,
    val: complex,
    plt_title: str | None = None,
    plot_2d: bool = False,
    indicators_tracepath: bool = True,
    xticks=(-1, -0.5, 0, 0.5, 1),
    save_file_prefix: str | None = None,
    show: bool = False,
    ax=None,
):
    X = np.linspace(-1, 1, 100)
    dim = len(vec) // 2

    B = sp.Matrix([genpoly(k) for k in range(dim)])
    eigenfunc = sp.lambdify(x_var, (B.T @ vec[:dim])[0], "numpy")
    Y = np.array(eigenfunc(X), dtype=complex)

    mag = np.max(np.abs(Y)) * 1.05

    def replnan(val, repl):
        return repl if np.isnan(val) else val

    if abs(np.real(val)) < mag * 1e-2:
        if abs(np.imag(val)) < 1e-3:
            tmax = 1
        else:
            tmax = 2 * np.pi / np.abs(np.imag(val))
        pathlen = np.imag(val) * tmax
    else:
        tmax = max(np.log(0.1) / np.real(val), np.log(2) / np.real(val))
        if abs(np.imag(val)) > 1e-3:
            tmax = min(replnan(2 * np.pi / np.abs(np.imag(val)), tmax), tmax)
        pathlen = (np.exp(tmax * np.real(val)) - 1) * abs(val) / np.real(val)
    t_images = np.linspace(0, tmax, max(1 + round(pathlen), 5))
    alphas = 1 - t_images / tmax
    switchpoint = 0.7
    alphas = np.where(
        alphas > switchpoint, alphas**4, switchpoint**4 * (alphas / switchpoint)
    )

    # maximize real( Y )^2. This is the mathematical way.
    rot = np.sqrt(np.conjugate(np.mean(Y**2)))
    rot /= abs(rot)
    Y *= rot

    if plot_2d:
        if ax is None:
            ax = plt.gca()
        ax.plot([-1, 1], [0, 0], "k")
        for t, alpha in zip(reversed(t_images), reversed(alphas)):
            ax.plot(X, np.real(Y * np.exp(val * t)), "b", alpha=alpha)
        ax.set_xticks(xticks)
        ax.set_xlim(-1, 1)
    else:
        Y *= 1j
        if ax is None:
            ax = plt.figure().add_subplot(projection="3d")

        ax.set_proj_type("persp")  # type:ignore
        ax.plot([-1, 1], [0, 0], [0, 0], "k")

        ax.set_xlim(-1, 1)
        ax.set_ylim(-mag, mag)
        ax.set_zlim(-mag, mag)  # type: ignore

        def make_indicator(x, alpha):
            y = eigenfunc(x) * rot * 1j
            if indicators_tracepath:
                # trace until tiny |e^{t lam}| < 0.01
                tmax_trace = tmax * 3 if np.real(val) < 0 else -tmax * 10
                ind = np.linspace(
                    0 if np.real(val) < 0 else t_images[-2],
                    tmax_trace,
                    max(5, 1 + round(pathlen * 25)),
                )
                path = y * np.exp(ind * val)
            else:
                ind = np.linspace(0, 1, 2)
                path = y * ind
            ax.plot(
                ind * 0 + x, np.real(path), np.imag(path), color=(0.5, 0.5, 0.5, alpha)
            )

        for xind in xticks:
            ax.plot([xind, xind], [0, mag * 0.1], [0, 0], "k")
            make_indicator(xind, 0.5)

        indicators = np.linspace(-1, 1, 30)
        for xind in indicators:
            make_indicator(xind, 0.2)

        for t, alpha in zip(t_images, alphas):
            Y_ = Y * np.exp(t * val)
            ax.plot(X, np.real(Y_), np.imag(Y_), "b", alpha=alpha)
        ax.set_axis_off()
    if plt_title is not None:
        ax.set_title(plt_title)

    if save_file_prefix is not None:
        plt.savefig(analysis_outfol / f"{save_file_prefix}.pdf")
        plt.clf()
        plt.close()
    elif show:
        plt.show()
        plt.clf()
        plt.close()


def eigenfunctions(
    scheme: Literal["symmetric", "midpoint", "crossover", "upwind"],
    deg: int,
    alpha: float,
    eig_filter=lambda lam: abs(np.real(lam)) > 1e-5,
):
    w = 0
    if scheme == "crossover":
        w = 1
        scheme_ = "midpoint"
    else:
        scheme_ = scheme

    eigs = np.linalg.eig(
        np.array(
            residual_to_system(
                get_residual_1D(method=scheme_, deg=deg, alpha=alpha, w=w)
            ),
            dtype=float,
        )
    )

    eig_filter_ = np.array([eig_filter(lam) for lam in eigs.eigenvalues])
    num_eigs = np.count_nonzero(eig_filter_)
    fig, ax = plt.subplots(
        ncols=num_eigs, figsize=(num_eigs * 3, 3), subplot_kw={"projection": "3d"}
    )
    fig.suptitle(
        f"Eigenfunctions in $\\mathcal P^{deg}$ ({scheme}, $\\alpha$ = {alpha:.1f})"
    )

    eig_ind = 0
    for i, lam in enumerate(eigs.eigenvalues):
        if not eig_filter_[i]:
            continue
        vec = eigs.eigenvectors[:, i]
        plot_eig(
            vec,
            lam,
            f"$\\lambda$ = {lam:.3}",
            save_file_prefix=f"eig_{scheme}{deg}" if eig_ind == num_eigs - 1 else None,
            ax=ax[eig_ind],
        )
        eig_ind += 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pass
    else:
        # flux_eigenvalues_plots()
        # flux_eigenvalue_anim()
        eigenfunctions("symmetric", deg=4, alpha=0)
        eigenfunctions("upwind", deg=4, alpha=0)
        eigenfunctions("midpoint", deg=4, alpha=1)
        eigenfunctions("crossover", deg=4, alpha=1)
