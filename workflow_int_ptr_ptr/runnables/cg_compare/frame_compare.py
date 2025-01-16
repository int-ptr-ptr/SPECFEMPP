import matplotlib.pyplot as plt
import matplotlib.colors as mpcol
import util.dump_reader
import util.dump_reader_aux
import os
import numpy as np


def compare_frames(
    test: util.dump_reader.dump_frame,
    prov: util.dump_reader.dump_frame,
    mapper: util.dump_reader_aux.field_remapper,
    t: float,
    show: bool = False,
    save_filename: str | None = None,
    precomputed_test_displacement_norm: np.ndarray | None = None,
    precomputed_prov_displacement_norm: np.ndarray | None = None,
    precomputed_displacement_error: np.ndarray | None = None,
    clear_on_completion=False,
):
    prov_dispnorm = (
        np.linalg.norm(prov.displacement, axis=-1)
        if precomputed_prov_displacement_norm is None
        else precomputed_prov_displacement_norm
    )
    test_dispnorm = (
        np.linalg.norm(test.displacement, axis=-1)
        if precomputed_test_displacement_norm is None
        else precomputed_test_displacement_norm
    )
    disperr = (
        np.linalg.norm(test.displacement - mapper(prov.displacement), axis=-1)
        if precomputed_displacement_error is None
        else precomputed_displacement_error
    )
    vmin = min(np.min(prov_dispnorm), np.min(test_dispnorm))
    vmax = max(np.max(prov_dispnorm), np.max(test_dispnorm))

    errvmin = 0
    errvmax = np.max(disperr)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    plt.sca(ax[0])
    test.plot_field(
        test_dispnorm,
        show=False,
        mode="contour",
        vmin=vmin,
        vmax=vmax,
        title="dG displacement $|s_{dG}|$",
    )
    plt.sca(ax[1])
    prov.plot_field(
        prov_dispnorm,
        show=False,
        mode="contour",
        vmin=vmin,
        vmax=vmax,
        title="cG displacement $|s_{cG}|$",
    )
    plt.sca(ax[2])
    test.plot_field(
        disperr,
        show=False,
        title="displacement error $|s_{cG} - s_{dG}|$ (on dG nodes)",
        mode="scatter",
        ptsize=5,
        plt_cell_margin=0.4,
        vmin=errvmin,
        vmax=errvmax,
    )
    # color bars
    axloc = ax[1].get_position().bounds
    cax = fig.add_axes((axloc[0] + axloc[2] + 0.01, axloc[1], 0.01, axloc[3]))
    fig.colorbar(
        cax=cax,
        mappable=None,  # type: ignore
        norm=mpcol.Normalize(vmin=vmin, vmax=vmax, clip=True),
    )

    axloc = ax[2].get_position().bounds
    ax[2].set_position((axloc[0] + 0.03, axloc[1], axloc[2], axloc[3]))
    axloc = ax[2].get_position().bounds
    cax = fig.add_axes((axloc[0] + axloc[2] + 0.01, axloc[1], 0.01, axloc[3]))
    fig.colorbar(
        cax=cax,
        mappable=None,  # type: ignore
        norm=mpcol.Normalize(vmin=errvmin, vmax=errvmax, clip=True),
    )
    fig.suptitle(f"dG vs cG (SPECFEM) t = {t:.2f}")
    if show:
        plt.show()

    if save_filename is not None:
        plt.savefig(save_filename)

    if clear_on_completion:
        plt.clf()
        plt.close()


if __name__ == "__main__":
    import util.config as config

    test = config.get("cg_compare.tests.0")
    folder = os.path.join(config.get("cg_compare.workspace_folder"), test["name"])

    prov = util.dump_reader.dump_series.load_from_file(
        os.path.join(folder, config.get("cg_compare.workspace_files.provenance_dump"))
    )
    framenum = int(np.argmin(np.abs(prov.time_indices - 300)))
    prov_frame = prov.get_frame_as_dump_frame(framenum)
    test_frame = util.dump_reader.read_dump_file(
        os.path.join(folder, config.get("cg_compare.workspace_files.dump_prefix"))
        + str(prov.time_indices[framenum])
        + ".dat"
    )
    remap = util.dump_reader_aux.field_remapper(prov_frame.pts, test_frame.pts)

    compare_frames(
        test_frame,
        prov_frame,
        remap,
        float(config.get("cg_compare.dt")) * prov.time_indices[framenum],
        show=True,
    )
