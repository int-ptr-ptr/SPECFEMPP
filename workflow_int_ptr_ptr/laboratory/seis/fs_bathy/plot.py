import os

import numpy as np

import workflow.util.seismo_reader as seismo_reader

dirname = os.path.dirname(__file__)
seismos = seismo_reader.SeismoDump(os.path.join(dirname, "OUTPUT_FILES/STATIONS"))
true = seismos.load_from_seismodir(
    os.path.join(dirname, "OUTPUT_FILES/seismo"),
    linestyle="dashdot",
    label="cG",
    color="r",
)
test = seismos.load_from_seismodir(
    os.path.join(dirname, "OUTPUT_FILES/seismo_dg"),
    linestyle="dotted",
    label="dG",
    color="g",
)

seismos.plot_onto(
    save_filename=os.path.join(dirname, "OUTPUT_FILES/seiscomp.pdf"),
    legend_kwargs={},
    plt_title="Pressure Seismograms",
)

errs = [
    None
    if true is None or test is None
    else np.stack([true[:, 0], (test[:, 1] - true[:, 1]) / np.max(true[:, 1])], axis=1)
    for true, test in zip(
        list(seismos.seismos[true]._seismos.values())[0],
        list(seismos.seismos[test]._seismos.values())[0],
    )
]
for i, err in enumerate(errs):
    list(seismos.seismos[test]._seismos.values())[0][i] = err
del seismos.seismos[true]

seismos.plot_onto(
    save_filename=os.path.join(dirname, "OUTPUT_FILES/seiscomp_err.pdf"),
    legend_kwargs={},
    plt_title="Relative error vs. time",
)

os.system(
    f"ffmpeg -i {os.path.join(dirname, 'OUTPUT_FILES/display/wavefield%04d00.png')} {os.path.join(dirname, 'OUTPUT_FILES/anim.mp4')} -y"
)
os.system(
    f"ffmpeg -i {os.path.join(dirname, 'OUTPUT_FILES/display_dg/wavefield%04d00.png')} {os.path.join(dirname, 'OUTPUT_FILES/anim_dg.mp4')} -y"
)
