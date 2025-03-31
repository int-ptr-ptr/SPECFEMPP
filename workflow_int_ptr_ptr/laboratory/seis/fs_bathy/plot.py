import os

import workflow.util.seismo_reader as seismo_reader

dirname = os.path.dirname(__file__)
seismos = seismo_reader.SeismoDump(os.path.join(dirname, "OUTPUT_FILES/STATIONS"))
seismos.load_from_seismodir(
    os.path.join(dirname, "OUTPUT_FILES/seismo"),
    linestyle="dashdot",
    label="cG",
    color="r",
)
seismos.load_from_seismodir(
    os.path.join(dirname, "OUTPUT_FILES/seismo_dg"),
    linestyle="dotted",
    label="dG",
    color="g",
)

seismos.plot_onto(
    save_filename=os.path.join(dirname, "OUTPUT_FILES/seiscomp.png"), legend_kwargs={}
)


os.system(
    f"ffmpeg -i {os.path.join(dirname, 'OUTPUT_FILES/display/wavefield%04d00.png')} {os.path.join(dirname, 'OUTPUT_FILES/anim.mp4')} -y"
)
os.system(
    f"ffmpeg -i {os.path.join(dirname, 'OUTPUT_FILES/display_dg/wavefield%04d00.png')} {os.path.join(dirname, 'OUTPUT_FILES/anim_dg.mp4')} -y"
)
