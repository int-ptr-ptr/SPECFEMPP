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
