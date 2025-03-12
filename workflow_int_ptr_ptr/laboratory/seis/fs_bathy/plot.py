import sys
import os

d = __file__
while not os.path.isdir(os.path.join(d, "runnables")):
    d = os.path.dirname(d)
sys.path.insert(0, d)
del d

import runnables.util.seismo_reader as seismo_reader  # noqa: E402


dirname = os.path.dirname(__file__)
seismos = seismo_reader.SeismoDump(os.path.join(dirname, "OUTPUT_FILES/STATIONS"))
seismos.load_from_seismodir(os.path.join(dirname, "OUTPUT_FILES/seismo"))
seismos.load_from_seismodir(os.path.join(dirname, "OUTPUT_FILES/seismo_dg"))
seismos.seismos[0].plot_linestyle = "dashdot"
seismos.seismos[1].plot_linestyle = "dotted"

seismos.plot_onto(show=True)
