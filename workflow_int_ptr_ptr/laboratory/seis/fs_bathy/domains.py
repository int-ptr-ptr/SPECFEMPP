import pathlib

import matplotlib.pyplot as plt
import numpy as np

import workflow.util.dump_reader as dump_reader

outfol = pathlib.Path(__file__).parent / "OUTPUT_FILES"

dcg = dump_reader.read_dump_file(str(outfol / "dumps_cg/d0.dat"))

medium1_pts = dcg.pts[dcg.medium_type == 0, ...]
medium2_pts = dcg.pts[dcg.medium_type == 1, ...]

if np.min(medium1_pts[..., 1]) > np.min(medium2_pts[..., 1]):
    medium1_pts, medium2_pts = medium2_pts, medium1_pts


def trace(medium_pts):
    xpts = np.unique(medium_pts[..., 0])
    dx = np.max(xpts[1:] - xpts[:-1]) * 1e-2
    xpts = np.concatenate([[xpts[0]], xpts[1:][xpts[1:] - xpts[:-1] > dx]])
    xbins = np.digitize(medium_pts[..., 0], (xpts[1:] + xpts[:-1]) / 2)
    lwall = medium_pts[xbins == 0, :]
    lwall = lwall[np.argsort(-lwall[:, 1]), :]
    rwall = medium_pts[xbins == np.max(xbins), :]
    rwall = rwall[np.argsort(rwall[:, 1]), :]
    min_vals = np.empty(len(xpts))
    max_vals = np.empty(len(xpts))
    for ibin in range(len(xpts)):
        collect = medium_pts[xbins == ibin, 1]
        min_vals[ibin] = np.min(collect)
        max_vals[ibin] = np.max(collect)
    return np.concatenate(
        [
            lwall,
            np.stack([xpts, min_vals], axis=1),
            rwall,
            np.flip(np.stack([xpts, max_vals], axis=1), axis=0),
        ],
        axis=0,
    )


path_ac = trace(medium2_pts)
path_el = trace(medium1_pts)

plt.plot(path_ac[:, 0], path_ac[:, 1], label="Acoustic domain")
plt.plot(path_el[:, 0], path_el[:, 1], label="Elastic domain")
plt.legend()
plt.gca().set_aspect(1)
plt.title("Acoustic-Elastic Domain")
plt.savefig(outfol / "fs_bathy.pdf")

dim_cg = dcg.nglob


ddg = dump_reader.read_dump_file(str(outfol / "dumps_dg/d0.dat"))
dim_dg = ddg.nglob


with open(outfol / "dof.txt", "w") as f:
    f.write(f"cG: {dim_cg}\ndG: {dim_dg}")
