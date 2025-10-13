# `ncmarked_conforming_grid`

A 4 x 4 grid of elements with the bottom two rows as elastic and the top two as acoustic.

To regenerate the database, convert the topography file into the files in `MESH` through `gmsh`, then run meshfem over `Par_file`:

```bash
python scripts/gmshlayerbuilder simple_dg_topo.dat MESH
xmeshfem2D -p Par_file
```


Relative to `provenance`:

```bash
python ../../../../../../scripts/gmshlayerbuilder simple_dg_topo.dat MESH
../../../../../../bin/xmeshfem2D -p Par_file
```
