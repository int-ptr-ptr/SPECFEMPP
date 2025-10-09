# `ncmarked_conforming_grid`

An 8 x 8 grid of elements with the bottom half as elastic and the top half as acoustic. This mesh has a sinusoidal surface of 1 wavelength and amplitude of 20% of the domain.

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
