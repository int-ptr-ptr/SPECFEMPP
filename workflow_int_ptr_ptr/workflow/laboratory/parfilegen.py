import os


def meshfem_config(
    nx: int,
    ny: int,
    vp1: float,
    vs1: float,
    vp2: float,
    vs2: float,
    stacey: bool,
    absLR: bool,
    absTB: bool,
    outfol: str,
    database_out: str,
    stations_out: str,
    topo_in_location: str,
):
    stacey_st = "true" if stacey else "false"
    absLR_st = "true" if absLR else "false"
    absTB_st = "true" if absTB else "false"
    return (
        f"""title                           = generated box parfile
NPROC                           = 1
OUTPUT_FILES                   = {outfol}

PARTITIONING_TYPE               = 3
NGNOD                           = 9
database_filename               = {os.path.join(outfol, database_out)}
use_existing_STATIONS           = .false.
nreceiversets                   = 2
anglerec                        = 0.d0
rec_normal_to_surface           = .false.
nrec                            = 3
xdeb                            = 0.5
zdeb                            = 0.7
xfin                            = 1
zfin                            = 0.7
record_at_surface_same_vertical = .false.
nrec                            = 3
xdeb                            = 1
zdeb                            = 0.45
xfin                            = 0.5
zfin                            = 0.45
record_at_surface_same_vertical = .false.
stations_filename              = {os.path.join(outfol, stations_out)}
nbmodels                        = 2
1 1 1 {vp1} {vs1} 0 0 9999 9999 0 0 0 0 0 0
2 1 1 {vp2} {vs2} 0 0 9999 9999 0 0 0 0 0 0
TOMOGRAPHY_FILE                 = ./DATA/tomo_file.xyz
read_external_mesh              = .false.
mesh_file                       = ./DATA/mesh_file
nodes_coords_file               = ./DATA/nodes_coords_file
materials_file                  = ./DATA/materials_file
free_surface_file               = ./DATA/free_surface_file
axial_elements_file             = ./DATA/axial_elements_file
absorbing_surface_file          = ./DATA/absorbing_surface_file
acoustic_forcing_surface_file   = ./DATA/MSH/Surf_acforcing_Bottom_enforcing_mesh
absorbing_cpml_file             = ./DATA/absorbing_cpml_file
tangential_detection_curve_file = ./DATA/courbe_eros_nodes
interfacesfile                  = {topo_in_location}
xmin                            = 0.d0
xmax                            = 1.d0
nx                              = {nx}
STACEY_ABSORBING_CONDITIONS     = .{stacey_st}.
absorbbottom                    = .{absTB_st}.
absorbright                     = .{absLR_st}.
absorbtop                       = .{absTB_st}.
absorbleft                      = .{absLR_st}.
nbregions                       = 2
1 {nx}  1 {ny // 2} 1
1 {nx}  {(ny // 2) + 1} {ny} 2
output_grid_Gnuplot             = .false.
output_grid_ASCII               = .false.
""",
        f"2\n2\n0 0\n1 0\n2\n0 1\n1 1\n{ny}",
    )


def sf_config(
    dt: float,
    nstep: float,
    outfolder: str,
    out_seismo: str,
    out_disp: str | None,
    stations: str,
    seismo_step_between_samples: int,
    database_in: str,
    source_in: str,
    domain_sep_subdivision: int | None,
):
    if out_disp is None:
        dispstr = ""
    else:
        dispstr = f"""
          display:
            format: PNG
            directory: {os.path.join(outfolder, out_disp)}
            field: displacement
            simulation-field: forward
            time-interval: 100
"""
    if domain_sep_subdivision is None:
        meshmod = ""
    else:
        meshmod = f"""
  mesh-modifiers:
    subdivisions:
      - material: 2
        x: {domain_sep_subdivision}
        z: {domain_sep_subdivision}
    interface-rules:
      - material1: 1
        material2: 2
        rule: domain-separation
"""

    return f"""parameters:
  header:
    title: FF_snell
    description: |
      Material systems : Acoustic domain (2)
      Interfaces : ...
      Sources : ...
      Boundary conditions : ...

  simulation-setup:
    quadrature:
      quadrature-type: GLL4

    solver:
      time-marching:
        type-of-simulation: forward
        time-scheme:
          type: Newmark
          dt: {dt:e}
          nstep: {nstep}

    simulation-mode:
      forward:
        writer:
          seismogram:
            format: ascii
            directory: {os.path.join(outfolder, out_seismo)}
{dispstr}

  receivers:
    stations: {stations}
    angle: 0.0
    seismogram-type:
      - pressure
    nstep_between_samples: {seismo_step_between_samples}

  run-setup:
    number-of-processors: 1
    number-of-runs: 1

  databases:
    mesh-database: {database_in}

  sources: {source_in}
{meshmod}
"""
