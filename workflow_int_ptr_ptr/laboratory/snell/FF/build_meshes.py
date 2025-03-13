import json
import os


def save_parfile(
    fname: str,
    nx: int,
    ny: int,
    vp2: float,
    vs1: float = 0,
    vs2: float = 0,
    use_stacey_BCs: bool = True,
    staceyLR: bool = True,
    staceyTB: bool = True,
    outfol: str = "OUTPUT_FILES",
    database_out: str = "mesh_out",
    stations_out="STATIONS",
    topo_file: str = "topo_unit_box.dat",
):
    folder = os.path.dirname(fname)
    # this method is populated at the end of the file for readability
    parfile, topo = gen_parfile_text(
        nx=nx,
        ny=ny,
        vp1=1,
        vs1=vs1,
        vp2=vp2,
        vs2=vs2,
        stacey=use_stacey_BCs,
        absLR=staceyLR,
        absTB=staceyTB,
        outfol=os.path.join(folder, outfol),
        database_out=database_out,
        stations_out=stations_out,
        topo_in_location=topo_file,
    )  # pyright: ignore
    with open(fname, "w") as f:
        f.write(parfile)

    with open(os.path.join(folder, topo_file), "w") as f:
        f.write(topo)


# the end of the file calls this. Consider this def as
# if __name__ == "__main__"
def call():
    dirname = os.path.dirname(__file__)

    _counter = dict()

    def snell_FF_init_parfile(nx, vp2):
        if vp2 in _counter:
            vpind = _counter[vp2]
        else:
            vpind = len(_counter)
            _counter[vp2] = vpind
        save_parfile(
            os.path.join(dirname, f"Par_File{nx}_{vpind}"),
            nx,
            nx,
            vp2,
            database_out=f"mesh{nx}_{vpind}",
            stations_out=f"stations{nx}_{vpind}",
            topo_file=f"topo_unit_box{nx}.dat",
        )

    for nx in [10, 20]:
        for vp2 in [0.25, 0.5, 1, 2, 4]:
            snell_FF_init_parfile(nx, vp2)
    with open(os.path.join(dirname, "meshconf.json"), "r") as f:
        json.dump({"vp2": _counter}, f)


def gen_parfile_text(
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
        f"""title                           = snell 1x1 square
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
xdeb                            = 0.35
zdeb                            = 0.35
xfin                            = 0.65
zfin                            = 0.35
record_at_surface_same_vertical = .false.
nrec                            = 3
xdeb                            = 0.35
zdeb                            = 0.65
xfin                            = 0.65
zfin                            = 0.65
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
        f"2\n2\n0 0\n1 0\n2\n0 1\n1 1\n{ny // 2}",
    )


if __name__ == "__main__":
    call()
