import os
import re
import struct
import types
from typing import Literal

import matplotlib.patches as mplpatches
import matplotlib.pyplot as plt
import matplotlib.tri as mpltri
import numpy as np

from . import GLL


def get_type_formatters(type, size):
    if type == "i":
        npch = "i"
        if size == 1:
            sch = "b"
        elif size == 2:
            sch = "h"
        elif size == 4:
            sch = "i"
        elif size == 8:
            sch = "q"
        else:
            raise ValueError(f"signed integer with unsupported size {size}")
    elif type == "j":
        npch = "u"
        if size == 1:
            sch = "B"
        elif size == 2:
            sch = "H"
        elif size == 4:
            sch = "I"
        elif size == 8:
            sch = "Q"
        else:
            raise ValueError(f"unsigned integer with unsupported size {size}")
    elif type == "f":
        npch = "f"
        if size == 2:
            sch = "e"
        elif size == 4:
            sch = "f"
        elif size == 8:
            sch = "d"
        else:
            raise ValueError(f"float with unsupported size {size}")
    else:
        raise ValueError(f"unknown datatype {type}")
    npch += str(size)
    return npch, lambda buffer: struct.iter_unpack(sch, buffer)


TYPE_CODE = dict()
_type = None
for _type in [
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    *[np.dtype("i" + str(s)) for s in [1, 2, 4, 8]],
]:
    TYPE_CODE[_type] = "i"

for _type in [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    *[np.dtype("u" + str(s)) for s in [1, 2, 4, 8]],
]:
    TYPE_CODE[_type] = "j"
for _type in [
    float,
    np.float16,
    np.float32,
    np.float64,
    *[np.dtype("f" + str(s)) for s in [2, 4, 8]],
]:
    TYPE_CODE[_type] = "f"
del _type


def read_sfdump(fname):
    data = dict()
    with open(fname, "rb") as f:
        header = b""
        while byte := f.read(1):
            header += byte
            if byte == b"]":
                # found header. Match?
                m = re.search(
                    r"([\w\.]+)<(\w+)\(size=(\d+)B\)>\[([\d,]+)\]",
                    header.decode("ascii"),
                )
                if m is None:
                    continue
                name = m.group(1)
                datatype = m.group(2)
                typesize = int(m.group(3))
                datashape = [int(s) for s in m.group(4).split(",")]
                blocksize = np.prod(datashape) * typesize

                # use data type string to figure out what to do
                try:
                    npcode, decodefunc = get_type_formatters(datatype, typesize)
                except ValueError as e:
                    raise Exception(f"Failed to parse header {m.string}") from e

                # 1d -> nd mapping to populate arr
                def recover_inds(i):
                    inds = [None] * len(datashape)
                    for place in range(len(datashape) - 1, -1, -1):
                        inds[place] = i % datashape[place]
                        i //= datashape[place]
                    return inds

                # loading data into np array
                arr = np.empty(datashape, dtype=np.dtype(npcode))
                buf = bytearray(blocksize)
                f.readinto(buf)

                for i, v in enumerate(decodefunc(buf)):
                    arr[*recover_inds(i)] = v[0]

                data[name] = arr
                header = b""

    return data


def write_sfdump(fname, data):
    with open(fname, "wb") as f:
        for k, v in data.items():
            f.write(k.encode("ascii"))
            extent_str = ",".join(str(ax) for ax in v.shape)
            f.write(
                f"<{TYPE_CODE[v.dtype]}(size={v.itemsize}B)>[{extent_str}]".encode(
                    "ascii"
                )
            )
            f.write(v.tobytes())


def recover_edgevals(field, edgeID):
    return np.where(
        edgeID == 1,
        field[:, -1, :, ...],
        np.where(
            edgeID == 2,
            field[:, :, 0, ...],
            np.where(edgeID == 3, field[:, 0, :, ...], field[:, :, -1, ...]),
        ),
    )


def build_gll_namespace(ngllxi: int, ngllga: int, ngll_aux: int | None = None):
    if ngll_aux is None:
        ngll_capacity = max(ngllxi, ngllga)
    else:
        ngll_capacity = max(ngllxi, ngllga, ngll_aux)

    knots_xi = GLL.get_knots(ngllxi - 1)
    knots_ga = GLL.get_knots(ngllga - 1)

    Lxi = np.array(GLL.build_lagrange_polys(ngllxi - 1))
    Lxi_p = np.array(GLL.polyderiv(Lxi.T)).T
    Lxi_pp = np.array(GLL.polyderiv(Lxi_p.T)).T
    Lga = np.array(GLL.build_lagrange_polys(ngllga - 1))
    Lga_p = np.array(GLL.polyderiv(Lga.T)).T
    Lga_pp = np.array(GLL.polyderiv(Lga_p.T)).T

    gll_weights = np.zeros((ngll_capacity + 1, ngll_capacity))
    gll_polys = np.zeros((ngll_capacity + 1, ngll_capacity, ngll_capacity))
    gll_polyderivs = np.zeros((ngll_capacity + 1, ngll_capacity, ngll_capacity - 1))
    gll_knots = np.zeros((ngll_capacity + 1, ngll_capacity))
    for i in range(2, ngll_capacity + 1):
        gll_weights[i, :i] = GLL.get_lagrange_weights(i - 1)
        gll_knots[i, :i] = GLL.get_knots(i - 1)
        L = np.array(GLL.build_lagrange_polys(i - 1))
        gll_polys[i, :i, :i] = L
        gll_polyderivs[i, :i, : i - 1] = np.array(GLL.polyderiv(L.T)).T

    return types.SimpleNamespace(
        ngll_x=ngllxi,
        ngll_z=ngllga,
        ngll_capacity=ngll_capacity,
        knots_x=knots_xi,
        knots_z=knots_ga,
        L_x=Lxi,
        Lp_x=Lxi_p,
        Lpp_x=Lxi_pp,
        L_z=Lga,
        Lp_z=Lga_p,
        Lpp_z=Lga_pp,
        Lp_at_knots_x=np.einsum(
            "nk,ka->na",
            Lxi_p,
            knots_xi[np.newaxis, :] ** np.arange(ngllxi - 1)[:, np.newaxis],
        ),
        Lp_at_knots_z=np.einsum(
            "nk,ka->na",
            Lga_p,
            knots_ga[np.newaxis, :] ** np.arange(ngllga - 1)[:, np.newaxis],
        ),
        weights_x=GLL.get_lagrange_weights(ngllxi - 1),
        weights_z=GLL.get_lagrange_weights(ngllga - 1),
        weights=gll_weights,
        L=gll_polys,
        Lp=gll_polyderivs,
        knots=gll_knots,
    )


class dump_frame:
    def __init__(self, data_simfield=None, data_edges=None):
        if data_simfield is None:
            return

        self.data_simfield = data_simfield
        self.data_edges = data_edges
        nspec, ngllga, ngllxi = data_simfield["pts"].shape[1:]
        medium_inds = data_simfield["assembly_index_mapping"][
            data_simfield["index_mapping"], :
        ]

        self.is_fully_medium = np.any(medium_inds == -1, axis=(1, 2))
        self.medium_type = self.is_fully_medium.argmax(axis=-1)

        elastic_medium_id = 0
        acoustic_medium_id = 1
        if "medium_type_refs" in data_simfield:
            medium_type_refs = data_simfield["medium_type_refs"]
            acoustic_medium_id = medium_type_refs[0]
            elastic_medium_id = medium_type_refs[1]
        self.elastic_medium_id = elastic_medium_id
        self.acoustic_medium_id = acoustic_medium_id

        self.nglob = np.max(data_simfield["index_mapping"])
        nglob_acoustic = data_simfield["acoustic_field"].shape[0]
        nglob_elastic = data_simfield["elastic_field"].shape[0]

        chi = np.full((nspec, ngllga, ngllxi), np.nan)
        if nglob_acoustic > 0:
            chi[...] = np.where(
                medium_inds[..., acoustic_medium_id] == -1,
                np.nan,
                data_simfield["acoustic_field"][
                    medium_inds[..., acoustic_medium_id], 0
                ],
            )
        chi_ddot = np.full((nspec, ngllga, ngllxi), np.nan)
        if nglob_acoustic > 0:
            chi_ddot[...] = np.where(
                medium_inds[..., acoustic_medium_id] == -1,
                np.nan,
                data_simfield["acoustic_field_ddot"][
                    medium_inds[..., acoustic_medium_id], 0
                ],
            )
        mass_inv_acoustic = np.full((nspec, ngllga, ngllxi), np.nan)
        if nglob_acoustic > 0:
            mass_inv_acoustic[...] = np.where(
                medium_inds[..., acoustic_medium_id] == -1,
                np.nan,
                data_simfield["acoustic_mass_inverse"][
                    medium_inds[..., acoustic_medium_id], 0
                ],
            )

        mass_inv_elastic = np.full((nspec, ngllga, ngllxi), np.nan)
        if nglob_elastic > 0:
            mass_inv_elastic[...] = np.where(
                medium_inds[..., elastic_medium_id] == -1,
                np.nan,
                data_simfield["elastic_mass_inverse"][
                    medium_inds[..., elastic_medium_id], 0
                ],
            )

        pts = data_simfield["pts"].transpose((1, 2, 3, 0))

        if data_edges is None:
            self.GLL = build_gll_namespace(ngllxi, ngllga)
        else:
            self.GLL = build_gll_namespace(
                ngllxi, ngllga, data_edges["edge_pos"].shape[1]
            )

        Lxi = self.GLL.L_x
        Lga = self.GLL.L_z
        Lxi_p = self.GLL.Lp_x
        Lga_p = self.GLL.Lp_z

        knots_xi = self.GLL.knots_x
        knots_ga = self.GLL.knots_z
        ngll_capacity = self.GLL.ngll_capacity

        # (spec index, xi power, gamma power, P component)
        P_poly = np.einsum("sjia,ik,jl->slka", pts, Lxi, Lga)
        dPdxi_poly = np.einsum("sjia,ik,jl->slka", pts, Lxi_p, Lga)
        dPdga_poly = np.einsum("sjia,ik,jl->slka", pts, Lxi, Lga_p)

        chi_poly = np.einsum("sji,ik,jl->slk", chi, Lxi, Lga)
        # (spec index, xi power, gamma power)
        dXdxi_poly = np.einsum("sji,ik,jl->slk", chi, Lxi_p, Lga)
        dXdga_poly = np.einsum("sji,ik,jl->slk", chi, Lxi, Lga_p)

        # (spec index, xi index, gamma index,P component, deriv direction)
        dP = np.empty((nspec, ngllxi, ngllga, 2, 2))
        dP[:, :, :, :, 0] = np.einsum(
            "sbap,ai,bj->sjip",
            dPdxi_poly,
            knots_xi[np.newaxis, :] ** np.arange(ngllxi - 1)[:, np.newaxis],
            knots_ga[np.newaxis, :] ** np.arange(ngllga)[:, np.newaxis],
        )
        dP[:, :, :, :, 1] = np.einsum(
            "sbap,ai,bj->sjip",
            dPdga_poly,
            knots_xi[np.newaxis, :] ** np.arange(ngllxi)[:, np.newaxis],
            knots_ga[np.newaxis, :] ** np.arange(ngllga - 1)[:, np.newaxis],
        )

        dP_inv = np.linalg.inv(dP)

        # (spec index, xi index, gamma index, deriv direction)
        dX_local = np.empty((nspec, ngllxi, ngllga, 2))
        dX_local[:, :, :, 0] = np.einsum(
            "sba,ai,bj->sji",
            dXdxi_poly,
            knots_xi[np.newaxis, :] ** np.arange(ngllxi - 1)[:, np.newaxis],
            knots_ga[np.newaxis, :] ** np.arange(ngllga)[:, np.newaxis],
        )
        dX_local[:, :, :, 1] = np.einsum(
            "sba,ai,bj->sji",
            dXdga_poly,
            knots_xi[np.newaxis, :] ** np.arange(ngllxi)[:, np.newaxis],
            knots_ga[np.newaxis, :] ** np.arange(ngllga - 1)[:, np.newaxis],
        )

        displacement_elastic = np.full((nspec, ngllga, ngllxi, 2), np.nan)
        displacement_elastic_ddot = np.full((nspec, ngllga, ngllxi, 2), np.nan)
        if nglob_elastic > 0:
            displacement_elastic[...] = np.where(
                medium_inds[..., elastic_medium_id, np.newaxis] == -1,
                np.nan,
                data_simfield["elastic_field"][medium_inds[..., elastic_medium_id], :],
            )
            displacement_elastic_ddot[...] = np.where(
                medium_inds[..., elastic_medium_id, np.newaxis] == -1,
                np.nan,
                data_simfield["elastic_field_ddot"][
                    medium_inds[..., elastic_medium_id], :
                ],
            )

        # displacement = np.empty((nspec, ngllga, ngllxi,2))
        # displacement_ddot = np.empty((nspec, ngllga, ngllxi,2))
        # displacement[self.is_fully_medium[:,elastic_medium_id],...] = displacement_elastic[self.is_fully_medium[:,elastic_medium_id],...]
        # displacement_ddot[self.is_fully_medium[:,elastic_medium_id],...] = displacement_elastic_ddot[self.is_fully_medium[:,elastic_medium_id],...]

        # displacement[self.is_fully_medium[:,acoustic_medium_id],...] = displacement_elastic[self.is_fully_medium[:,acoustic_medium_id],...]
        # displacement_ddot[self.is_fully_medium[:,acoustic_medium_id],...] = displacement_elastic_ddot[self.is_fully_medium[:,acoustic_medium_id],...]

        assert ngllga == ngllxi, "we are only supporting when ngllx==ngllz"
        self.nspec = nspec
        self.ngllx = ngllxi
        self.ngllz = ngllga
        self.X = chi
        self.pts = pts
        self.P = pts
        self.Xddot = chi_ddot
        self.displacement_elastic = displacement_elastic
        self.displacement_elastic_ddot = displacement_elastic_ddot
        self.mass_inverse_acoustic = mass_inv_acoustic
        self.mass_inverse_elastic = mass_inv_elastic

        self.ngll_capacity = ngll_capacity

        # V = L_a(x)L_b(y)
        # dV/dx = L_a'(x) L_b(y),   dV/dy = L_a(x) L_b'(y)
        # dV/dx(x_i,x_j) = {b==j} L_a'(x_i)    dV/dy(x_i,x_j) = {a==i} L_b'(x_j)
        dV_local = np.zeros((ngllxi, ngllga, ngllxi, ngllga, 2))
        dV_local[:, np.arange(ngllga), :, np.arange(ngllga), 0] = (
            self.GLL.Lp_at_knots_x.T
        )  # [shape_ind, pos_ind].T
        dV_local[np.arange(ngllxi), :, np.arange(ngllxi), :, 1] = (
            self.GLL.Lp_at_knots_z.T
        )  # [shape_ind, pos_ind].T

        dV = np.einsum("sjilg,ijabl->sjibag", dP_inv, dV_local)
        dV_local = np.transpose(dV_local, (1, 0, 3, 2, 4))[np.newaxis, ...]

        self.fields = types.SimpleNamespace(X=chi, pts=pts)
        self.differentials = types.SimpleNamespace(
            dP=dP,
            dP_inv=dP_inv,
            l=types.SimpleNamespace(dX=dX_local, dV=dV_local),
            g=types.SimpleNamespace(
                dX=np.einsum("sjilg,sjil->sjig", dP_inv, dX_local), dV=dV
            ),
        )

        self.displacement = np.where(
            self.medium_type[:, None, None, None] == 1,
            displacement_elastic,
            self.differentials.g.dX,
        )

        self.data_polys = types.SimpleNamespace(P_poly=P_poly, X_poly=chi_poly)
        self.cell_centers = np.mean(pts, axis=(1, 2))

        lci_parts = types.SimpleNamespace()
        to_add_media = []
        to_add_interfaces = []
        if "edge_type_refs" in data_simfield:
            edge_refs = np.empty(
                (np.max(data_simfield["edge_type_refs"]) + 1,), dtype=int
            )
            for i, v in enumerate(data_simfield["edge_type_refs"]):
                edge_refs[v] = i
            if "acoustic_acoustic_ispecs" in data_simfield:
                lci_parts.FF = types.SimpleNamespace(
                    medium=types.SimpleNamespace(),
                    interface=types.SimpleNamespace(),
                )
                lci_parts.FF.medium.ispec = data_simfield["acoustic_acoustic_ispecs"]
                lci_parts.FF.medium.edge_type = edge_refs[
                    data_simfield["acoustic_acoustic_edgetypes"]
                ]
                to_add_media.append(lci_parts.FF.medium)
                to_add_interfaces.append(lci_parts.FF.interface)
                lci_parts.FF.interface.label_prefix = "acoustic_acoustic"

                lci_parts.FF.medium.normal = data_simfield["acoustic_acoustic_normal"]
                lci_parts.FF.medium.contravariant_normal = data_simfield[
                    "acoustic_acoustic_contranormal"
                ]
                lci_parts.FF.interface.relax_param = data_simfield[
                    "acoustic_acoustic_relaxparam"
                ]
                lci_parts.FF.interface.medium1_dLn = data_simfield[
                    "acoustic_acoustic_dLn1"
                ]
                lci_parts.FF.interface.medium2_dLn = data_simfield[
                    "acoustic_acoustic_dLn2"
                ]

            if "acoustic_elastic_ispecs1" in data_simfield:
                lci_parts.FS = types.SimpleNamespace(
                    medium1=types.SimpleNamespace(),
                    medium2=types.SimpleNamespace(),
                    interface=types.SimpleNamespace(),
                )
                lci_parts.FS.medium1.ispec = data_simfield["acoustic_elastic_ispecs1"]
                lci_parts.FS.medium2.ispec = data_simfield["acoustic_elastic_ispecs2"]
                lci_parts.FS.medium1.edge_type = edge_refs[
                    data_simfield["acoustic_elastic_edgetypes1"]
                ]
                lci_parts.FS.medium2.edge_type = edge_refs[
                    data_simfield["acoustic_elastic_edgetypes2"]
                ]
                to_add_media.append(lci_parts.FS.medium1)
                to_add_media.append(lci_parts.FS.medium2)
                to_add_interfaces.append(lci_parts.FS.interface)
                lci_parts.FS.interface.label_prefix = "acoustic_elastic"

                lci_parts.FS.medium2.normal = data_simfield["acoustic_elastic_normal"]
            if "elastic_elastic_ispecs" in data_simfield:
                lci_parts.SS = types.SimpleNamespace(
                    medium=types.SimpleNamespace(),
                    interface=types.SimpleNamespace(),
                )
                lci_parts.SS.medium.ispec = data_simfield["elastic_elastic_ispecs"]
                lci_parts.SS.medium.edge_type = edge_refs[
                    data_simfield["elastic_elastic_edgetypes"]
                ]
                to_add_media.append(lci_parts.SS.medium)
                to_add_interfaces.append(lci_parts.SS.interface)
                lci_parts.SS.interface.label_prefix = "elastic_elastic"

            for ns in to_add_media:
                ns.ngll = ngllxi
                ns.global_field_at_edge = lambda field, ns=ns: recover_edgevals(
                    field[ns.ispec, ...],
                    ns.edge_type[:, *[np.newaxis for _ in field.shape[2:]]],
                )

                def interp_edge(efield, t, ns):
                    if not isinstance(t, np.ndarray):
                        t = np.array(t)
                    fieldinds = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: (len(efield.shape) - 2)]
                    return np.einsum(
                        f"ei{fieldinds},ik,...k->...e{fieldinds}",
                        efield,
                        self.GLL.L[ns.ngll, :, :],
                        t[..., np.newaxis] ** np.arange(ngll_capacity),
                    )

                ns.interpolate_edge_field = lambda efield, t, ns=ns: interp_edge(
                    efield, t, ns
                )
                ns.pts = ns.global_field_at_edge(self.pts)

            for ns in to_add_interfaces:
                ns.medium1_ind = data_simfield[f"{ns.label_prefix}_interface_inds1"]
                ns.medium2_ind = data_simfield[f"{ns.label_prefix}_interface_inds2"]
                ns.medium1_param_start = data_simfield[
                    f"{ns.label_prefix}_interface_paramstart1"
                ]
                ns.medium2_param_start = data_simfield[
                    f"{ns.label_prefix}_interface_paramstart2"
                ]
                ns.medium1_param_end = data_simfield[
                    f"{ns.label_prefix}_interface_paramend1"
                ]
                ns.medium2_param_end = data_simfield[
                    f"{ns.label_prefix}_interface_paramend2"
                ]
                ns.medium1_mortar_trans = data_simfield[
                    f"{ns.label_prefix}_interface_mortartrans1"
                ]
                ns.medium2_mortar_trans = data_simfield[
                    f"{ns.label_prefix}_interface_mortartrans2"
                ]
                ns.Jw = data_simfield[f"{ns.label_prefix}_interface_jw"]
                ns.size = ns.medium1_ind.shape[0]
                ns.nquad = ns.medium1_mortar_trans.shape[1]
                ns.nedge1quad = ns.medium1_mortar_trans.shape[2]
                ns.nedge2quad = ns.medium2_mortar_trans.shape[2]
                ns.edge1_to_mortar = lambda efield, ns=ns: np.einsum(
                    "ei...,eki->ek...",
                    efield[ns.medium1_ind, ...],
                    ns.medium1_mortar_trans,
                )
                ns.edge2_to_mortar = lambda efield, ns=ns: np.einsum(
                    "ei...,eki->ek...",
                    efield[ns.medium2_ind, ...],
                    ns.medium2_mortar_trans,
                )
                ns.edge1_to_param_start = lambda efield, ns=ns: np.einsum(
                    "ei...,ik,ek->e...",
                    efield[ns.medium1_ind, ...],
                    self.GLL.L[ns.nedge1quad, :, :],
                    ns.medium1_param_start[:, np.newaxis] ** np.arange(ngll_capacity),
                )
                ns.edge1_to_param_end = lambda efield, ns=ns: np.einsum(
                    "ei...,ik,ek->e...",
                    efield[ns.medium1_ind, ...],
                    self.GLL.L[ns.nedge1quad, :, :],
                    ns.medium1_param_end[:, np.newaxis] ** np.arange(ngll_capacity),
                )
                ns.edge2_to_param_start = lambda efield, ns=ns: np.einsum(
                    "ei...,ik,ek->e...",
                    efield[ns.medium2_ind, ...],
                    self.GLL.L[ns.nedge2quad, :, :],
                    ns.medium2_param_start[:, np.newaxis] ** np.arange(ngll_capacity),
                )
                ns.edge2_to_param_end = lambda efield, ns=ns: np.einsum(
                    "ei...,ik,ek->e...",
                    efield[ns.medium2_ind, ...],
                    self.GLL.L[ns.nedge2quad, :, :],
                    ns.medium2_param_end[:, np.newaxis] ** np.arange(ngll_capacity),
                )

        self.LCI = lci_parts

        if data_edges is not None:
            self.data_capacity = data_edges["edge_data"].shape[1]

            self.edges = types.SimpleNamespace(
                count=data_edges["edge_intdat"].shape[0],
                specID=data_edges["edge_intdat"][:, 0],
                edgeID=data_edges["edge_intdat"][:, 1],
                ngll=data_edges["edge_intdat"][:, 2],
                pos=data_edges["edge_pos"],
                data=data_edges["edge_data"],
                data_inds=types.SimpleNamespace(
                    NX=0,
                    NZ=1,
                    DET=2,
                    DS=3,
                    FIELD=4,
                    FIELDNDERIV=6,
                    SPEEDPARAM=8,
                    SHAPENDERIV=slice(9, 9 + ngll_capacity),
                ),
            )

            self.edges.pos_polys = np.einsum(
                "eid,eik->ekd", self.edges.pos, self.GLL.L[self.edges.ngll, :, :]
            )

            def edges_interp_pos(t, e_inds=None):
                if not isinstance(t, np.ndarray):
                    t = np.array(t)
                if e_inds is None:
                    return np.einsum(
                        "eid,eik,...k->...ed",
                        self.edges.pos,
                        self.GLL.L[self.edges.ngll, :, :],
                        t[..., np.newaxis] ** np.arange(ngll_capacity),
                    )
                else:
                    return np.einsum(
                        "...id,...ik,...k->...d",
                        self.edges.pos[e_inds, ...],
                        self.GLL.L[self.edges.ngll[e_inds], :, :],
                        t[..., np.newaxis] ** np.arange(ngll_capacity),
                    )

            self.edges.interp_pos = edges_interp_pos

            self.edges.normals = np.where(
                self.edges.edgeID[:, np.newaxis, np.newaxis] == 0,
                dP[self.edges.specID, :, -1, :, 1],
                np.where(
                    self.edges.edgeID[:, np.newaxis, np.newaxis] == 1,
                    -dP[self.edges.specID, -1, :, :, 0],
                    np.where(
                        self.edges.edgeID[:, np.newaxis, np.newaxis] == 2,
                        -dP[self.edges.specID, :, 0, :, 1],
                        dP[self.edges.specID, 0, :, :, 0],
                    ),
                ),
            )  # 90 degrees off from normals
            self.edges.tangentials = np.where(
                self.edges.edgeID[:, np.newaxis, np.newaxis] == 0,
                dP[self.edges.specID, :, -1, :, 1],
                np.where(
                    self.edges.edgeID[:, np.newaxis, np.newaxis] == 1,
                    dP[self.edges.specID, -1, :, :, 0],
                    np.where(
                        self.edges.edgeID[:, np.newaxis, np.newaxis] == 2,
                        dP[self.edges.specID, :, 0, :, 1],
                        dP[self.edges.specID, 0, :, :, 0],
                    ),
                ),
            )
            self.edges.normals = np.einsum(
                "ij,...j->...i", np.array([[0, 1], [-1, 0]]), self.edges.normals
            )
            self.edges.normals /= np.linalg.norm(self.edges.normals, ord=2, axis=-1)[
                :, :, np.newaxis
            ]

            self.intersections = types.SimpleNamespace(
                count=data_edges["intersect_intdat"].shape[0],
                a_ref_ind=data_edges["intersect_intdat"][:, 0],
                b_ref_ind=data_edges["intersect_intdat"][:, 1],
                ngll=data_edges["intersect_intdat"][:, 2],
                a_ngll=data_edges["intersect_intdat"][:, 3],
                b_ngll=data_edges["intersect_intdat"][:, 4],
                a_param_start=data_edges["intersect_floatdat"][:, 0],
                a_param_end=data_edges["intersect_floatdat"][:, 1],
                b_param_start=data_edges["intersect_floatdat"][:, 2],
                b_param_end=data_edges["intersect_floatdat"][:, 3],
                relax_param=data_edges["intersect_floatdat"][:, 4],
                a_mortar_trans=data_edges["intersect_mortartrans"][:, 0, ...],
                b_mortar_trans=data_edges["intersect_mortartrans"][:, 1, ...],
                data=data_edges["intersect_data"],
                data_inds=types.SimpleNamespace(
                    FLUX_TOTAL_A=slice(ngll_capacity * 0, ngll_capacity * (0 + 1)),
                    FLUX1_A=slice(ngll_capacity * 1, ngll_capacity * (1 + 1)),
                    FLUX2_A=slice(ngll_capacity * 2, ngll_capacity * (2 + 1)),
                    FLUX3_A=slice(ngll_capacity * 3, ngll_capacity * (3 + 1)),
                    FLUX_TOTAL_B=slice(ngll_capacity * 4, ngll_capacity * (4 + 1)),
                    FLUX1_B=slice(ngll_capacity * 5, ngll_capacity * (5 + 1)),
                    FLUX2_B=slice(ngll_capacity * 6, ngll_capacity * (6 + 1)),
                    FLUX3_B=slice(ngll_capacity * 7, ngll_capacity * (7 + 1)),
                    UJMP=slice(ngll_capacity * 8, ngll_capacity * (8 + 1)),
                    CDU_AVG=slice(ngll_capacity * 9, ngll_capacity * (9 + 1)),
                    IS_ON_BDRY_A=slice(ngll_capacity * 10, ngll_capacity * (10 + 1)),
                    IS_ON_BDRY_B=slice(ngll_capacity * 11, ngll_capacity * (11 + 1)),
                ),
            )
            self.edges.net_fluxes = np.zeros((self.edges.count, ngll_capacity))
            self.edges.net_fluxes[self.intersections.a_ref_ind, :] += (
                self.intersections.data[:, self.intersections.data_inds.FLUX_TOTAL_A]
            )
            self.edges.net_fluxes[self.intersections.b_ref_ind, :] += (
                self.intersections.data[:, self.intersections.data_inds.FLUX_TOTAL_B]
            )

            # since the knots are from GLL (known), we can recover mortar_trans parameters: x_iL_i = id, for knots x_i
            self.intersections.a_mortar_trans_knots = np.einsum(
                "ei,eki->ek",
                self.GLL.knots[self.intersections.a_ngll, ...],
                self.intersections.a_mortar_trans,
            )
            self.intersections.b_mortar_trans_knots = np.einsum(
                "ei,eki->ek",
                self.GLL.knots[self.intersections.b_ngll, ...],
                self.intersections.b_mortar_trans,
            )

            self.mortar_data = types.SimpleNamespace(
                a=np.einsum(
                    "sdp,smp->sdm",
                    self.edges.data[self.intersections.a_ref_ind, ...],
                    self.intersections.a_mortar_trans,
                ),
                b=np.einsum(
                    "sdp,smp->sdm",
                    self.edges.data[self.intersections.b_ref_ind, ...],
                    self.intersections.b_mortar_trans,
                ),
            )

    def edge_vals_of_fields(self, field):
        if hasattr(self, "edges"):
            return recover_edgevals(
                field[self.edges.specID, ...],
                self.edges.edgeID[:, *[np.newaxis for _ in field.shape[2:]]],
            )
        return None

    def plot_field(
        self,
        field,
        plt_cell_margin=0.2,
        draw_cell_borders=True,
        figsize=None,
        title=None,
        show=True,
        mode: Literal["scatter", "contour"] = "scatter",
        ptsize=None,
        vmin=None,
        vmax=None,
        current_axes=None,
    ):
        pt_centers = self.cell_centers[:, np.newaxis, np.newaxis, :]
        pts_plt = (1 - plt_cell_margin) * self.pts + plt_cell_margin * pt_centers
        if figsize is not None and current_axes is None:
            plt.figure(figsize=figsize)
        active = plt if current_axes is None else current_axes
        if draw_cell_borders:
            active.plot(self.pts[:, :, -1, 0].T, self.pts[:, :, -1, 1].T, ":k")
            active.plot(self.pts[:, :, 0, 0].T, self.pts[:, :, 0, 1].T, ":k")
            active.plot(self.pts[:, -1, :, 0].T, self.pts[:, -1, :, 1].T, ":k")
            active.plot(self.pts[:, -1, :, 0].T, self.pts[:, 0, :, 1].T, ":k")
        if mode == "scatter":
            active.scatter(
                pts_plt[..., 0], pts_plt[..., 1], ptsize, field, vmin=vmin, vmax=vmax
            )
        elif mode == "contour":
            if not hasattr(self, "full_domain_triangulation"):
                c = 1e-4  # offset triangulation to remove degenerate triangles
                pts_mesh = ((1 - c) * self.pts + c * pt_centers).reshape(
                    self.nspec * self.ngllz * self.ngllx, 2
                )
                self.full_domain_triangulation = mpltri.Triangulation(
                    pts_mesh[:, 0], pts_mesh[:, 1]
                )
            tri = self.full_domain_triangulation
            try:
                active.tricontourf(
                    tri,
                    field.reshape((self.nspec * self.ngllz * self.ngllx,)),
                    100,
                    vmin=vmin,
                    vmax=vmax,
                )
            except ValueError:
                # probably because we have NaN or inf values
                global_mean = np.mean(pt_centers, axis=(0, 1, 2))
                active.text(
                    global_mean[0],
                    global_mean[1],
                    "non-finite value;\n cannot contour.",
                    color="red",
                )
        if title is not None:
            if current_axes is None:
                plt.title(title)
            else:
                current_axes.set_title(title)
        if show:
            plt.show()

    def plot_edge_field(
        self, field, plt_cell_margin=0.5, figsize=None, title=None, show=True
    ):
        pt_centers = self.cell_centers[:, np.newaxis, np.newaxis, :]
        pts_plt = (1 - plt_cell_margin) * self.pts + plt_cell_margin * pt_centers
        pts_plt_edge = self.edge_vals_of_fields(pts_plt)
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.scatter(pts_plt[..., 0], pts_plt[..., 1], 1)
        if pts_plt_edge is not None:
            plt.scatter(pts_plt_edge[..., 0], pts_plt_edge[..., 1], None, field)
        if title is not None:
            plt.title(title)
        if show:
            plt.show()

    def plot_assembly(self, plt_cell_margin=0.5, figsize=None, title=None, show=True):
        pt_centers = self.cell_centers[:, np.newaxis, np.newaxis, :]
        pts_plt = (1 - plt_cell_margin) * self.pts + plt_cell_margin * pt_centers
        if figsize is not None:
            plt.figure(figsize=figsize)
        for iglob in range(self.nglob):
            inds = np.where(self.data_simfield["index_mapping"] == iglob)
            poly = mplpatches.Polygon(pts_plt[*inds, :], edgecolor="k")
            plt.gca().add_patch(poly)
        plt.scatter(pts_plt[..., 0], pts_plt[..., 1], 1)
        if title is not None:
            plt.title(title)
        if show:
            plt.show()

    def plot_LCI(self, plt_cell_margin=0.5, figsize=None, title=None, show=True):
        pt_centers = self.cell_centers[:, np.newaxis, np.newaxis, :]
        pts_plt = (1 - plt_cell_margin) * self.pts + plt_cell_margin * pt_centers
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.scatter(pts_plt[..., 0], pts_plt[..., 1], 1)
        plt.scatter(self.LCI.FF.medium.pts[..., 0], self.LCI.FF.medium.pts[..., 1], 1)
        pts_inter = self.LCI.FF.interface.edge1_to_mortar(self.LCI.FF.medium.pts)
        plt.plot(pts_inter[:, :, 0].T, pts_inter[:, :, 1].T, ":xr")
        pts_inter = self.LCI.FS.interface.edge1_to_mortar(self.LCI.FS.medium1.pts)
        plt.plot(pts_inter[:, :, 0].T, pts_inter[:, :, 1].T, ":xg")
        pts_inter = self.LCI.SS.interface.edge1_to_mortar(self.LCI.SS.medium.pts)
        plt.plot(pts_inter[:, :, 0].T, pts_inter[:, :, 1].T, ":xb")
        if title is not None:
            plt.title(title)
        if show:
            plt.show()


class dump_series:
    def __init__(self):
        self.statics_defined = False
        self.gll_defined = False
        self.statics = dict()
        self.framedata = dict()
        self.size_timeseries = -1
        self.time_indices = np.array((), dtype=int)
        self.GLL = None

    def set_statics(self, statics: dict[str, np.ndarray]):
        self.statics_defined = True
        self.statics = statics
        for k, v in statics.items():
            self._guess_gll(k, v)

    def init_per_frame_data(self, size_timeseries: int):
        self.size_timeseries = size_timeseries
        self.time_indices = np.empty(size_timeseries, dtype=int)
        self.framedata = dict()

    def _guess_gll(self, dataname, array):
        if not self.gll_defined:
            if dataname == "pts":
                gllx = array.shape[3]
                gllz = array.shape[2]
            elif dataname == "index_mapping":
                gllx = array.shape[2]
                gllz = array.shape[1]
            else:
                return
            self.GLL = build_gll_namespace(gllx, gllz)
            self.gll_defined = True

    def set_frame(
        self, index_timeseries: int, index_dumpnum: int, data: dict[str, np.ndarray]
    ):
        if self.size_timeseries < index_timeseries or index_timeseries < 0:
            raise ValueError(
                f"index {index_timeseries} is out of bounds for series of size {self.size_timeseries}"
            )
        self.time_indices[index_timeseries] = index_dumpnum
        for key, value in data.items():
            if key not in self.framedata:
                self.framedata[key] = np.empty(
                    (self.size_timeseries,) + value.shape, dtype=value.dtype
                )
            self.framedata[key][index_timeseries, ...] = value

    def get_frame_as_dump_frame(self, ind: int):
        return dump_frame(
            self.statics | {k: v[ind, ...] for k, v in self.framedata.items()}
        )

    def save_to_file(self, filename):
        data = (
            {"statics." + k: v for k, v in self.statics.items()}
            | {"framedata." + k: v for k, v in self.framedata.items()}
            | {"timeseries": self.time_indices}
        )
        write_sfdump(filename, data)

    def get_subseries(self, indices):
        sub = dump_series()
        sub.statics_defined = self.statics_defined
        sub.gll_defined = self.gll_defined
        sub.statics = self.statics
        sub.GLL = self.GLL

        sub.framedata = {k: v[indices, ...] for k, v in self.framedata.items()}
        sub.time_indices = self.time_indices[indices]
        sub.size_timeseries = len(indices)

        return sub

    @staticmethod
    def load_from_file(filename):
        data = read_sfdump(filename)
        statics = dict()
        framedata = dict()
        timeseries = None
        for k, v in data.items():
            if k.startswith("statics."):
                statics[k.replace("statics.", "")] = v
                continue
            if k.startswith("framedata."):
                framedata[k.replace("framedata.", "")] = v
                continue
            if k == "timeseries":
                timeseries = v

        if timeseries is None:
            raise ValueError("File has no time series!")

        series = dump_series()
        if len(statics) > 0:
            series.set_statics(statics)

        series.size_timeseries = len(timeseries)
        series.time_indices = timeseries
        series.framedata = framedata

        return series


def read_dump_file(
    simfield_file: str, edges_file: str | None = None, statics_data=None
):
    assert simfield_file.endswith(".dat")
    if edges_file is not None:
        assert edges_file.endswith(".dat")
    data_simfield = read_sfdump(simfield_file)
    suffix = re.search(r"(\d+)\.dat", simfield_file)
    assert suffix is not None

    if statics_data is None:
        prefix = simfield_file[: suffix.start()]
        statics_file = prefix + "statics.dat"
        if os.path.exists(statics_file):
            statics_data = read_sfdump(statics_file)
            data_simfield.update(statics_data)
    else:
        data_simfield.update(statics_data)

    if edges_file is not None:
        data_edges = read_sfdump(edges_file)
    else:
        data_edges = None
    return dump_frame(data_simfield, data_edges)


def load_series(simfield_dump_prefix: str):
    statics = None
    if os.path.exists(simfield_dump_prefix + "statics.dat"):
        statics = read_sfdump(simfield_dump_prefix + "statics.dat")

    data_frames = dict()
    shapes = dict()
    dtypes = dict()
    folder = os.path.dirname(simfield_dump_prefix)
    files = os.listdir(folder)
    for file in files:
        m = re.match(os.path.basename(simfield_dump_prefix) + r"(\d+).dat", file)
        if m:
            data = read_sfdump(os.path.join(folder, file))
            for key in list(data.keys()):
                if key in statics:
                    del data[key]
                    continue
                if key in shapes:
                    assert shapes[key] == data[key].shape
                else:
                    shapes[key] = data[key].shape

                if key in dtypes:
                    assert dtypes[key] == data[key].dtype
                else:
                    dtypes[key] = data[key].dtype

            data_frames[int(m.group(1))] = data
    num_frames = len(data_frames)

    out = dump_series()
    if statics is not None:
        out.set_statics(statics)
    out.init_per_frame_data(num_frames)

    for i, k in enumerate(data_frames.keys()):
        out.set_frame(i, k, data_frames[k])

    return out


if __name__ == "__main__":
    pass
    from . import config

    test = config.get("cg_compare.tests.2")
    folder = os.path.join(config.get("cg_compare.workspace_folder"), test["name"])

    d = read_dump_file(
        os.path.join(folder, config.get("cg_compare.workspace_files.dump_prefix"))
        + "0.dat"
    )
    d.plot_LCI()
    # d.plot_field(np.linalg.norm(d.displacement, axis=-1), mode="scatter")

    # ser = load_series(os.path.join(folder, config.get("cg_compare.workspace_files.dump_prefix")))

    # if ser.statics_defined:
    #     print("statics")
    #     for i,a in ser.statics.items():
    #         print(f"  {i}: {a.shape}")

    # print("framedata")
    # for i,a in ser.framedata.items():
    #     print(f"  {i}({a.dtype}): {a.shape}")

    # ser.save_to_file(os.path.join(folder, config.get("cg_compare.workspace_files.provenance_dump")))

    # ser2 = dump_series.load_from_file(os.path.join(folder, config.get("cg_compare.workspace_files.provenance_dump")))

    # if ser2.statics_defined:
    #     print("statics")
    #     for i,a in ser2.statics.items():
    #         print(f"  {i}: {a.shape}")

    # print("framedata")
    # for i,a in ser2.framedata.items():
    #     print(f"  {i}({a.dtype}): {a.shape}")
