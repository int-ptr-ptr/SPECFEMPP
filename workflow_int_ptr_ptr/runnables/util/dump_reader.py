import numpy as np
import matplotlib.pyplot as plt
import re
import struct
import types

from . import GLL


def read_sfdump(fname):
    types = {
        "i": (int, lambda buffer: struct.iter_unpack("i", buffer)),
        "f": (float, lambda buffer: struct.iter_unpack("f", buffer)),
    }
    data = dict()
    with open(fname, "rb") as f:
        header = b""
        while byte := f.read(1):
            header += byte
            if byte == b"]":
                # found header. Match?
                m = re.search(
                    r"(\w+)<(\w+)\(size=(\d+)B\)>\[([\d,]+)\]", header.decode("ascii")
                )
                if m is None:
                    continue
                name = m.group(1)
                datatype = m.group(2)
                typesize = int(m.group(3))
                datashape = [int(s) for s in m.group(4).split(",")]
                blocksize = np.prod(datashape) * typesize

                # use data type string to figure out what to do
                if datatype not in types:
                    raise Exception(f"type '{datatype}' has no set behavior!")

                # 1d -> nd mapping to populate arr
                def recover_inds(i):
                    inds = [None] * len(datashape)
                    for place in range(len(datashape) - 1, -1, -1):
                        inds[place] = i % datashape[place]
                        i //= datashape[place]
                    return inds

                # loading data into np array
                arr = np.empty(datashape, dtype=types[datatype][0])
                decodefunc = types[datatype][1]
                buf = bytearray(blocksize)
                f.readinto(buf)

                for i, v in enumerate(decodefunc(buf)):
                    arr[*recover_inds(i)] = v[0]

                data[name] = arr
                header = b""

    return data


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


class dump_reader:
    def __init__(self, data_simfield, data_edges):
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

        chi = np.empty((nspec, ngllga, ngllxi))
        chi[...] = np.where(
            medium_inds[..., acoustic_medium_id] == -1,
            np.nan,
            data_simfield["acoustic_field"][medium_inds[..., acoustic_medium_id], 0],
        )
        chi_ddot = np.empty((nspec, ngllga, ngllxi))
        chi_ddot[...] = np.where(
            medium_inds[..., acoustic_medium_id] == -1,
            np.nan,
            data_simfield["acoustic_field_ddot"][
                medium_inds[..., acoustic_medium_id], 0
            ],
        )
        mass_inv_acoustic = np.empty((nspec, ngllga, ngllxi))
        mass_inv_acoustic[...] = np.where(
            medium_inds[..., acoustic_medium_id] == -1,
            np.nan,
            data_simfield["acoustic_mass_inverse"][
                medium_inds[..., acoustic_medium_id], 0
            ],
        )

        mass_inv_elastic = np.empty((nspec, ngllga, ngllxi))
        mass_inv_elastic[...] = np.where(
            medium_inds[..., elastic_medium_id] == -1,
            np.nan,
            data_simfield["elastic_mass_inverse"][
                medium_inds[..., elastic_medium_id], 0
            ],
        )

        pts = data_simfield["pts"].transpose((1, 2, 3, 0))

        knots_xi = GLL.get_knots(ngllxi - 1)
        knots_ga = GLL.get_knots(ngllga - 1)

        Lxi = np.array(GLL.build_lagrange_polys(ngllxi - 1))
        Lxi_p = np.array(GLL.polyderiv(Lxi.T)).T
        Lga = np.array(GLL.build_lagrange_polys(ngllga - 1))
        Lga_p = np.array(GLL.polyderiv(Lga.T)).T

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

        displacement_elastic = np.empty((nspec, ngllga, ngllxi, 2))
        displacement_elastic_ddot = np.empty((nspec, ngllga, ngllxi, 2))
        displacement_elastic[...] = np.where(
            medium_inds[..., elastic_medium_id, np.newaxis] == -1,
            np.nan,
            data_simfield["elastic_field"][medium_inds[..., elastic_medium_id], :],
        )
        displacement_elastic_ddot[...] = np.where(
            medium_inds[..., elastic_medium_id, np.newaxis] == -1,
            np.nan,
            data_simfield["elastic_field_ddot"][medium_inds[..., elastic_medium_id], :],
        )
        # displacement = np.empty((nspec, ngllga, ngllxi,2))
        # displacement_ddot = np.empty((nspec, ngllga, ngllxi,2))
        # displacement[self.is_fully_medium[:,elastic_medium_id],...] = displacement_elastic[self.is_fully_medium[:,elastic_medium_id],...]
        # displacement_ddot[self.is_fully_medium[:,elastic_medium_id],...] = displacement_elastic_ddot[self.is_fully_medium[:,elastic_medium_id],...]

        # displacement[self.is_fully_medium[:,acoustic_medium_id],...] = displacement_elastic[self.is_fully_medium[:,acoustic_medium_id],...]
        # displacement_ddot[self.is_fully_medium[:,acoustic_medium_id],...] = displacement_elastic_ddot[self.is_fully_medium[:,acoustic_medium_id],...]

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

        ngll_capacity = data_edges["edge_pos"].shape[1]
        self.ngll_capacity = ngll_capacity

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

        self.GLL = types.SimpleNamespace(
            ngll_x=ngllxi,
            ngll_z=ngllga,
            knots_x=knots_xi,
            knots_z=knots_ga,
            L_x=Lxi,
            Lp_x=Lxi_p,
            L_z=Lga,
            Lp_z=Lga_p,
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

        self.data_polys = types.SimpleNamespace(P_poly=P_poly, X_poly=chi_poly)
        self.cell_centers = np.mean(pts, axis=(1, 2))

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
        return recover_edgevals(
            field[self.edges.specID, ...],
            self.edges.edgeID[:, *[np.newaxis for _ in field.shape[2:]]],
        )

    def plot_field(
        self,
        field,
        plt_cell_margin=0.2,
        draw_cell_borders=True,
        figsize=None,
        title=None,
        show=True,
    ):
        pt_centers = self.cell_centers[:, np.newaxis, np.newaxis, :]
        pts_plt = (1 - plt_cell_margin) * self.pts + plt_cell_margin * pt_centers
        if figsize is not None:
            plt.figure(figsize=figsize)
        if draw_cell_borders:
            plt.plot(self.pts[:, :, -1, 0].T, self.pts[:, :, -1, 1].T, ":k")
            plt.plot(self.pts[:, :, 0, 0].T, self.pts[:, :, 0, 1].T, ":k")
            plt.plot(self.pts[:, -1, :, 0].T, self.pts[:, -1, :, 1].T, ":k")
            plt.plot(self.pts[:, -1, :, 0].T, self.pts[:, 0, :, 1].T, ":k")
        plt.scatter(pts_plt[..., 0], pts_plt[..., 1], None, field)
        if title is not None:
            plt.title(title)
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
        plt.scatter(pts_plt_edge[..., 0], pts_plt_edge[..., 1], None, field)
        if title is not None:
            plt.title(title)
        if show:
            plt.show()


def read_dump_file(simfield_file: str, edges_file: str):
    data_simfield = read_sfdump(simfield_file)
    data_edges = read_sfdump(edges_file)
    return dump_reader(data_simfield, data_edges)
