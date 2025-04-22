from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from . import symgll


class GLLStore:
    gll_num: dict[int, symgll.GLL]
    gll_sym: dict[int, symgll.GLL]

    def __init__(self):
        self.gll_num = {}
        self.gll_sym = {}

    def get_gll(self, degree: int, symbolic=False) -> symgll.GLL:
        arr = self.gll_sym if symbolic else self.gll_num

        if degree not in arr:
            arr[degree] = symgll.GLL(degree=degree, as_symbolic=symbolic)
        return arr[degree]


_gll = GLLStore().get_gll


class EdgeUtils:
    @staticmethod
    def inds_of_edge(edge: int, degree: int):
        flp = edge // 4
        edge = edge % 4
        if edge == 0:  # right
            inds = np.full((degree + 1), degree, dtype=int), np.arange(degree + 1)
        elif edge == 2:  # left
            inds = np.full((degree + 1), 0, dtype=int), np.arange(degree + 1)
        elif edge == 1:  # top
            inds = np.arange(degree + 1), np.full((degree + 1), degree, dtype=int)
        else:  # bottom
            inds = np.arange(degree + 1), np.full((degree + 1), 0, dtype=int)
        if flp:
            inds = np.flip(inds[0]), np.flip(inds[1])
        return inds


class Element2D:
    degree: int
    gll: symgll.GLL

    def __init__(self, degree: int, symbolic: bool = False):
        self.degree = degree
        self.gll = _gll(degree, symbolic)

    def mass_matrix(self, use_gll_quadrature: bool = True):
        dot_prods = (
            self.gll.quad_dot_products
            if use_gll_quadrature
            else self.gll.true_dot_products
        )
        return dot_prods[:, None, :, None] * dot_prods[None, :, None, :]

    def stiffness_matrix(self, use_gll_quadrature: bool = True):
        return -self.gll.stiffness2D(quadrature=use_gll_quadrature)

    def stiffness_normalderiv(
        self, edge: int, use_gll_quadrature: bool = True, integrate: bool = False
    ):
        """Returns (test=v,field=u) int ( v (grad u) . n ) dV  if integrate is true,
        or ( v (grad u) . n ) otherwise.

        returned is a 4-index:  (vx_ind, vy_ind, ux_ind, uy_ind)
        """
        if edge % 2 == 0:
            nxi = 1 if edge == 0 else -1
            nga = 0
        else:
            nxi = 0
            nga = 1 if edge == 1 else -1

        edgeinds = EdgeUtils.inds_of_edge(edge, self.degree)

        assert use_gll_quadrature, (
            "stiffness_normalderiv currently only supported with GLL"
        )

        # dn L{xin, yin} (edge) as [xin,yin,edge]
        edge_nderiv = np.zeros((self.degree + 1,) * 3, dtype=self.gll.get_type())
        # (L'x Ly) (edge) = L'x(edgex) delta(edgey,y)
        edge_nderiv[:, edgeinds[1], edgeinds[1]] = (
            self.gll.deriv_at_knots[:, edgeinds[0]] * nxi
        )
        # (Lx L'y) (edge) = L'y(edgey) delta(edgex,x)
        edge_nderiv[edgeinds[0], :, edgeinds[0]] += (
            self.gll.deriv_at_knots[:, edgeinds[1]].T * nga
        )

        mat = np.zeros((self.degree + 1,) * 4, dtype=self.gll.get_type())
        if integrate:
            mat[edgeinds[0], edgeinds[1], :, :] = np.einsum(
                "xye,e->exy", edge_nderiv, self.gll.weights
            )
        else:
            mat[edgeinds[0], edgeinds[1], :, :] = np.einsum("xye->exy", edge_nderiv)
        return mat

    def field_to_basis(self, field: Callable[[Any, Any], Any]):
        return np.array(
            [[field(x, y) for y in self.gll.knots] for x in self.gll.knots],
            dtype=self.gll.get_type(),
        )


class MortarTransfer:
    transfer_tensor: np.ndarray

    def __init__(
        self, mesh: "Mesh2D", start_parameter: float = -1, end_parameter: float = 1
    ):
        """Generates the transfer tensor

        Args:
            start_parameter (float, optional): location in edge coords for the start of the mortar. Defaults to -1.
            end_parameter (float, optional): ocation in edge coords for the end of the mortar. Defaults to 1.
        """
        gll = _gll(mesh.degree, mesh.symbolic)
        self.transfer_tensor = gll.interpolate_poly(
            start_parameter + (end_parameter - start_parameter) * (gll.knots + 1) / 2
        )


class BoundaryCondition(ABC):
    @abstractmethod
    def apply_stiffness(
        self, mesh: "Mesh2D", stiffmat: np.ndarray, use_gll_quadrature: bool = True
    ) -> None: ...

    def apply_stiffness_timederiv(
        self, mesh: "Mesh2D", stiffmat: np.ndarray, use_gll_quadrature: bool = True
    ) -> None: ...


class UpwindBC(BoundaryCondition):
    def __init__(
        self,
        elem1: int,
        side1: int,
        elem2: int | None = None,
        side2: int = 0,
        flip: bool = False,
    ):
        self.elem1 = elem1
        self.elem2 = elem2
        self.side1 = side1
        self.side2 = side2
        self.flip = flip

    def apply_stiffness(self, mesh, stiffmat, use_gll_quadrature: bool = True):
        elem1 = mesh.get_element(self.elem1)
        normal1 = elem1.stiffness_normalderiv(
            self.side1, use_gll_quadrature=use_gll_quadrature, integrate=True
        )
        normal1 /= 2
        gl1 = mesh.matrix_global_to_local_subview(self.elem1)

        if self.elem2 is not None:
            elem2 = mesh.get_element(self.elem2)
            normal2 = elem2.stiffness_normalderiv(
                self.side2, use_gll_quadrature=use_gll_quadrature, integrate=True
            )
            normal2 /= 2
            gl2 = mesh.matrix_global_to_local_subview(self.elem2)
            stiffmat[gl2] += normal2

            stiffmat[
                gl2[0][*EdgeUtils.inds_of_edge(self.side2, mesh.degree), :, :], gl1[1]
            ] -= normal1[*EdgeUtils.inds_of_edge(self.side1, mesh.degree), :, :]
            stiffmat[
                gl1[0][*EdgeUtils.inds_of_edge(self.side1, mesh.degree), :, :], gl2[1]
            ] -= normal2[*EdgeUtils.inds_of_edge(self.side2, mesh.degree), :, :]

        stiffmat[gl1] += normal1
        return

    def apply_stiffness_timederiv(
        self, mesh, stiffmat, use_gll_quadrature: bool = True
    ):
        elem1 = mesh.get_element(self.elem1)
        mass1 = elem1.mass_matrix(use_gll_quadrature=use_gll_quadrature)[
            *EdgeUtils.inds_of_edge(self.side1, mesh.degree), :, :
        ]
        mass1 /= -2
        gl1 = mesh.matrix_global_to_local_subview(self.elem1)

        if self.elem2 is not None:
            elem2 = mesh.get_element(self.elem2)
            mass2 = elem2.mass_matrix(use_gll_quadrature=use_gll_quadrature)[
                *EdgeUtils.inds_of_edge(self.side2, mesh.degree), :, :
            ]
            mass2 /= -2
            gl2 = mesh.matrix_global_to_local_subview(self.elem2)
            stiffmat[gl2] += mass2

            stiffmat[
                gl2[0][*EdgeUtils.inds_of_edge(self.side2, mesh.degree), :, :], gl1[1]
            ] += mass1
            stiffmat[
                gl1[0][*EdgeUtils.inds_of_edge(self.side1, mesh.degree), :, :], gl2[1]
            ] += mass2

        stiffmat[gl1] += mass1
        return


class MidpointBC(BoundaryCondition):
    def __init__(
        self,
        elem1: int,
        side1: int,
        elem2: int,
        side2: int,
        flip: bool = False,
        w: float = 1,
    ):
        self.elem1 = elem1
        self.elem2 = elem2
        self.side1 = side1
        self.side2 = side2
        self.flip = flip
        self.w = w

    def apply_stiffness(self, mesh, stiffmat, use_gll_quadrature: bool = True):
        elem1 = mesh.get_element(self.elem1)
        elem2 = mesh.get_element(self.elem2)
        gl1 = mesh.matrix_global_to_local_subview(self.elem1)
        gl2 = mesh.matrix_global_to_local_subview(self.elem2)
        gl1_edge = gl1[0][*EdgeUtils.inds_of_edge(self.side1, mesh.degree), :, :]
        gl2_edge = gl2[0][*EdgeUtils.inds_of_edge(self.side2, mesh.degree), :, :]

        normal1 = elem1.stiffness_normalderiv(
            self.side1, use_gll_quadrature=use_gll_quadrature, integrate=True
        )[*EdgeUtils.inds_of_edge(self.side1, mesh.degree), :, :]
        normal2 = elem2.stiffness_normalderiv(
            self.side2, use_gll_quadrature=use_gll_quadrature, integrate=True
        )[*EdgeUtils.inds_of_edge(self.side2, mesh.degree), :, :]

        stiffmat[gl1_edge, gl1[1]] += normal1 * (1 - self.w) / 2
        stiffmat[gl2_edge, gl2[1]] += normal2 * (1 - self.w) / 2

        normal2 /= 2

        stiffmat[gl2_edge, gl1[1]] -= normal1 * (1 + self.w) / 2
        stiffmat[gl1_edge, gl2[1]] -= normal2 * (1 + self.w) / 2

        return


class Mesh2D:
    symbolic: bool
    degree: int
    _elements: list[Element2D]
    _index_mapping: np.ndarray
    _index_mapping_outdated: bool
    nglob: int
    _connections_ind: np.ndarray
    _connections_type: np.ndarray
    _connections_outdated: bool

    boundary_conditions: list[BoundaryCondition]

    def __init__(self, degree: int, symbolic: bool):
        self.degree = degree
        self.symbolic = symbolic
        self._elements = list()
        self._connections_ind = np.empty((0, 4), dtype=int)
        self._connections_type = np.empty((0, 4), dtype=int)
        self._index_mapping_outdated = True
        self._connections_outdated = False
        self._update_index_mapping()
        self.boundary_conditions = list()

    def _update_connections(self):
        if not self._connections_outdated:
            return
        ind_old = self._connections_ind
        type_old = self._connections_type
        self._connections_ind = np.full((len(self._elements), 4), -1, dtype=int)
        self._connections_type = np.empty((len(self._elements), 4), dtype=int)

        nelem_old = ind_old.shape[0]
        self._connections_ind[:nelem_old, :] = ind_old
        self._connections_type[:nelem_old, :] = type_old
        self._connections_outdated = False

    def _update_index_mapping(self):
        self._update_connections()
        if not self._index_mapping_outdated:
            return
        self._index_mapping = np.full(
            (len(self._elements), self.degree + 1, self.degree + 1), -1, dtype=int
        )
        nglob = 0
        for ielem, element in enumerate(self._elements):
            # fails if element connects to itself
            nelem_to_add = np.count_nonzero(self._index_mapping[ielem, :, :] == -1)
            self._index_mapping[
                ielem, *np.where(self._index_mapping[ielem, :, :] == -1)
            ] = np.arange(nglob, nglob + nelem_to_add)
            nglob += nelem_to_add
            for iconn in range(4):
                adjind = self._connections_ind[ielem, iconn]
                if adjind == ielem:
                    raise NotImplementedError(
                        f"Self-adjacency not supported. (ielem = {ielem})"
                    )
                if adjind != -1 and adjind > ielem:
                    self._index_mapping[
                        adjind,
                        *EdgeUtils.inds_of_edge(
                            self._connections_type[ielem, iconn], self.degree
                        ),
                    ] = self._index_mapping[
                        ielem,
                        *EdgeUtils.inds_of_edge(iconn, self.degree),
                    ]
        self.nglob = nglob
        self._index_mapping_outdated = False

    def add_element_by_adjacency(
        self, self_edge: int, other_elem: int, other_edge: int, flip: None | bool = None
    ):
        if flip is None:
            flip = bool((self_edge & 0b100) ^ (other_edge & 0b100))
        self_edge &= 0b11
        other_edge &= 0b11

        flip_bit = 0b100 * flip

        self_elem = len(self._elements)
        self.add_element()
        self._update_connections()

        self._connections_ind[self_elem, self_edge] = other_elem
        self._connections_type[self_elem, self_edge] = other_edge | flip_bit
        self._connections_ind[other_elem, other_edge] = self_elem
        self._connections_type[other_elem, other_edge] = self_edge | flip_bit

    def get_element(self, ind: int):
        return self._elements[ind]

    def add_element(self):
        self._elements.append(Element2D(degree=self.degree, symbolic=self.symbolic))
        self._connections_outdated = True
        self._index_mapping_outdated = True

    def matrix_global_to_local_subview(self, ind: int):
        return (
            self._index_mapping[ind, :, :, None, None],
            self._index_mapping[ind, None, None, :, :],
        )

    def mass_matrix(self, use_gll_quadrature: bool = True):
        self._update_index_mapping()
        mat = np.zeros(
            (self.nglob, self.nglob), dtype=symgll.symtype if self.symbolic else float
        )
        for ielem, element in enumerate(self._elements):
            mat[
                self._index_mapping[ielem, :, :, None, None],
                self._index_mapping[ielem, None, None, :, :],
            ] += element.mass_matrix(use_gll_quadrature=use_gll_quadrature)
        return mat

    def stiffness_matrix(
        self, use_gll_quadrature: bool = True, include_boundary_terms=True
    ):
        self._update_index_mapping()
        mat = np.zeros(
            (self.nglob, self.nglob), dtype=symgll.symtype if self.symbolic else float
        )
        for ielem, element in enumerate(self._elements):
            mat[
                self._index_mapping[ielem, :, :, None, None],
                self._index_mapping[ielem, None, None, :, :],
            ] += element.stiffness_matrix(use_gll_quadrature=use_gll_quadrature)
        if include_boundary_terms:
            for bc in self.boundary_conditions:
                bc.apply_stiffness(self, mat, use_gll_quadrature=use_gll_quadrature)
        return mat

    def time_derivative_stiffness_matrix(self, use_gll_quadrature: bool = True):
        self._update_index_mapping()
        mat = np.zeros(
            (self.nglob, self.nglob), dtype=symgll.symtype if self.symbolic else float
        )
        for bc in self.boundary_conditions:
            bc.apply_stiffness_timederiv(
                self, mat, use_gll_quadrature=use_gll_quadrature
            )
        return mat


if __name__ == "__main__":
    import sympy as sp

    DEGREE = 2

    mesh_sym = Mesh2D(DEGREE, True)
    mesh_sym.add_element()
    MMs = mesh_sym.mass_matrix(use_gll_quadrature=False)
    SMs = mesh_sym.stiffness_matrix(use_gll_quadrature=False)

    GLL = _gll(DEGREE, True)
    L = GLL.polys
    Lp = GLL.deriv_polys

    # basis functions in both variables, in global basis
    bfunc = np.empty(mesh_sym.nglob, dtype=GLL.get_type())
    bfunc[mesh_sym._index_mapping[0]] = np.array(
        [[p * q.subs({"x": "y"}) for p in L] for q in L]
    )

    # vectorize options
    integ = np.vectorize(lambda v: sp.integrate(v, ("x", -1, 1), ("y", -1, 1)))
    mass_mat = integ(bfunc[:, None] * bfunc[None, :])
    dx = np.vectorize(lambda v: sp.diff(v, "x"))
    dy = np.vectorize(lambda v: sp.diff(v, "y"))
    stiff_mat = integ(
        dx(bfunc[:, None]) * dx(bfunc[None, :])
        + dy(bfunc[:, None]) * dy(bfunc[None, :])
    )

    # stiffness1D
    deriv_dots = np.array([[sp.integrate(p * q, ("x", -1, 1)) for p in Lp] for q in Lp])

    # mass1D
    dots = np.array([[sp.integrate(p * q, ("x", -1, 1)) for p in L] for q in L])

    assert np.all(stiff_mat == SMs)
    assert np.all(mass_mat == MMs)
