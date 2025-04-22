from typing import Any

import numpy as np
import sympy as sp

import workflow.util.GLL as NUMGLL

x = sp.Symbol("x")
symtype = sp.core.add.Add


# Bonnet's formula for finding Legendre polynomials
def leg(n):
    """Calculates the n^th degree legendre polynomial defined on [-1,1].
    The returned value is a coefficient array where the polynomial
    is given by sum(a[i] * x**i)
    """
    Pkm1 = sp.Integer(1)  # P_{k-1} ; initially k=1
    Pk = x  # P_k     ; initially k=1
    if n == 0:
        return Pkm1
    if n == 1:
        return Pk

    for k in range(1, n):
        Pkp1 = sp.simplify(
            (sp.Integer(2 * k + 1) * Pk * x - sp.Integer(k) * Pkm1) / (k + 1)
        )

        # inc k
        Pkm1 = Pk
        Pk = Pkp1

    return Pk


def _polyinteg(p):
    """Integrates the polynomial p over the interval [-1,1]"""
    return sp.integrate(p, [(x, -1, 1)])


def polydot(p, q):
    """Returns the inner product of p and q over [-1,1]"""
    return _polyinteg(p * q)


def polyderiv(p):
    """Returns the derivative p'"""
    return sp.diff(p, x)


def polyeval(p, point) -> Any:
    """Returns the evaluated p(x)"""
    if isinstance(p, list):
        return [polyeval(k, point) for k in p]
    if isinstance(p, np.ndarray):
        return np.array([polyeval(k, point) for k in p], dtype=p.dtype)

    return sp.lambdify(x, p)(point)


def get_knots(n, as_symbolic=False, dtype=float, i_know_what_im_doing=False) -> Any:
    """Estimates the roots to be used for GLL quadrature"""
    if as_symbolic:
        if n > 6 and not i_know_what_im_doing:
            raise ValueError(
                f"Symbolic root solve for n={n}, which requires a {n - 1} degree root-solve.\n"
                "Set i_know_what_im_doing = True to proceed."
            )
        return np.sort(np.array([-1, *sp.solve(polyderiv(leg(n))), 1], dtype=symtype))  # type: ignore
    return np.sort(np.array([-1, *sp.nroots(polyderiv(leg(n))), 1], dtype=dtype))  # type: ignore


def build_lagrange_polys(
    n, as_symbolic=False, dtype=float, i_know_what_im_doing=False, knots=None
) -> Any:
    if knots is None:
        knots = get_knots(
            n,
            as_symbolic=as_symbolic,
            dtype=dtype,
            i_know_what_im_doing=i_know_what_im_doing,
        )

    L = [sp.sympify(1)] * (n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                L[i] = sp.simplify(L[i] * (x - knots[j]) / (knots[i] - knots[j]))
    if as_symbolic:
        return np.array(L, dtype=symtype)
    coefs = [list(reversed(sp.Poly(p, x, domain="R").all_coeffs())) for p in L]  # type:ignore
    return np.array(coefs, dtype=dtype)


def get_lagrange_weights(
    n,
    as_symbolic=False,
    dtype=float,
    i_know_what_im_doing=False,
    knots=None,
) -> Any:
    if knots is None:
        knots = get_knots(
            n,
            as_symbolic=as_symbolic,
            dtype=dtype,
            i_know_what_im_doing=i_know_what_im_doing,
        )
    P = leg(n)
    if as_symbolic:
        return (
            sp.Rational(2, n * (n + 1))
            * np.array(
                [
                    1,
                    *(
                        P.subs({x: k}) ** (-2)  # type:ignore
                        for k in knots[1:-1]
                    ),
                    1,
                ],
                dtype=symtype if as_symbolic else dtype,
            )
        )
    else:
        return np.array(
            [1, *polyeval(P, knots[1:-1]) ** (-2), 1],
            dtype=symtype if as_symbolic else dtype,
        ) * (2 / (n * (n + 1)))


class GLL:
    def __init__(
        self,
        degree: int,
        as_symbolic: bool = False,
        dtype=float,
        i_know_what_im_doing: bool = False,
    ):
        self.knots = get_knots(
            degree,
            as_symbolic=as_symbolic,
            dtype=dtype,
            i_know_what_im_doing=i_know_what_im_doing,
        )
        self.polys = build_lagrange_polys(
            degree,
            as_symbolic=as_symbolic,
            dtype=dtype,
            i_know_what_im_doing=i_know_what_im_doing,
            knots=self.knots,
        )
        self.weights = get_lagrange_weights(
            degree,
            as_symbolic=as_symbolic,
            dtype=dtype,
            i_know_what_im_doing=i_know_what_im_doing,
            knots=self.knots,
        )
        self.degree = degree
        self.is_symbolic = as_symbolic

        deriv_func = polyderiv if as_symbolic else NUMGLL.polyderiv
        self.deriv_polys = np.array(
            [deriv_func(p) for p in self.polys], dtype=self.get_type()
        )
        self.deriv_at_knots = np.array(
            [self.field_at_knots1D(p) for p in self.deriv_polys], dtype=self.get_type()
        )
        self._populate_dot_products()

    def _populate_dot_products(self):
        result_true = np.empty(
            (self.degree + 1, self.degree + 1), dtype=self.get_type()
        )
        result_quad = np.zeros(
            (self.degree + 1, self.degree + 1), dtype=self.get_type()
        )
        dotfunc = polydot if self.is_symbolic else NUMGLL.polydot
        for i in range(self.degree + 1):
            result_quad[i, i] = self.weights[i]
            for j in range(self.degree + 1):
                result_true[i, j] = dotfunc(self.polys[i], self.polys[j])
        self.true_dot_products = result_true
        self.quad_dot_products = result_quad

        self.true_deriv_dot_poly = np.einsum(
            "im,in->mn",
            self.deriv_at_knots,
            self.true_dot_products,
        )
        self.quad_deriv_dot_poly = np.einsum(
            "im,in->mn",
            self.deriv_at_knots,
            self.quad_dot_products,
        )
        self.true_poly_dot_deriv = np.einsum(
            "mj,jn->mn",
            self.true_dot_products,
            self.deriv_at_knots,
        )
        self.quad_poly_dot_deriv = np.einsum(
            "mj,jn->mn",
            self.quad_dot_products,
            self.deriv_at_knots,
        )
        self.true_deriv_dot_deriv = np.einsum(
            "im,ij,jn->mn",
            self.deriv_at_knots,
            self.true_dot_products,
            self.deriv_at_knots,
        )
        self.quad_deriv_dot_deriv = np.einsum(
            "im,ij,jn->mn",
            self.deriv_at_knots,
            self.quad_dot_products,
            self.deriv_at_knots,
        )

    def get_type(self):
        return symtype if self.is_symbolic else float

    def get_weights(self):
        return self.weights

    def dot(self, I_ind: int | np.ndarray, J_ind: int | np.ndarray, quadrature=True):
        if isinstance(I_ind, int):
            I_ind = np.array(I_ind)
        if isinstance(J_ind, int):
            J_ind = np.array(J_ind)

        if quadrature:
            return self.true_dot_products[I_ind, J_ind]
        else:
            return self.quad_dot_products[I_ind, J_ind]

    def field_at_knots1D(self, field):
        if self.is_symbolic:
            return [field.subs({x: k}) for k in self.knots]
        else:
            return [NUMGLL.polyeval(field, k) for k in self.knots]

    def interpolate_poly(self, pt):
        if self.is_symbolic:
            return np.array([sp.lambdify(x, p)(pt) for p in self.polys])
        return np.einsum(
            "ij,...j->i...",
            self.polys,
            np.array(pt)[..., None] ** np.arange(self.degree + 1),
        )

    def integrate1D(self, field, quadrature=True, field_as_coefs=True):
        if quadrature:
            if field_as_coefs:
                field = self.field_at_knots1D(field)

            return sum(field[i] * self.weights[i] for i in range(self.degree + 1))
        else:
            if field_as_coefs:
                field = sum(field[i] * self.polys[i] for i in range(self.degree + 1))

            integ = _polyinteg if self.is_symbolic else NUMGLL._polyinteg
            return integ(field)

    def stiffness2D(self, riemannian_metric=None, quadrature=True):
        if self.is_symbolic:
            if riemannian_metric:
                raise NotImplementedError(
                    "symgll.GLL.stiffness2D with riemannian metric is not yet supported "
                    "for symbolic GLL"
                )
            result = np.zeros(
                (self.degree + 1, self.degree + 1, self.degree + 1, self.degree + 1),
                dtype=self.get_type(),
            )
            for i in range(self.degree + 1):
                dLi = self.deriv_polys[i]
                for j in range(i, self.degree + 1):
                    dLj = self.deriv_polys[j]
                    if quadrature:
                        dot_deriv = self.integrate1D(
                            dLi * dLj, quadrature=True, field_as_coefs=True
                        )
                    else:
                        # use dot product, since derivs are lower degree
                        dot_deriv = np.einsum(
                            "i,ij,j",
                            self.deriv_at_knots[i],
                            self.true_dot_products,
                            self.deriv_at_knots[j],
                        )
                    # (out x, out y, in x, in y)

                    # derivatives along x   (L'i L: times L'j L:)
                    result[i, :, j, :] += dot_deriv * self.true_dot_products
                    # derivatives along y   (L: L'i times L: L'j)
                    result[:, i, :, j] += dot_deriv * self.true_dot_products

                    if j != i:
                        result[j, :, i, :] += dot_deriv * self.true_dot_products
                        result[:, j, :, i] += dot_deriv * self.true_dot_products
        else:
            if riemannian_metric:
                raise NotImplementedError(
                    "symgll.GLL.stiffness2D with riemannian metric is not yet supported "
                    "for non-symbolic GLL"
                )
            # for now, this is just a copy/paste from above, but we may want to optimize each branch
            result = np.zeros(
                (self.degree + 1, self.degree + 1, self.degree + 1, self.degree + 1),
                dtype=self.get_type(),
            )
            for i in range(self.degree + 1):
                dLi = self.deriv_polys[i]
                for j in range(i, self.degree + 1):
                    dLj = self.deriv_polys[j]
                    if quadrature:
                        if self.is_symbolic:
                            dot_deriv = self.integrate1D(
                                dLi * dLj, quadrature=True, field_as_coefs=True
                            )
                        else:
                            dot_deriv = self.integrate1D(
                                self.deriv_at_knots[i] * self.deriv_at_knots[j],
                                quadrature=True,
                                field_as_coefs=False,
                            )
                    else:
                        # use dot product, since derivs are lower degree
                        dot_deriv = np.einsum(
                            "i,ij,j",
                            self.deriv_at_knots[i],
                            self.true_dot_products,
                            self.deriv_at_knots[j],
                        )

                    result[i, :, j, :] += dot_deriv * self.true_dot_products
                    result[:, i, :, j] += dot_deriv * self.true_dot_products

                    result[j, :, i, :] += dot_deriv * self.true_dot_products
                    result[:, j, :, j] += dot_deriv * self.true_dot_products
        return result
