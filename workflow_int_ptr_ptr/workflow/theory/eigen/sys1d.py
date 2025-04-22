from typing import Literal

import sympy as sp

display = sp.pprint

t = sp.Symbol("t", real=True)
w, alpha = sp.symbols("w,alpha", real=True)


xlow = -1
xhigh = 1
lam = sp.symbols("lambda")
x = sp.Symbol("x", real=True)
u = sp.Function("u")(x)  # type: ignore
utilde = sp.Function(r"\tilde u")(x)  # type: ignore
v = sp.Function("v")(x)  # type: ignore
vtilde = sp.Function(r"\tilde v")(x)  # type: ignore


def _resid_1D(u, utilde=utilde, xlow=xlow, xhigh=xhigh, bd_terms=True):
    if bd_terms:
        bdplus = (utilde * (sp.simplify(sp.diff(u, x)) - lam * u) / 2).subs({x: xlow})
        bdminus = (utilde * (-sp.simplify(sp.diff(u, x)) - lam * u) / 2).subs(
            {x: xhigh}
        )
        rhs = (
            bdplus
            + bdminus
            - sp.integrate(sp.diff(utilde, x) * sp.diff(u, x), [(x, xlow, xhigh)])  # type: ignore
        )
    else:
        rhs = -sp.integrate(sp.diff(utilde, x) * sp.diff(u, x), [(x, xlow, xhigh)])  # type: ignore
    return sp.simplify(lam**2 * sp.integrate(utilde * u, [(x, xlow, xhigh)]) - rhs)


sig = sp.Function("sigma")(x)  # type: ignore
sigtilde = sp.Function(r"\tilde \sigma")(x)  # type: ignore


def _resid_system_1D(
    v, sig, vtilde=vtilde, sigtilde=sigtilde, xlow=xlow, xhigh=xhigh, bd_terms=True
):
    if bd_terms:
        bdplus = (+1) * (vtilde * sig + sigtilde * v).subs({x: xhigh})
        bdminus = (-1) * (vtilde * sig + sigtilde * v).subs({x: xlow})
        rhs = (
            bdplus
            + bdminus
            - sp.integrate(
                sp.diff(vtilde, x) * sig + sp.diff(sigtilde, x) * v, [(x, xlow, xhigh)]
            )
        )
    else:
        rhs = -sp.integrate(
            sp.diff(vtilde, x) * sig + sp.diff(sigtilde, x) * v, [(x, xlow, xhigh)]
        )  # type: ignore
    return sp.simplify(
        lam * sp.integrate(vtilde * v + sigtilde * sig, [(x, xlow, xhigh)]) - rhs
    )


def _coupled_resid_self_1D(resid_a, B, method="upwind", param_w=w, alpha=alpha, U=None):
    w = param_w
    if U is None:
        U = B.T
    jump_penalty = (
        (alpha / 2)
        * (B.subs({x: xlow}) - B.subs({x: xhigh}))
        * (U.subs({x: xhigh}) - U.subs({x: xlow}))
    )
    dudn_high = sp.diff(U, x).subs({x: xhigh})
    dudn_low = -sp.diff(U, x).subs({x: xlow})  # type: ignore
    if method == "upwind":
        # map high vals to low side
        bd_a = B.subs({x: xlow}) * (-dudn_high + lam * U.subs({x: xhigh})) / 2  # type: ignore

        # map low vals to high side
        bd_b = B.subs({x: xhigh}) * (-dudn_low + lam * U.subs({x: xlow})) / 2
        return sp.simplify(
            resid_a - (bd_a + bd_b + jump_penalty)
        )  # minus, since RHS is reversed.
    elif method == "midpoint":
        # ~u {{ sig }} . n
        return sp.simplify(
            resid_a
            - (  # minus since RHS is reversed.
                B.subs({x: xlow}) * ((1 - w) / 2 * dudn_low - (1 + w) / 2 * dudn_high)
                + B.subs({x: xhigh})
                * ((1 - w) / 2 * dudn_high - (1 + w) / 2 * dudn_low)
                + jump_penalty
            )
        )
    elif method == "symmetric":
        dvdn_avg = (sp.diff(B, x).subs({x: xhigh}) + sp.diff(B, x).subs({x: xlow})) / 2  # type: ignore
        dudn_avg = (dudn_high - dudn_low) / 2
        ujmp = U.subs({x: xhigh}) - U.subs({x: xlow})
        vjmp = B.subs({x: xhigh}) - B.subs({x: xlow})
        return sp.simplify(resid_a - (dvdn_avg * ujmp + vjmp * dudn_avg + jump_penalty))
    else:
        raise ValueError(f"Unknown method '{method}'.")


def genpoly(deg):
    # return sp.simplify(sp.legendre(deg,x).subs({x:(xhigh+xlow)/2 + x*(xhigh-xlow)/2}))
    if deg < 2:
        return x ** (deg)
    if deg < 4:
        return (x - xlow) * (xhigh - x) * x ** (deg - 2)  # type: ignore
    else:
        return (x - xlow) ** 2 * (xhigh - x) ** 2 * x ** (deg - 4)  # type: ignore


def get_residual_1D(
    method: Literal["symmetric", "midpoint", "upwind"], deg: int = 4, alpha=alpha, w=w
):
    dim = deg + 1
    B = sp.Matrix([genpoly(k) for k in range(dim)])
    resid_matrix_orig = sp.expand(_resid_1D(B.T, B, bd_terms=(method == "upwind")))
    residual_matrix = _coupled_resid_self_1D(
        resid_matrix_orig, B, method=method, param_w=w, alpha=alpha
    )

    return residual_matrix


def residual_to_system(residual):
    dim = residual.shape[0]
    resid_diff = sp.diff(residual, lam)
    resid_ddif = sp.diff(resid_diff, lam)
    matC = residual.subs({lam: 0})
    matB = resid_diff.subs({lam: 0})
    matA = resid_ddif.subs({lam: 0}) / 2  # type: ignore
    matAinv = matA.inv()

    # return resid_couple
    return sp.Matrix([[sp.zeros(dim), sp.eye(dim)], [-matAinv @ matC, -matAinv @ matB]])
