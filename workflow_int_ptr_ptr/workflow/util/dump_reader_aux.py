import numpy as np

try:
    from . import dump_reader
except ImportError:
    import dump_reader


class field_remapper:
    """
    Class for mapping fields of X to Y, by providing the coefficients in the different basis.
    """

    def __init__(self, pts_x, pts_y):
        # pts indices = (ispec, iz, ix, icomp)
        nspecx, ngllzx, ngllxx, ncompx = pts_x.shape
        nspecy, ngllzy, ngllxy, ncompy = pts_y.shape
        assert ncompx == 2 and ncompy == 2, "pts must both be 2D"

        GLLX = dump_reader.build_gll_namespace(ngllxx, ngllzx)
        self.GLLX = GLLX

        ispec_closest = np.empty((nspecy, ngllzy, ngllxy), dtype=int)
        # batch all y points in each element: find the closest x-element for each.
        # we assume y is in an x-element if distance to lerp(spec_point,spec_center, eps) is minimized
        x_centers = np.mean(pts_x, axis=(1, 2))
        y_centers = np.mean(pts_y, axis=(1, 2))
        shft = 1e-2
        x_shft = shft * x_centers[:, None, None, :] + (1 - shft) * pts_x
        y_shft = shft * y_centers[:, None, None, :] + (1 - shft) * pts_y
        for i in range(nspecy):
            ispec_closest[i, ...] = np.argmin(
                np.min(
                    np.linalg.norm(
                        y_shft[i, np.newaxis, np.newaxis, np.newaxis, :, :, :]
                        - x_shft[:, :, :, np.newaxis, np.newaxis, :],
                        ord=2,
                        axis=-1,
                    ),  # (ispecx,izx,ixx,izy,ixy)
                    axis=(1, 2),
                ),  # (ispecx,izy,ixy)
                axis=0,
            )
        self.ispec_closest = ispec_closest

        y_interpolants = pts_x[ispec_closest, :, :, :]

        # optimize with newtons
        def refine(coords, maxiters, coordtol=1e-8, gradtol=1e-10):
            # minimize: f = 1/2 |y_interpolants[...,a,b] L_b(xi) L_a(gamma) - pts_y|^2
            # d_err = |y_interpolants[...,a,b] L_b(xi) L_a(gamma) - pts_y|

            # write Ui as the interpolation function y_interpolants[...,a,b,i] L_b(xi) L_a(gamma)
            # dj f = (Ui - yi) * (dj Ui)
            # dk dj f = (dk Ui) * (dj Ui) + (Ui - yi) * (dk dj Ui)
            for _ in range(maxiters):
                xi_pows = coords[..., 0, np.newaxis] ** np.arange(GLLX.ngll_x)
                ga_pows = coords[..., 1, np.newaxis] ** np.arange(GLLX.ngll_z)

                # U - y
                Umy = (
                    np.einsum(
                        "...abi,aj,bk,...j,...k->...i",
                        y_interpolants,
                        GLLX.L_z,
                        GLLX.L_x,
                        ga_pows,
                        xi_pows,
                    )
                    - pts_y
                )

                maxerr = np.max(np.einsum("...i,...j->...", Umy, Umy))
                if maxerr < coordtol:
                    return

                dU = np.empty((nspecy, ngllzy, ngllxy, 2, 2, 1))
                dU[..., 0, 0] = np.einsum(
                    "...abi,aj,bk,...j,...k->...i",
                    y_interpolants,
                    GLLX.L_z,
                    GLLX.Lp_x,
                    ga_pows,
                    xi_pows[..., :-1],
                )
                dU[..., 1, 0] = np.einsum(
                    "...abi,aj,bk,...j,...k->...i",
                    y_interpolants,
                    GLLX.Lp_z,
                    GLLX.L_x,
                    ga_pows[..., :-1],
                    xi_pows,
                )

                maxgrad = np.max(np.einsum("...ia,...ja->...", dU, dU))
                if maxgrad < gradtol:
                    return

                # hess of f (not U)
                H = np.einsum("...bia,...bja->...ij", dU, dU)
                H[..., 0, 0] += np.einsum(
                    "...abi,aj,bk,...j,...k,...i->...",
                    y_interpolants,
                    GLLX.L_z,
                    GLLX.Lpp_x,
                    ga_pows,
                    xi_pows[..., :-2],
                    Umy,
                )
                H[..., 1, 0] += np.einsum(
                    "...abi,aj,bk,...j,...k,...i->...",
                    y_interpolants,
                    GLLX.Lp_z,
                    GLLX.Lp_x,
                    ga_pows[..., :-1],
                    xi_pows[..., :-1],
                    Umy,
                )
                H[..., 0, 1] = H[..., 1, 0]
                H[..., 1, 1] += np.einsum(
                    "...abi,aj,bk,...j,...k,...i->...",
                    y_interpolants,
                    GLLX.Lpp_z,
                    GLLX.L_x,
                    ga_pows[..., :-2],
                    xi_pows,
                    Umy,
                )

                coords -= np.linalg.solve(H, np.einsum("...ijk,...i->...jk", dU, Umy))[
                    ..., 0
                ]

        coords = np.empty((nspecy, ngllzy, ngllxy, 2))
        # initial guess: closest point
        indZ, indX = np.unravel_index(
            np.argmin(
                np.linalg.norm(
                    y_interpolants - pts_y[..., None, None, :], ord=2, axis=-1
                ).reshape((nspecy, ngllzy, ngllxy, ngllzx * ngllxx)),
                axis=-1,
            ),
            (ngllzx, ngllxx),
        )
        coords[..., 0] = GLLX.knots_x[indX]
        coords[..., 1] = GLLX.knots_z[indZ]
        refine(coords, 20)

        self.lag_xi_closest = np.einsum(
            "aj,...j->...a", GLLX.L_x, coords[..., 0, None] ** np.arange(GLLX.ngll_x)
        )
        self.lag_ga_closest = np.einsum(
            "aj,...j->...a", GLLX.L_z, coords[..., 1, None] ** np.arange(GLLX.ngll_z)
        )

    def transfer_field(self, field):
        """_summary_

        Args:
            field (ndarray): the field (indices [ispecy,iz,ix,...]) to map to Y nodes

        Returns:
            ndarray: the mapped field (indices [ispec,iz,ix,...] in Y nodes)
        """

        f_interpolants = field[self.ispec_closest, ...]
        return np.einsum(
            "suvab...,suva,suvb->suv...",
            f_interpolants,
            self.lag_ga_closest,
            self.lag_xi_closest,
        )

    def __call__(self, field):
        return self.transfer_field(field)


if __name__ == "__main__":
    import os

    import config

    test = config.get("cg_compare.tests.0")
    folder = os.path.join(config.get("cg_compare.workspace_folder"), test["name"])

    prov = dump_reader.dump_series.load_from_file(
        os.path.join(folder, config.get("cg_compare.workspace_files.provenance_dump"))
    )
    test = dump_reader.read_dump_file(
        os.path.join(folder, config.get("cg_compare.workspace_files.dump_prefix"))
        + "0.dat"
    )
    prov_frame = prov.get_frame_as_dump_frame(
        int(np.argmin(np.abs(prov.time_indices - 300)))
    )
    remap = field_remapper(prov_frame.pts, test.pts)

    def testfcn(f):
        if isinstance(f, np.ndarray):
            f_prov = f
        else:
            f_prov = f(prov_frame.pts)
        f_test = remap(f_prov)

        maxf = np.max(f_prov)
        minf = np.min(f_prov)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 10))
        prov_frame.plot_field(f_prov, show=False, ptsize=30, vmin=minf, vmax=maxf)
        test.plot_field(f_test, show=False, ptsize=10, vmin=minf, vmax=maxf)
        plt.show()

    # testfcn(lambda pts: pts[...,0])
    # testfcn(lambda pts: pts[...,1])
    # testfcn(lambda pts: np.linalg.norm(pts[...] - 0.5, axis=-1))
    testfcn(np.linalg.norm(prov_frame.displacement, axis=-1))
