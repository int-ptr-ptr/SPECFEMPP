import numpy as np
import scipy.optimize

from .. import lagrange

L = lagrange.build_lagrange_polys([-1, 0, 1])
Lp = lagrange.differentiate_polys(L)

maxfind_coefs_a = -Lp[:, 1]
maxfind_coefs_b = Lp[:, 0]


def sample_face(
    face_nodes: np.ndarray,
    local_coords: np.ndarray,
    result_arr: np.ndarray | None = None,
) -> np.ndarray:
    """samples the face at the given coordinates.

    Parameters
    ----------
    face_nodes : np.ndarray
        shape - (...,9,3) array of node locations (MSH_QUAD_9 layout)
    local_coords : np.ndarray
        shape - (...,2) array of local coordinates
    result_arr : np.ndarray | None, optional
        if set, instead of creating a new array, the result is placed into this. By default None

    Returns
    -------
    np.ndarray
        shape - (...,3) array of global coordinates
    """

    # i - node
    # k - power
    # d - dimension of local coordinate
    coefs = np.einsum("ik,...dk->...id", L, local_coords[..., None] ** np.arange(3))
    if result_arr is None:
        out = face_nodes[..., 0, :] * coefs[..., 0, 0] * coefs[..., 0, 1]
    else:
        out = result_arr
        out[...] = face_nodes[..., 0, :] * coefs[..., 0, 0] * coefs[..., 0, 1]
    out += face_nodes[..., 1, :] * coefs[..., 2, 0] * coefs[..., 0, 1]
    out += face_nodes[..., 2, :] * coefs[..., 2, 0] * coefs[..., 2, 1]
    out += face_nodes[..., 3, :] * coefs[..., 0, 0] * coefs[..., 2, 1]
    out += face_nodes[..., 4, :] * coefs[..., 1, 0] * coefs[..., 0, 1]
    out += face_nodes[..., 5, :] * coefs[..., 2, 0] * coefs[..., 1, 1]
    out += face_nodes[..., 6, :] * coefs[..., 1, 0] * coefs[..., 2, 1]
    out += face_nodes[..., 7, :] * coefs[..., 0, 0] * coefs[..., 1, 1]
    out += face_nodes[..., 8, :] * coefs[..., 1, 0] * coefs[..., 1, 1]
    return out


def sample_face_jacobian(
    face_nodes: np.ndarray,
    local_coords: np.ndarray,
    result_arr: np.ndarray | None = None,
) -> np.ndarray:
    """samples the Jacobian matrix of a face at the given coordinates.

    Parameters
    ----------
    face_nodes : np.ndarray
        shape - (...,9,3) array of node locations (MSH_QUAD_9 layout)
    local_coords : np.ndarray
        shape - (...,2) array of local coordinates
    result_arr : np.ndarray | None, optional
        if set, instead of creating a new array, the result is placed into this. By default None

    Returns
    -------
    np.ndarray
        shape - (...,3,2) array of floats (stacked 3x2 Jacobian matrices)
    """
    stack_shape = np.broadcast_shapes(face_nodes.shape[:-2], local_coords.shape[:-1])

    # remaining indices: (node index, dimension of local coordinate, dimension of derivative)
    coefs = np.empty(stack_shape + (3, 2, 2))

    # i - node
    # k - power
    # d - dimension of local coordinate
    coefs[..., :, (0, 1), (1, 0)] = np.einsum(
        "ik,...dk->...id", L, local_coords[..., None] ** np.arange(3)
    )  # L(v)
    coefs[..., :, (0, 1), (0, 1)] = np.einsum(
        "ik,...dk->...id", Lp, local_coords[..., None] ** np.arange(2)
    )  # L'(u)

    if result_arr is None:
        out = np.empty(stack_shape + (3, 2))
    else:
        out = result_arr
    out[...] = face_nodes[..., 0, :, None] * coefs[..., 0, 0, :] * coefs[..., 0, 1, :]
    out += face_nodes[..., 1, :, None] * coefs[..., 2, 0, :] * coefs[..., 0, 1, :]
    out += face_nodes[..., 2, :, None] * coefs[..., 2, 0, :] * coefs[..., 2, 1, :]
    out += face_nodes[..., 3, :, None] * coefs[..., 0, 0, :] * coefs[..., 2, 1, :]
    out += face_nodes[..., 4, :, None] * coefs[..., 1, 0, :] * coefs[..., 0, 1, :]
    out += face_nodes[..., 5, :, None] * coefs[..., 2, 0, :] * coefs[..., 1, 1, :]
    out += face_nodes[..., 6, :, None] * coefs[..., 1, 0, :] * coefs[..., 2, 1, :]
    out += face_nodes[..., 7, :, None] * coefs[..., 0, 0, :] * coefs[..., 1, 1, :]
    out += face_nodes[..., 8, :, None] * coefs[..., 1, 0, :] * coefs[..., 1, 1, :]
    return out


def locate_intersection(
    face1: np.ndarray,
    face2: np.ndarray,
    face1_local_coords_guess: np.ndarray,
    face2_local_coords_guess: np.ndarray,
):
    """Finds a point on either face that minimizes the distance between the surfaces.

    Returns the tuple (face1_coords, face2_coords, residual displacement)

    Parameters
    ----------
    face1 : np.ndarray
        nodal representation of face1
    face2 : np.ndarray
        nodal representation of face2
    face1_local_coords_guess : np.ndarray
        initial guess in local coordinates
    face2_local_coords_guess : np.ndarray
        initial guess in local coordinates
    """

    def func(x):
        return sample_face(face1, x[:2]) - sample_face(face2, x[2:])
        # return np.concatenate(
        #     [
        #         # residual
        #         sample_face(face1, x[:2]) - sample_face(face2, x[2:]),
        #         # penalty
        #         x,
        #     ]
        # )

    def jac(x):
        return np.concatenate(
            (sample_face_jacobian(face1, x[:2]), -sample_face_jacobian(face2, x[2:])),
            axis=-1,
        )

    result = scipy.optimize.least_squares(
        fun=func,
        x0=np.concatenate([face1_local_coords_guess, face2_local_coords_guess]),
        jac=jac,  # type: ignore
        bounds=((-1, -1, -1, -1), (1, 1, 1, 1)),
        method="trf",
    )

    return result.x[:2], result.x[2:], result.fun


def locate_intersection2(
    face1: np.ndarray,
    face2: np.ndarray,
    face1_local_coords_guess: np.ndarray,
    face2_local_coords_guess: np.ndarray,
    length_scale: float,
):
    """Finds a point on either face that minimizes the distance between the surfaces.

    Returns the tuple (face1_coords, face2_coords, residual displacement)

    Parameters
    ----------
    face1 : np.ndarray
        nodal representation of face1
    face2 : np.ndarray
        nodal representation of face2
    face1_local_coords_guess : np.ndarray
        initial guess in local coordinates
    face2_local_coords_guess : np.ndarray
        initial guess in local coordinates
    """
    max_iters = 100
    displacement_eps = 1e-5
    face1_local_coords = np.copy(face1_local_coords_guess)
    face2_local_coords = np.copy(face2_local_coords_guess)

    face1_loc = sample_face(face1, face1_local_coords)
    face2_loc = sample_face(face2, face2_local_coords)

    face1_jac = sample_face_jacobian(face1, face1_local_coords)
    face2_jac = sample_face_jacobian(face2, face2_local_coords)

    for iiter in range(max_iters):
        resid = face2_loc - face1_loc

        # compute orthogonal residual
        face1_norm = np.cross(face1_jac[:, 0], face1_jac[:, 1])
        face1_norm /= np.linalg.norm(face1_norm)
        face2_norm = np.cross(face2_jac[:, 0], face2_jac[:, 1])
        face2_norm /= np.linalg.norm(face2_norm)

        # we break into cases based on if the tangents are ortho or not
        sintheta = np.linalg.norm(np.cross(face1_norm, face2_norm))

        if sintheta > 0.1:  # approx 5.<something> degrees
            # planes intersect. choose a point on that intersection

            if np.linalg.norm(resid) < displacement_eps * length_scale:
                return face1_local_coords, face2_local_coords, resid

            along_intersection = np.cross(face1_norm, face2_norm)
            along_intersection /= np.linalg.norm(along_intersection)

            face1_stepdir = np.cross(along_intersection, face1_norm)
            face2_stepdir = np.cross(along_intersection, face2_norm)

            (neg_face1stepfac, face2_stepfac), resid_norm2 = np.linalg.lstsq(
                np.stack([face1_stepdir, face2_stepdir], axis=1), resid
            )[:2]
            assert np.shape(resid_norm2) == (1,) and resid_norm2 < 1e-10, (
                "Step should have zero residual."
            )
            face1_stepdir *= -neg_face1stepfac
            face2_stepdir *= face2_stepfac

            # convert to local coordinates
            face1_localstep = np.linalg.lstsq(face1_jac, face1_stepdir)[0]
            face2_localstep = np.linalg.lstsq(face2_jac, face2_stepdir)[0]
        else:
            # planes are (near) parallel. shift down to 2D (local coords)
            resid_in_face1_local = np.linalg.lstsq(face1_jac, resid)[0]
            resid_in_face2_local = np.linalg.lstsq(face2_jac, resid)[0]

            # midpoint (split the difference)
            face1_localstep = resid_in_face1_local / 2
            face2_localstep = -resid_in_face2_local / 2

            if (
                np.linalg.norm(resid_in_face1_local) < displacement_eps
                and np.linalg.norm(resid_in_face2_local) < displacement_eps
            ):
                return face1_local_coords, face2_local_coords, resid

        face1_local_coords += face1_localstep
        face2_local_coords += face2_localstep

        # clamp to boundary
        face1_local_coords[:] = np.minimum(1, face1_local_coords)
        face1_local_coords[:] = np.maximum(-1, face1_local_coords)
        face2_local_coords[:] = np.minimum(1, face2_local_coords)
        face2_local_coords[:] = np.maximum(-1, face2_local_coords)


def get_interior_tangent_vector(
    face_jacobian: np.ndarray, coord: np.ndarray, projected_plane_normal: np.ndarray
):
    """Recovers the inward facing normal (or tangent bisector for corners) and the angle made
    to the boundary. This is used to check if the tangent half or quarter space overlaps between
    two faces
    """
    eps = 1e-2
    plane_proj = projected_plane_normal / np.linalg.norm(projected_plane_normal)

    left_right_boundary = False
    v1 = np.zeros(3)
    if coord[0] < eps - 1:
        v1 = face_jacobian[:, 0] / np.linalg.norm(face_jacobian[:, 0])
        left_right_boundary = True

    elif coord[0] > 1 - eps:
        v1 = -face_jacobian[:, 0] / np.linalg.norm(face_jacobian[:, 0])
        left_right_boundary = True

    top_bottom_boundary = False
    v2 = np.zeros(3)
    if coord[1] < eps - 1:
        v2 = face_jacobian[:, 1] / np.linalg.norm(face_jacobian[:, 1])
        top_bottom_boundary = True

    elif coord[1] > 1 - eps:
        v2 = -face_jacobian[:, 1] / np.linalg.norm(face_jacobian[:, 1])
        top_bottom_boundary = True

    if left_right_boundary and not top_bottom_boundary:
        v1 -= plane_proj * np.dot(plane_proj, v1)
        v1 /= np.linalg.norm(v1)
        return v1, np.pi / 2
    if top_bottom_boundary and not left_right_boundary:
        v2 -= plane_proj * np.dot(plane_proj, v2)
        v2 /= np.linalg.norm(v2)
        return v2, np.pi / 2
    if left_right_boundary and top_bottom_boundary:
        v1 -= plane_proj * np.dot(plane_proj, v1)
        v1 /= np.linalg.norm(v1)
        v2 -= plane_proj * np.dot(plane_proj, v2)
        v2 /= np.linalg.norm(v2)
        vbisect = v1 + v2
        vbisect /= np.linalg.norm(vbisect)
        return vbisect, np.acos(np.dot(v1, v2)) / 2

    v1[0] = 1
    return v1, np.pi


def faces_intersect(
    face1: np.ndarray,
    face2: np.ndarray,
    *,
    matching_localcoords: tuple[np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Computes whether or not face1 and face2 have a large enough intersection
    for the sake of a nonconforming interface.

    If `matching_localcoords = (face1_local_coords, face2_local_coords)` is provided,
    then the `locate_intersection` routine to find matching points is skipped.
    """
    # rough guess on length scale
    length_scale = max(
        np.linalg.norm(np.std(face1, axis=0), axis=-1),
        np.linalg.norm(np.std(face2, axis=0), axis=-1),
    )

    # we may want to improve this, but later

    # find the intersection, check if the planes are close enough to parallel,
    # the distance is small enough, and:
    #   if both found values are on the boundary, then the inward directions
    #   are in a similar direction.

    if matching_localcoords is None:
        face1_coord, face2_coord, resid = locate_intersection(
            face1, face2, np.zeros(2), np.zeros(2)
        )

        if np.linalg.norm(resid) > length_scale * 5e-2:
            return False
    else:
        face1_coord, face2_coord = matching_localcoords

    face1_jac = sample_face_jacobian(face1, face1_coord)
    face2_jac = sample_face_jacobian(face2, face2_coord)

    face1_norm = np.cross(face1_jac[:, 0], face1_jac[:, 1])
    face2_norm = np.cross(face2_jac[:, 0], face2_jac[:, 1])

    face1_norm /= np.linalg.norm(face1_norm)
    face2_norm /= np.linalg.norm(face2_norm)

    sintheta = np.linalg.norm(np.cross(face1_norm, face2_norm))

    if sintheta > 0.1:  # approx 5.<something> degrees
        return False

    # project onto avg direction of face1 and face2 norms (face*_norm are normalized,
    # so projection_normal gives the direction of the angle bisector)
    projection_normal = (
        face1_norm + face2_norm
        if np.dot(face1_norm, face2_norm) > 0
        else face1_norm - face2_norm
    )

    # CHECK: inward directions overlap
    face1_interior, face1_theta = get_interior_tangent_vector(
        face1_jac, face1_coord, projected_plane_normal=projection_normal
    )
    face2_interior, face2_theta = get_interior_tangent_vector(
        face2_jac, face2_coord, projected_plane_normal=projection_normal
    )

    if face1_theta >= np.pi or face2_theta >= np.pi:
        # interior
        return True

    # assuming the tangent planes are the same, we want to see if the cones intersect:
    # face*_interior is the bisector of the cone, face*_theta is the angle from the bisector
    # to the surface of the cone (think FOV)
    # pad by e-2, which would be the intersection sliver.
    if np.dot(face1_interior, face2_interior) < np.cos(
        max(face1_theta + face2_theta - 5e-2, 0)
    ):
        return False

    return True
