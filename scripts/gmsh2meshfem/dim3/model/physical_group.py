from dataclasses import dataclass

import numpy as np

from ...gmsh_dep import GmshContext
from ...helper.index_mapping import IndexMapping
from .faces import FaceType


@dataclass
class SurfacePhysicalGroup:
    name: str
    element_inds: np.ndarray
    element_faces: np.ndarray

    @staticmethod
    def from_gmsh(
        gmsh: GmshContext,
        name: str,
        element_nodes: np.ndarray,
        node_reindexing: IndexMapping,
    ) -> "SurfacePhysicalGroup":
        dim = 2

        # =========================================
        # recover physical group tags from the name
        # =========================================
        grouptags = []
        for _dim, _tag in gmsh.model.get_physical_groups():
            if _dim == dim and gmsh.model.get_physical_name(_dim, _tag) == name:
                # found a tag with this identity
                grouptags.append(_tag)

        # breakpoint()
        # entitylist = []
        # for tag in grouptags:
        #     entitylist.extend(
        #         gmsh.model.get_entities_for_physical_group(dim=dim, tag=tag)
        #     )

        # ============================
        # get element faces from nodes
        # ============================
        elements = []
        faces = []

        for grouptag in grouptags:
            node_tags, coord_ = gmsh.model.mesh.get_nodes_for_physical_group(
                dim, grouptag
            )
            node_tags = node_reindexing.apply(node_tags)


            # before facial expansion, get all elements that have nodes in this physical group
            rough_element_match = np.where(
                np.any(np.isin(element_nodes, node_tags), axis=-1)
            )[0]

            # count a face if every node in the face is in the physical group
            fine_element_match, face_match = np.where(
                np.all(
                    np.isin(
                        element_nodes[
                            rough_element_match[:, None, None],
                            FaceType.HEX_27_edge_to_inds_matrix()[None, :, :],
                        ],
                        node_tags,
                    ),
                    axis=-1,
                )
            )
            elements.extend(rough_element_match[fine_element_match])
            faces.extend(face_match)

        return SurfacePhysicalGroup(
            name=name,
            element_inds=np.array(elements, dtype=int),
            element_faces=np.array(faces, dtype=np.uint8),
        )
