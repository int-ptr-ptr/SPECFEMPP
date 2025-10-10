from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Iterable

import numpy as np
from _gmsh2meshfem.gmsh_dep import GmshContext

from .boundary import BoundarySpec
from .edges import ConformingInterfaces
from .index_mapping import ComposedIndexMapping, IndexMapping, JoinedIndexMapping
from .nonconforming_interfaces import (
    NonconformingInterfaces,
)
from .physical_group import (
    NullPhysicalGroup,
    PhysicalGroup,
    UnionPhysicalGroup,
    physical_group_from_name,
)
from .plotter import plot_model


# TODO: consider using some sort of joint node and element index mapping container during
# construction, which would simplify the process. In particular, `element_nodes`
# needs to be shared around a lot.
@dataclass
class Model:
    """Result generated from LayeredBuilder. This does not need gmsh to be initialized."""

    nodes: np.ndarray
    elements: np.ndarray
    materials: np.ndarray
    boundaries: BoundarySpec
    _node_gmshtag_to_index_mapping: IndexMapping
    conforming_interfaces: ConformingInterfaces = field(
        default_factory=ConformingInterfaces
    )
    nonconforming_interfaces: NonconformingInterfaces = field(
        default_factory=NonconformingInterfaces
    )
    physical_groups: dict[str, PhysicalGroup] = field(default_factory=dict)

    def plot(self):
        """Displays, using matplotlib, the mesh corresponding to this model."""
        plot_model(self.nodes, self.elements)

    @staticmethod
    def union(model1: "Model", model2: "Model") -> "Model":
        """Joins two model instances. If two surfaces are to be exported into the same mesh,
        then call this.
        """
        nodemap1 = model1._node_gmshtag_to_index_mapping
        nodemap2 = model2._node_gmshtag_to_index_mapping
        nodemapu = JoinedIndexMapping(
            nodemap1,
            nodemap2,
        )
        nodes = np.empty((nodemapu.original_tag_list.size, 3))
        nodes[nodemapu.apply(nodemap1.original_tag_list)] = model1.nodes
        nodes[nodemapu.apply(nodemap2.original_tag_list)] = model2.nodes

        elem1_remapped = nodemapu.left_to_joined.apply(model1.elements)
        elem2_remapped = nodemapu.right_to_joined.apply(model2.elements)

        # equate elements if they have same nodes
        elemu, elemu_inds, elemu_inv = np.unique(
            np.concatenate([elem1_remapped, elem2_remapped], axis=0),
            return_index=True,
            return_inverse=True,
            return_counts=False,
            axis=0,
        )
        num_elems1 = elem1_remapped.shape[0]
        num_elems2 = elem2_remapped.shape[0]

        # we may want to consider updating JoinedIndexMapping to allow the
        # passing of index equivalencies. For now, hack it.
        elem_left_map = IndexMapping(elemu_inv[:num_elems1])
        elem_right_map = IndexMapping(elemu_inv[num_elems1:])
        elem_ind_remap = JoinedIndexMapping(elem_left_map, elem_right_map)

        # remap element ids in bdspec
        bdspec1 = model1.boundaries.remapped_elements(elem_ind_remap.left_to_joined)
        bdspec2 = model2.boundaries.remapped_elements(elem_ind_remap.right_to_joined)
        combined_bdries = BoundarySpec.union(bdspec1, bdspec2)

        # remap element and node ids in physical groups
        combined_physical_groups = {}
        for name in model1.physical_groups.keys() | model2.physical_groups.keys():
            pg1 = model1.physical_groups.get(name, NullPhysicalGroup(name))
            pg2 = model2.physical_groups.get(name, NullPhysicalGroup(name))

            pg1 = pg1.remap_indices(
                node_mapping=nodemapu.left_to_joined,
                element_mapping=elem_ind_remap.left_to_joined,
            )
            pg2 = pg2.remap_indices(
                node_mapping=nodemapu.right_to_joined,
                element_mapping=elem_ind_remap.right_to_joined,
            )

            combined_physical_groups[name] = UnionPhysicalGroup(name, pg1, pg2)

        # remap element ids in materials
        elem_materials = np.full(elemu.shape[0], 0, dtype=np.uint8)
        elem_materials[elemu_inv[:num_elems1]] = model1.materials
        elem_materials[elemu_inv[num_elems1:]] = model2.materials

        # remap element ids in conforming interfaces
        nci1 = dataclass_replace(
            model1.nonconforming_interfaces,
            elements_a=elem_ind_remap.left_to_joined.apply(
                model1.nonconforming_interfaces.elements_a
            ),
            elements_b=elem_ind_remap.left_to_joined.apply(
                model1.nonconforming_interfaces.elements_b
            ),
        )
        nci2 = dataclass_replace(
            model2.nonconforming_interfaces,
            elements_a=elem_ind_remap.right_to_joined.apply(
                model2.nonconforming_interfaces.elements_a
            ),
            elements_b=elem_ind_remap.right_to_joined.apply(
                model2.nonconforming_interfaces.elements_b
            ),
        )
        # match nonconforming interfaces
        ncis = NonconformingInterfaces.join(nci1, nci2)
        for a_entity in bdspec1.boundary_entity_spec.keys():
            for b_entity in bdspec2.boundary_entity_spec.keys():
                ncis.concatenate(
                    NonconformingInterfaces.between_entities(
                        combined_bdries, a_entity, b_entity, nodes, elemu
                    )
                )

        return Model(
            nodes=nodes,
            elements=elemu,
            materials=elem_materials,
            boundaries=combined_bdries,
            _node_gmshtag_to_index_mapping=nodemapu,
            conforming_interfaces=ConformingInterfaces.join(
                model1.conforming_interfaces,
                model2.conforming_interfaces,
                elemu_inv[:num_elems1],
                elemu_inv[num_elems1:],
            ),
            nonconforming_interfaces=ncis,
            physical_groups=combined_physical_groups,
        )

    @staticmethod
    def from_meshed_surface(
        surface: list[int] | int,
        gmsh: GmshContext,
        physical_group_captures: Iterable[str] | None = None,
    ) -> "Model":
        """Given an initialized mesh in gmsh, constructs a Model
        that stores the data of a surface or collection of
        surfaces with the given tag(s). The resulting Model is
        fully functional, even with a deactivated GmshContext.

        Args:
            surface (list[int] | int): gmsh surface tag(s)
            gmsh (GmshContext): the gmsh handshake to secure active environment.
            physical_group_captures (list[str] | None): a list of the physical groups to store.
        """
        if isinstance(surface, list):
            if len(surface) == 0:
                msg = "No surface tags specified. Cannot create a model."
                raise ValueError(msg)
            if len(surface) == 1:
                surface = surface[0]
        if isinstance(surface, int):
            # single surface, can be done manually
            meshnodes = gmsh.model.mesh.get_nodes()
            node_indexing = IndexMapping(meshnodes[0])
            node_locs = np.reshape(meshnodes[1], (-1, 3))

            # gmsh.model.mesh.get_elements gives elements of different types.
            # each of these captures a case:

            def on_mesh_tri(triname):
                msg = f"Cannot mesh {triname}. Must be quad."
                raise ValueError(msg)

            def on_MSH_QUA_4(elems, nodes):
                msg = (
                    "At the moment, 4-node quads have not been implemented. "
                    "Please mesh at order-2 by setting "
                    '`gmsh.option.setNumber("Mesh.ElementOrder", 2)`.'
                )
                raise NotImplementedError(msg)

            def on_MSH_QUA_8(elems, nodes):
                msg = (
                    "At the moment, 8-node quads have not been implemented. "
                    "The 9th node (center) must be placed manually."
                )
                raise NotImplementedError(msg)

            element_nodes_list = []
            layer_indices_list = []

            def on_MSH_QUA_9(elems, nodes):
                element_nodes_list.append(
                    node_indexing.apply(np.reshape(nodes, (-1, 9)))
                )
                layer_indices_list.append(
                    np.full(element_nodes_list[-1].shape[0], surface, dtype=np.uint8)
                )

            # https://gitlab.onelab.info/gmsh/gmsh/blob/master/src/common/GmshDefines.h
            gmsh.for_element_types_in_entity(
                2,
                surface,
                {
                    3: on_MSH_QUA_4,
                    16: on_MSH_QUA_8,
                    10: on_MSH_QUA_9,
                    2: lambda a, b: on_mesh_tri("3-node triangle"),
                    9: lambda a, b: on_mesh_tri("6-node 2nd order triangle"),
                    20: lambda a, b: on_mesh_tri(
                        "9-node 3rd order incomplete triangle"
                    ),
                    21: lambda a, b: on_mesh_tri("10-node 3rd order triangle"),
                    22: lambda a, b: on_mesh_tri(
                        "12-node 4th order incomplete triangle"
                    ),
                    23: lambda a, b: on_mesh_tri("15-node 4th order triangle"),
                    24: lambda a, b: on_mesh_tri(
                        "15-node 5th order incomplete triangle"
                    ),
                    25: lambda a, b: on_mesh_tri("21-node 5th order triangle"),
                },
            )
            element_nodes = np.concatenate(element_nodes_list, axis=0)
            layer_indices = np.concatenate(layer_indices_list, axis=0)

            boundary_entities = gmsh.model.get_boundary(
                [(2, surface)], oriented=False, recursive=False
            )

            physical_groups: dict[str, PhysicalGroup] = {}
            for name in (
                [] if physical_group_captures is None else physical_group_captures
            ):
                physical_groups[name] = physical_group_from_name(
                    gmsh,
                    node_indexing.invert(element_nodes),
                    node_indexing,
                    node_locs,
                    name,
                )

            return Model(
                nodes=node_locs,
                elements=element_nodes,
                materials=layer_indices,
                boundaries=BoundarySpec.from_model_entity(
                    gmsh,
                    [tag for dim, tag in boundary_entities if dim == 1],
                    node_indexing.invert(element_nodes),
                    node_indexing,
                    node_locs,
                ),
                _node_gmshtag_to_index_mapping=node_indexing,
                conforming_interfaces=ConformingInterfaces.from_element_node_matrix(
                    element_nodes
                ),
                physical_groups=physical_groups,
            )
        else:
            return Model.union(
                Model.from_meshed_surface(
                    surface[0], gmsh, physical_group_captures=physical_group_captures
                ),
                Model.from_meshed_surface(
                    surface[1:], gmsh, physical_group_captures=physical_group_captures
                ),
            )
