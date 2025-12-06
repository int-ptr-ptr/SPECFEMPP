
.. _assembly_boundary:

``specfem::assembly::boundaries``
=================================

.. doxygenclass:: specfem::assembly::boundaries
    :members:

Dimension-Specific Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: specfem::assembly::boundaries< specfem::dimension::type::dim2 >
    :members:

Data Access Functions
^^^^^^^^^^^^^^^^^^^^^^

.. doxygengroup:: BoundaryConditionDataAccess
    :content-only:

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    acoustic_free_surface
    stacey
