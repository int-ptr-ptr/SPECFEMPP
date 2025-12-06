.. _MESHFEM3D_Parameter_documentation:

MESHFEM3D Parameter Documentation
=================================

SPECFEM++ uses a modified version of the mesher provided by the original
`SPECFEM3D <https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/>`_
code. The modifications isolate only the necessary parameters for the meshing
process and remove those needed by the solver from the original ``Par_File``.
We document only the parameters that are used by the meshing process.
However you should also refer to the `SPECFEM3D documentation
<https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/>`_ for a more
in-depth description of the mesher.

To define the meshing parameters, you will need to create a ``Mesh_Par_File``.
This is a simple text file that contains the parameters you wish to use. The
parameters are defined in the following format:

.. code-block:: bash

    parameter_name = parameter_value

Parameter Description
---------------------

.. toctree::
    :maxdepth: 1

    mesh_par_file
    interfaces_file
