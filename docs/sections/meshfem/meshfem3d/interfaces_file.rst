.. _interfaces_file:

Interfaces File
===============

Topography files are used to define the surface topography of the mesh. The
topography file is a simple text file that describes the topography of every
interface in the simulation domain.  For example the following topography file
describes a simple 1 layer model with a flat surface and a flat interface file.

.. code-block:: bash
    :caption: Example ``interfaces.txt``

    # number of interfaces
    1
    #
    # We describe each interface below, structured as a 2D-grid, with several parameters :
    # number of points along XI and ETA, minimal XI ETA coordinates
    # and spacing between points which must be constant.
    # Then the records contain the Z coordinates of the NXI x NETA points.
    #
    # interface number 1 (topography, top of the mesh)
    .true. 2 2 0.0 0.0 1000.0 1000.0
    path/to/interface1.txt
    #
    # for each layer, we give the number of spectral elements in the vertical direction
    #
    # layer number 1 (top layer)
    16

Here the simplest case of an interface file is show:

.. code-block:: bash
    :caption: Example ``interfaces1.txt``

    0
    0
    0
    0
