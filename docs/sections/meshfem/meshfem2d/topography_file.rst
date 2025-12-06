
.. _topography_file:

Topography File
+++++++++++++++

Topography files are used to define the surface topography of the mesh. The
topography file is a simple text file that describes the topography of every
interface in the simulation domain.  For example the following topography file
describes a simple 2 layer model with a flat surface and a flat interface
between the two layers:

.. code-block:: bash
    :caption: Example ``topography.dat``

    #
    # number of interfaces
    #
    3
    #
    # for each interface below, we give the number of points and then x,z for each point
    #
    #
    # interface number 1 (bottom of the mesh)
    #
    2
    0 0
    6400 0
    #
    # interface number 2 (ocean bottom)
    #
    2
    0 2400
    6400 2400
    #
    # interface number 3 (topography, top of the mesh)
    #
    2
    0 4800
    6400 4800
    #
    # for each layer, we give the number of spectral elements in the vertical direction
    #
    #
    # layer number 1 (bottom layer)
    #
    54
    #
    # layer number 2 (top layer)
    #
    54
