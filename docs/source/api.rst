=================
API Documentation
=================

All of the interface provided by `geocalc` is documented here.
You can read about the calculations performed by the functions in :doc:`theory`.

--------------
CPW geometries
--------------
.. automodule:: geocalc.cpw

Geometry Parameter conversion utilities
=======================================
.. autofunction:: geocalc.cpw.ab2SW
.. autofunction:: geocalc.cpw.SW2ab
.. autofunction:: geocalc.cpw.DW2ab
.. autofunction:: geocalc.cpw.dW2ab

Calculating CPW properties
==========================
.. autofunction:: geocalc.cpw.capacitance
.. autofunction:: geocalc.cpw.impedance
.. autofunction:: geocalc.cpw.characteristics

--------------
IDC geometries
--------------
.. automodule:: geocalc.idc

Geometry Parameter conversion utilities
=======================================
.. autofunction:: geocalc.idc.wg2etalmbd
.. autofunction:: geocalc.idc.etalmbd2wg

Calculating the IDC capacitance
===============================
.. autofunction:: geocalc.idc.capacitance

---------
Utilities
---------
.. autofunction:: geocalc.stack_layers
