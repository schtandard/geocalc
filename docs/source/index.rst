=======
geocalc
=======

`geocalc` is a Python 3 package for calculating physical parameters for device geometries.

.. warning::

   `geocalc` is currently still in version 0.
   The interface may change in incompatible ways between versions, without warning.
   Most of it should be fairly stable but be aware of this possibility and make a note of the exact version you are using for critical applications.

.. warning::

   In particular, the thickness correction functionality in the :mod:`cpw` module and the entire :mod:`cps` module are considered experimental.
   Their documentation is not very good or complete and they in particular are subject to changes.

Easily calculate the impedance of a given CPW structure

.. code-block:: pycon

   >>> from geocalc import cpw
   >>> import numpy as np
   >>> below = [(500, 11.7), (1500, 1)]
   >>> cpw.impedance(500, 700, below)
   39.9064460552812

or find the right values for a given impedance

.. code-block:: pycon

   >>> from geocalc import cpw
   >>> import numpy as np
   >>> from scipy.optimize import root_scalar
   >>> below = [(500, 11.7), (1500, 1)]
   >>> fun = lambda S: cpw.impedance(*cpw.SW2ab(S, 200), below) - 50
   >>> sol = root_scalar(fun, bracket=[1, 1000])
   >>> sol.root
   383.8928758406656

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   api
   theory
   genindex
