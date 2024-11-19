"""Utilities for coplanar stripline (CPSL) geometries.

The functions in this module accept floats or numpy arrays of equal shape for
geometry parameters and return an object (or objects) of the same type and shape.

The meanings of the geometry parameter names are shown in this drawing (crossection view)::

                 ├─ a ─┤
                 ├───── b ─────┤
           ┌───────────┐       ┌───────────────────┐  ┬
           │           │       │                   │  │
           │           │       │                   │  │
           │           │       │      ground       │  │ t
           │           │       │                   │  │
           │           │       │                   │  │
           └───────────┘       └───────────────────┘  ┴
           ├──── S ────┤
                       ├── W ──┤

           ┌───────────────────────────────────────┐  ┬
           │             substrate                    │ h
           └───────────────────────────────────────┘  ┴


Source of the equations:
Title:
Coplanar Waveguide Circuits, Components, and Systems
Chapter:
6.2.4 Coplanar Stripline with Infinitely Wide Ground Plane on a Dielectric
Substrate of Finite Thickness
Page: 160+161

Author:
RAINEE N. SIMONS
NASA Glenn Research Center
Cleveland, Ohio

Publisher:
WILEY-INTERSCIENCE
A JOHN WILEY & SONS, INC., PUBLICATION
NEW YORK · CHICHESTER · WEINHEIM · BRISBANE · SINGAPORE · TORONTO
"""

import numpy as np
from scipy.special import ellipk
from scipy.constants import epsilon_0 as eps_0


__all__ = ['ab2SW', 'SW2ab',
           'C_0', 'Z_0_air', 'Z_0']

def ab2SW(a, b):
    """Calculate `S` and `W` from `a` and `b`."""
    return 2 * a, b - a

def SW2ab(S, W):
    """Calculate `a` and `b` from `S` and `W`."""
    a = S / 2
    return a, a + W

def C_0(a, b):
    r"""Calculate the capacitance per unit length of the line in the absence of the dielectric substrate

    Arguments:
        a, b: Parameters describing the CPW geometry.

    Returns:
        capacitance C_0
    """
    k_3 = np.sqrt((2 * a) / (a + b))
    k_3p = np.sqrt(1 - k_3 ** 2)
    return 2 * eps_0 * ellipk(k_3p) / ellipk(k_3)

def Z_0_air(a,b):
    r"""Calculate the characteristic impedance in the absence of the dielectric substrate

    Arguments:
        a, b: Parameters describing the CPW geometry.

    Returns:
        capacitance Z_0(air)
    """
    k_3 = np.sqrt((2 * a) / (a + b))
    k_3p = np.sqrt(1 - k_3 ** 2)
    return 60 * np.pi * ellipk(k_3p) / ellipk(k_3)

def epsilon_eff(a, b, h, eps_r):
    r"""Calculate the effective dielectric constant

    Arguments:
        a, b: Parameters describing the CPW geometry.
        h: Height of the substrate.
        eps_r: Relative permittivity of the substrate.

    Returns:
        dielectric constant epsilon_eff
    """
    k_3 = np.sqrt((2 * a) / (a + b))
    k_3p = np.sqrt(1 - k_3 ** 2)

    k_4 = np.sqrt((np.exp((2 * np.pi * a) / h) - 1) / (np.exp((np.pi * (b + a)) / h) - 1))
    k_4p = np.sqrt(1 - k_4 ** 2)

    return 1 + ((eps_r - 1) / 2) * (ellipk(k_3) / ellipk(k_3p)) * (ellipk(k_4p) / ellipk(k_4))

def Z_0(a, b, h, eps_r):
    r"""Calculate the characteristic impedance

    Arguments:
        a, b: Parameters describing the CPW geometry.
        h: Height of the substrate.
        eps_r: Relative permittivity of the substrate.

    Returns:
        characteristic impedance Z_0
    """
    k_3 = np.sqrt((2 * a) / (a + b))
    k_3p = np.sqrt(1 - k_3 ** 2)
    eps_eff = epsilon_eff(a, b, h, eps_r)
    return ((60 * np.pi) / np.sqrt(eps_eff)) * (ellipk(k_3p) / ellipk(k_3))