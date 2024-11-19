"""Utilities for coplanar stripline (CPS) geometries with infinite ground plane on a dielectric substrate of finite thickness.

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
           │             substrate                 │  │ h
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
           'capacitance', 'impedance']

def ab2SW(a, b):
    """Calculate `S` and `W` from `a` and `b`."""
    return 2 * a, b - a

def SW2ab(S, W):
    """Calculate `a` and `b` from `S` and `W`."""
    a = S / 2
    return a, a + W

def m_3(a, b):
    """Calculate the m_3 (aka k_3^2) parameter for `a` and `b`"""
    return (2 * a) / (a + b)

def m_4(a, b, h):
    """Calculate the m_4 (aka k_4^2) parameter for `a` and `b`"""
    return (np.exp((2 * np.pi * a) / h) - 1) / (np.exp((np.pi * (b + a)) / h) - 1)

def m_prime(m):
    """Calculate the m_prime (aka k_prime^2) parameter for a given `m` """
    return 1 - m ** 2

def capacitance(a, b):
    r"""Calculate the capacitance per unit length of the line in the absence of the dielectric substrate

    Arguments:
        a, b: Parameters describing the CPW geometry.

    Returns:
        capacitance C_0
    """
    return 2 * eps_0 * ellipk(m_prime(m_3(a,b))) / ellipk(m_3(a,b))

def Z_0_air(a,b):
    r"""Calculate the characteristic impedance in the absence of the dielectric substrate

    Arguments:
        a, b: Parameters describing the CPW geometry.

    Returns:
        capacitance Z_0(air)
    """
    return 60 * np.pi * ellipk(m_prime(m_3(a,b))) / ellipk(m_3(a,b))

def epsilon_eff(a, b, h, eps_r):
    r"""Calculate the effective dielectric constant

    Arguments:
        a, b: Parameters describing the CPW geometry.
        h: Height of the substrate.
        eps_r: Relative permittivity of the substrate.

    Returns:
        dielectric constant epsilon_eff
    """
    return 1 + ((eps_r - 1) / 2) * (ellipk(m_3(a,b)) / ellipk(m_prime(m_3(a,b)))) * (ellipk(m_prime(m_4(a,b))) / ellipk(m_4(a,b)))

def impedance(a, b, h, eps_r):
    r"""Calculate the characteristic impedance

    Arguments:
        a, b: Parameters describing the CPW geometry.
        h: Height of the substrate.
        eps_r: Relative permittivity of the substrate.

    Returns:
        characteristic impedance Z_0
    """
    eps_eff = epsilon_eff(a, b, h, eps_r)
    return ((60 * np.pi) / np.sqrt(eps_eff)) * (ellipk(m_prime(m_3(a,b))) / ellipk(m_3(a,b)))
