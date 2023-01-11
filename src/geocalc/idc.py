"""Utilities for interdigital capacitor (IDC) geometries.

The functions in this module accept floats or numpy arrays of equal shape for
geometry parameters and return an object (or objects) of the same type and shape.

The meanings of the geometry parameter names are shown in this drawing::

                ├──────────── l ────────────┤
            ┌───────────────────────────────┐ ┌─┐
            │ ┌─────────────────────────────┘ │ │
            │ │ ┌─────────────────────────────┘ │ ┬ w   ┬
            │ │ └─────────────────────────────┐ │ ┴     │
            │ └─────────────────────────────┐ │ │       │ lmbd = 2 * (w + g)
            │ ┌─────────────────────────────┘ │ │ ┬ g   │
            │ │ ┌─────────────────────────────┘ │ ┴     ┴
            │ │ └─────────────────────────────┐ │
            │ └─────────────────────────────┐ │ │         eta = w / (w + g)
            │ ┌─────────────────────────────┘ │ │
            │ │ ┌─────────────────────────────┘ │
            └─┘ └───────────────────────────────┘

"""

# This __future__ import will become obsolete in Python 3.10.
from __future__ import annotations
import numpy as np
from scipy.special import ellipk, ellipj
from scipy.constants import epsilon_0 as eps_0
import mpmath
jtheta = np.vectorize(mpmath.jtheta, otypes=['float64'], excluded={0, 1})
from typing import Union, Optional, Iterable, Callable
from numpy.typing import ArrayLike

__all__ = ['wg2etalmbd', 'etalmbd2wg',
           'capacitance']

def wg2etalmbd(w, g):
    """Calculate `lmbd` and `eta` from `w` and `g`."""
    tmp = w + g
    return w / tmp, 2 * tmp

def etalmbd2wg(eta, lmbd):
    """Calculate `w` and `g` from `lmbd` and `eta`."""
    tmp = lmbd * eta
    return tmp / 2, (lmbd - tmp) / 2

def mi_interface(eta: ArrayLike, r: ArrayLike) -> np.array:
    # Consider special cases.
    # For r = 0 we want m = 0, for r = np.inf we want m = sin(pi / 2 * eta)**2.
    good_r = np.array((0 < r) & (r < np.inf))
    fallback = np.ones_like(r) * np.sin(np.pi / 2 * eta)**2
    q = np.exp(-4 * np.pi * r, where=good_r, out=np.zeros_like(fallback))
    k = np.divide(jtheta(2, 0, q), jtheta(3, 0, q),
                  where=q != 0, out=np.zeros_like(q))**2
    m = k**2
    t2 = ellipj(ellipk(m) * eta, m)[0]
    t4 = np.divide(1, k, where=k != 0, out=np.zeros_like(k))
    return np.divide(t2**2 * (t4**2 - 1), t4**2 - t2**2,
                     where=t4 != 0, out=fallback)

def me_interface(eta: ArrayLike, r: ArrayLike) -> np.array:
    # Consider special cases.
    # For r = 0 we want m = 0, for r = np.inf we want m = 2 * sqrt(eta) / (1 + eta).
    good_r = np.array((0 < r) & (r < np.inf))
    fallback = np.ones_like(r) * 2 * eta / (1 + eta)**2
    t3 = np.cosh(np.divide(np.pi * (1 - eta),  (8 * r),
                           where=good_r, out=np.zeros_like(fallback)))
    t4 = np.cosh(np.divide(np.pi * (1 + eta),  (8 * r),
                           where=good_r, out=np.zeros_like(fallback)))
    return np.divide(t4**2 - t3**2, t3**2 * (t4**2 - 1),
                     where=good_r, out=fallback)

def _m_cover(mintfunc: Callable, eta: ArrayLike, r: ArrayLike) -> np.array:
    if np.any(r < np.inf):
        raise ValueError("IDCs with a metal cover are not implemented!")
    return mintfunc(eta, r)

def mi_cover(eta: ArrayLike, r: ArrayLike) -> np.array:
    return _m_cover(mi_interface, eta, r)

def me_cover(eta: ArrayLike, r: ArrayLike) -> np.array:
    return _m_cover(me_interface, eta, r)

def fieldvol(m: np.array) -> np.array:
    """Calculate the field volume :math:`v(h)` from :math:`m = m(h)`.

    Arguments:
        m: The :math:`m(h)` value.

    Returns:
        The field volume :math:`v(h)`.

    """
    return ellipk(m) / ellipk(1 - m)

def _capacitance_auxaux(mintfunc: Callable, mcovfunc: Callable,
                        eta: ArrayLike, r: ArrayLike, eps_r: ArrayLike) -> np.array:
    eeta, rr = np.meshgrid(eta, r, indexing='ij', sparse=True)
    # Calculate the m(h).
    m = np.concatenate((np.zeros_like(eeta),
                        mintfunc(eeta, rr[:, :-1]),
                        mcovfunc(eeta, rr[:, -1:])),
                       axis=1)
    # Calculate the accumulated vacuum capacitances.
    Cvac_accum = eps_0 * fieldvol(m)
    # Calculate the layer vacuum capacitances.
    Cvac = Cvac_accum[:, 1:] - Cvac_accum[:, :-1]
    # Calculate the capacitance.
    C = np.sum(Cvac * eps_r, axis=1)
    return C.reshape(np.shape(eta))

def _capacitance_aux(eta: ArrayLike, lmbd: ArrayLike, h: ArrayLike, eps_r: ArrayLike) -> tuple[np.array, np.array]:
    """Compute the half-plane partial specific capacitances.

    The results of this computation for both half planes can then be used to
    calculate the capacitance of an IDC.

    Arguments:
        eta, lmbd: Parameters describing the IDC geometry.
        h: The dielectric interface heights.
            The last value is interpreted to mark a metal cover.
        eps_r: The relative permittivities of the dielectics.
            Should be the same shape as `h`.

    Returns:
        The partial specific capacitances for internal and external finger gaps
        of the IDC in the half-plane.

    """
    r = np.array(h) / np.array(lmbd)
    Ci = _capacitance_auxaux(mi_interface, mi_cover, eta, r, eps_r)
    Ce = _capacitance_auxaux(me_interface, me_cover, eta, r, eps_r)
    return Ci, Ce

def capacitance(n: ArrayLike, w: ArrayLike, g: ArrayLike, l: ArrayLike = 1,
                layers_beneath: Optional[Iterable[tuple]] = [(np.inf, 1)],
                layers_above: Optional[Iterable[tuple]] = [(np.inf, 1)]) -> Union[float, np.array]:
    """Compute the capacitance.

    You can use :func:`.stack_layers` to create values for `layers_beneath`
    and `layers_above`.

    First the specific capacitance is calculated, then it is multiplied by `l`
    and returned. The unit of the specific capacitance will be F / m irrespective
    of the unit of the input lengths (as long as they all have the same unit).
    Only `l` must be given in m for a valid return value in F.

    Giving a value of `1` for `l` (the default) is equivalent to requesting the
    specific capacitance in F / m.

    Note that capacitances for IDCs with metal covers cannot currently be calculated.

    Arguments:
        n: Number of fingers.
        w: Width of each finger.
        g: Width of each gap between fingers.
        l: Length of the overlap of the fingers.
            Must be given in m.
        layers_beneath: The dielectric layers beneath the IDC.
            Each layer should be given as a tuple containing the height at which
            the layer ends and the relative permittivity of that layer. The last
            layer is considered to be terminated by a metal cover. The default
            corresponds to infinite vacuum.
        layers_above: The dielectric layers above the IDC.
            (In the same format as `layers_beneath`.)

    Returns:
        The IDC's capacitance in the same shape as `n`, `w`, `g` and `l`.

    """
    # Make sure everything's an array.
    n = np.array(n)
    eta, lmbd = wg2etalmbd(np.array(w), np.array(g))
    l = np.array(l)
    # Now do the calculation.
    lower_Ci, lower_Ce = _capacitance_aux(eta, lmbd, *zip(*layers_beneath))
    upper_Ci, upper_Ce = _capacitance_aux(eta, lmbd, *zip(*layers_above))
    Ci = lower_Ci + upper_Ci
    Ce = lower_Ce + upper_Ce
    C = (n - 3) / 2 * Ci + 2 * (Ci * Ce) / (Ci + Ce)
    return l * C
