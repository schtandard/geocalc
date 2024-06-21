"""Utilities for interdigital capacitor (IDC) geometries.

The functions in this module accept floats or numpy arrays of equal shape for
geometry parameters and return an object (or objects) of the same type and shape.

The meanings of the geometry parameter names are shown in this drawing (top view)::

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
from scipy.constants import epsilon_0 as eps_0
from scipy.special import ellipk, ellipj
import mpmath
jtheta = np.vectorize(mpmath.jtheta, otypes=['float64'], excluded={0, 1})
import typing
from typing import Callable, Union, Literal
from numpy.typing import ArrayLike
from ._util import LayerSpec, _layerspec, _ppc_C, _ppcedge_C, _Cohn_C

__all__ = ['wg2etalmbd', 'etalmbd2wg',
           'capacitance']

ThicknessCorrectionMethod = Literal[
    'ppc', 'ppcedge', 'Cohn',
]
thickness_correction_method = 'ppc'

class ThicknessCorrectionScope:
    """A context manager for temporarily changing thickness_correction_method.

    The method is changed upon entering and changed back upon exiting.

    Arguments:
        method: Method to change to.

    """

    def __init__(self, method: ThicknessCorrectionMethod):
        if method not in typing.get_args(ThicknessCorrectionMethod):
            raise ValueError(f"Unknown t correction method {method!r}.")
        self._method = method
        self._outer_method = None

    def __enter__(self):
        global thickness_correction_method
        self._outer_method = thickness_correction_method
        thickness_correction_method = self._method

    def __exit__(self, type, value, traceback):
        global thickness_correction_method
        thickness_correction_method = self._outer_method

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
    fallback = np.ones_like(r) * 4 * eta / (1 + eta)**2
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
                        eeta: ArrayLike, rr: ArrayLike, eeps_r: ArrayLike) -> np.array:
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
    C = np.sum(Cvac * eeps_r, axis=1)
    return C

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
    # eta and lmbd should have the same shape, as should h and eps_r.
    eeta, hh = np.meshgrid(eta, h, indexing='ij', sparse=True)
    llmbd, eeps_r = np.meshgrid(lmbd, eps_r, indexing='ij', sparse=True)
    rr = hh / llmbd
    Ci = _capacitance_auxaux(mi_interface, mi_cover, eeta, rr, eeps_r)
    Ce = _capacitance_auxaux(me_interface, me_cover, eeta, rr, eeps_r)
    # Reshape the results to make sure that scalar inputs lead to scalar outputs.
    return Ci.reshape(np.shape(eta)), Ce.reshape(np.shape(eta))

def _tcorr_none_capacitance(eta, lmbd, below, above):
    lower_Ci, lower_Ce = _capacitance_aux(eta, lmbd, *zip(*_layerspec(below)))
    upper_Ci, upper_Ce = _capacitance_aux(eta, lmbd, *zip(*_layerspec(above)))
    Ci = lower_Ci + upper_Ci
    Ce = lower_Ce + upper_Ce
    return Ci, Ce

def _tcorr_ppc_C_aux(g, t, Cfun, epsr_above):
    return epsr_above * Cfun(t / g)

def _tcorr_ppc_C(g, above, t):
    first_h, first_epsr = _layerspec(above)[0]
    if np.any(first_h < t):
        raise ValueError("Conductor thicknesses larger than the first"
                         " dielectric layer are not currently supported.")
    return _tcorr_ppc_C_aux(g, t, _ppc_C, first_epsr)

def _tcorr_ppc_capacitance_aux(eta, lmbd, below, above, ppc_C):
    lower_Ci, lower_Ce = _capacitance_aux(eta, lmbd, *zip(*_layerspec(below)))
    upper_Ci, upper_Ce = _capacitance_aux(eta, lmbd, *zip(*_layerspec(above)))
    Ci = lower_Ci + upper_Ci + 2 * ppc_C
    Ce = lower_Ce + upper_Ce + 2 * ppc_C
    return Ci, Ce

def _tcorr_ppc_capacitance(w, g, eta, lmbd, below, above, t):
    return _tcorr_ppc_capacitance_aux(eta, lmbd, below, above,
                                      _tcorr_ppc_C(g, above, t))

def _tcorr_ppcedge_C_aux(g, t, Cfun, epsr_below, epsr_above):
    lmbd = t / g
    Cvac = Cfun(lmbd)
    Cvac_ppc = _ppc_C(lmbd)
    single_edge_contribution = (Cvac - Cvac_ppc) / 2
    return epsr_above * Cvac + (epsr_below - epsr_above) * single_edge_contribution

def _tcorr_ppcedge_C(g, below, above, t):
    first_h_below, first_epsr_below = _layerspec(below)[0]
    if np.any(first_h_below < 2 * g):
        raise ValueError("Substrate thicknesses smaller than twice the gap size"
                         " are not currently supported.")
    first_h_above, first_epsr_above = _layerspec(above)[0]
    if np.any(first_h_above < t + 2 * g):
        raise ValueError("Conductor thicknesses larger than the first"
                         " dielectric layer minus twice the gap size"
                         " are not currently supported.")
    return _tcorr_ppcedge_C_aux(g, t, _ppcedge_C, first_epsr_below, first_epsr_above)

def _tcorr_ppcedge_capacitance(w, g, eta, lmbd, below, above, t):
    return _tcorr_ppc_capacitance_aux(eta, lmbd, below, above,
                                      _tcorr_ppcedge_C(g, below, above, t))

def _tcorr_Cohn_C(g, below, above, t):
    first_h_below, first_epsr_below = _layerspec(below)[0]
    if np.any(first_h_below < 2 * g):
        raise ValueError("Substrate thicknesses smaller than twice the gap size"
                         " are not currently supported.")
    first_h_above, first_epsr_above = _layerspec(above)[0]
    if np.any(first_h_above < t + 2 * g):
        raise ValueError("Conductor thicknesses larger than the first"
                         " dielectric layer minus twice the gap size"
                         " are not currently supported.")
    return _tcorr_ppcedge_C_aux(g, t, _Cohn_C, first_epsr_below, first_epsr_above)

def _tcorr_Cohn_capacitance(w, g, eta, lmbd, below, above, t):
    return _tcorr_ppc_capacitance_aux(eta, lmbd, below, above,
                                      _tcorr_Cohn_C(g, below, above, t))

ThicknessCorrectionMethod = Literal[
    'ppc', 'ppcedge', 'Cohn',
]
def set_thickess_correction_method(method: ThicknessCorrectionMethod):
    """Set the thickness correction method to use.

    When you calculate an IDC capacitance with non-zero metallization thickness `t`,
    the method (last) set up using this function will be used.

    Arguments:
        method: Name of the method to use.

    Raises:
        ValueError: When an unknown method name is supplied.

    """
    global _tcorr_capacitance
    if method == 'ppc':
        _tcorr_capacitance = _tcorr_ppc_capacitance
    elif method == 'ppcedge':
        _tcorr_capacitance = _tcorr_ppcedge_capacitance
    elif method == 'Cohn':
        _tcorr_capacitance = _tcorr_Cohn_capacitance
    else:
        raise ValueError(f"Unknown t correction method {method!r}.")

_tcorr_capacitance = None
set_thickess_correction_method('ppc')

def _capacitance(n: ArrayLike, w: ArrayLike, g: ArrayLike, l: ArrayLike = 1,
                 below: LayerSpec = None,
                 above: LayerSpec = None,
                 t: ArrayLike = 0) -> Union[float, np.array]:
    # Make sure everything's an array.
    n = np.asarray(n)
    w = np.asarray(w)
    g = np.asarray(g)
    eta, lmbd = wg2etalmbd(w, g)
    l = np.asarray(l)
    t = np.asarray(t)
    # Now do the calculation.
    if np.any(t):
        if thickness_correction_method == 'ppc':
            _capfun = _tcorr_ppc_capacitance
        elif thickness_correction_method == 'ppcedge':
            _capfun = _tcorr_ppcedge_capacitance
        elif thickness_correction_method == 'Cohn':
            _capfun = _tcorr_Cohn_capacitance
        else:
            raise ValueError(f"Unknown t correction method {thickness_correction_method!r}.")
        Ci, Ce = _capfun(w, g, eta, lmbd, below, above, t)
    else:
        Ci, Ce = _tcorr_none_capacitance(eta, lmbd, below, above)
    C = (n - 3) / 2 * Ci + 2 * (Ci * Ce) / (Ci + Ce)
    return l * C

def capacitance(n: ArrayLike, w: ArrayLike, g: ArrayLike, l: ArrayLike = 1,
                below: LayerSpec = None,
                above: LayerSpec = None, *,
                t: ArrayLike = 0) -> Union[float, np.array]:
    """Compute the capacitance.

    You can use :func:`.stack_layers` to create values for `below`
    and `above`.

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
        below: The dielectric layers below the IDC.
            Each layer should be given as a tuple containing the height at which
            the layer ends and the relative permittivity of that layer. The last
            layer is considered to be terminated by a metal cover. The default
            corresponds to infinite vacuum.
        above: The dielectric layers above the IDC.
            (In the same format as `layers_below`.)
        t: Metallization thickness of the IDC.

    Returns:
        The IDC's capacitance in the same shape as `n`, `w`, `g` and `l`.

    """
    return _capacitance(n, w, g, l, below, above, t)
