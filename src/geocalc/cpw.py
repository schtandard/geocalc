"""Utilities for coplanar waveguide (CPW) geometries.

The functions in this module accept floats or numpy arrays of equal shape for
geometry parameters and return an object (or objects) of the same type and shape.

The meanings of the geometry parameter names are shown in this drawing (crossection view)::

                                      ├─ a ─┤
                                      ├───── b ─────┤
                                      ├─────────────── c ───────────────┤
                                      ├─── d ───┤
    ┌───────────────────┐       ┌───────────┐       ┌───────────────────┐  ┬
    │                   │       │           │       │                   │  │
    │                   │       │           │       │                   │  │
    │      ground       │       │           │       │      ground       │  │ t
    │                   │       │           │       │                   │  │
    │                   │       │           │       │                   │  │
    └───────────────────┘       └───────────┘       └───────────────────┘  ┴
                                ├──── S ────┤       ├──────── X ────────┤
                                            ├── W ──┤
                            ├──────── D ────────┤

The Parameters `c`, `X` and `t` are currrently not considered by any of the functions.

"""

from __future__ import annotations
import numpy as np
from scipy.constants import epsilon_0 as eps_0, c as c_vac
from scipy.special import ellipk
import typing
from typing import Union, Literal
from numpy.typing import ArrayLike
from ._util import LayerSpec, _layerspec, _ppc_C, _ppcedge_C, _Cohn_C

__all__ = ['ab2SW', 'SW2ab', 'DW2ab', 'dW2ab',
           'capacitance', 'impedance', 'characteristics']

ThicknessCorrectionMethod = Literal[
    'Gupta', 'Garg', 'Ashesh',
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

def ab2SW(a, b):
    """Calculate `S` and `W` from `a` and `b`."""
    return 2 * a, b - a

def SW2ab(S, W):
    """Calculate `a` and `b` from `S` and `W`."""
    a = S / 2
    return a, a + W

def DW2ab(D, W):
    """Calculate `a` and `b` from `D` and `W`."""
    return (D - W) / 2, (D + W) / 2

def dW2ab(d, W):
    """Calculate `a` and `b` from `d` and `W`."""
    return d - W / 2, d + W / 2

def _m_general(hyp_fun: np.ufunc, a: ArrayLike, b: ArrayLike, h: ArrayLike) -> np.array:
    r"""Calculate a certain :math:`m(h)`.

    The calculation is

    .. math::

        m(h) = \biggl( \frac{ f\bigl( \frac{\pi a}{2 h} \bigr) }{ f\bigl( \frac{\pi b}{2 h} \bigr) } \biggr)^2

    where :math:`f` is the given hyperolic function.

    Arguments:
        hyp_fun: The hyperbolic function to use.
            Should be either :func:`np.sinh` or :func:`np.tanh`.
        a, b: Parameters describing the CPW geometry.
        h: The heights of dielectric surfaces to consider.

    Returns:
        The :math:`m` values.

    """
    aa, hh = np.meshgrid(a, h, indexing='ij', sparse=True)
    bb, _  = np.meshgrid(b, h, indexing='ij', sparse=True)
    # Consider special cases.
    # For h = 0 we want m = 0, for h = np.inf we want m = (a / b)**2.
    good_hh = np.array((0 < hh) & (hh < np.inf))
    fallback = np.array(aa / bb * np.isinf(hh))
    kk = np.divide(hyp_fun(np.divide(np.pi * aa, 2 * hh, where=good_hh)),
                   hyp_fun(np.divide(np.pi * bb, 2 * hh, where=good_hh)),
                   where=good_hh,
                   out=fallback)
    return kk**2

def m_interface(a: ArrayLike, b: ArrayLike, h: ArrayLike) -> np.array:
    r"""Calculate :math:`m(h)` for dielectric interfaces.

    The calculation is

    .. math::

        m(h) = \biggl( \frac{ \sinh\bigl( \frac{\pi a}{2 h} \bigr) }{ \sinh\bigl( \frac{\pi b}{2 h} \bigr) } \biggr)^2 .

    Arguments:
        a, b: Parameters describing the CPW geometry.
        h: The heights of dielectric interfaces.

    Returns:
        The :math:`m` values.

    """
    return _m_general(np.sinh, a, b, h)

def m_cover(a: ArrayLike, b: ArrayLike, h: ArrayLike) -> np.array:
    r"""Calculate :math:`m(h)` for metal covers.

    The calculation is

    .. math::

        m(h) = \biggl( \frac{ \tanh\bigl( \frac{\pi a}{2 h} \bigr) }{ \tanh\bigl( \frac{\pi b}{2 h} \bigr) } \biggr)^2 .

    Arguments:
        a, b: Parameters describing the CPW geometry.
        h: The heights of dielectric surfaces to consider.

    Returns:
        The :math:`m` values.

    """
    return _m_general(np.tanh, a, b, h)

def fieldvol(m: np.array) -> np.array:
    """Calculate the field volume :math:`v(h)` from :math:`m = m(h)`.

    Arguments:
        m: The :math:`m(h)` value.

    Returns:
        The field volume :math:`v(h)`.

    """
    return 2 * ellipk(m) / ellipk(1 - m)

def _capacitance_aux(a: ArrayLike, b: ArrayLike, h: ArrayLike, eps_r: ArrayLike) -> tuple[np.array, np.array]:
    """Compute the half-plane specific capacitance and its vacuum value.

    The result of this computation for both half planes can then be used to
    calculate the impedance of a CPW.

    Arguments:
        a, b: Parameters describing the CPW geometry.
        h: The dielectric interface heights.
            The last value is interpreted to mark a metal cover.
        eps_r: The relative permittivities of the dielectics.
            Should be the same shape as `h`.

    Returns:
        The specific capacitance contributions of the dielectic layers and the
        total specific vacuum capacitance of the half space.

    """
    # Calculate the m(h).
    m = np.concatenate((np.zeros((np.size(a), 1)),
                        m_interface(a, b, h[:-1]),
                        m_cover(a, b, h[-1:])),
                       axis=1)
    # Calculate the accumulated vacuum capacitances.
    Cvac_accum = eps_0 * fieldvol(m)
    # Remember the total capacitance.
    Cvac_tot = Cvac_accum[:, -1]
    # Calculate the layer vacuum capacitances.
    Cvac = Cvac_accum[:, 1:] - Cvac_accum[:, :-1]
    # Calculate the capacitance.
    C = np.sum(Cvac * eps_r, axis=1)
    # Reshape the results to make sure that scalar inputs lead to scalar outputs.
    return C.reshape(np.shape(a)), Cvac_tot.reshape(np.shape(a))

def _tcorr_none_capacitance(a, b, below, above):
    lower_C, lower_Cvac = _capacitance_aux(a, b, *zip(*_layerspec(below)))
    upper_C, upper_Cvac = _capacitance_aux(a, b, *zip(*_layerspec(above)))
    C = lower_C + upper_C
    Cvac = lower_Cvac + upper_Cvac
    return C, Cvac

def _tcorr_Gupta_Delta(a, b, t):
    # Microstrip Lines and Slotlines (Kuldip Gupta et al., 1996 Artech House)
    # Make sure t has the same shape as a.
    t = np.ones_like(a) * t
    good_t = t != 0
    arg = np.divide(8 * np.pi * a, t,
                    where=good_t,
                    out=np.ones_like(t))
    return 1.25 * t / np.pi * (1 + np.log(arg))

def _tcorr_Gupta_epseff(a, b, t, epseff):
    m = (a / b)**2
    ratio = 0.7 * t / (b - a)
    return epseff - (epseff - 1) * ratio / (ellipk(m) / ellipk(1 - m) + ratio)

def _tcorr_Gupta_capacitance(a, b, below, above, t):
    C_raw, Cvac_raw = _tcorr_none_capacitance(a, b, below, above)
    a_raw, b_raw = a, b
    halfDelta = _tcorr_Gupta_Delta(a, b, t) / 2
    a = a + halfDelta
    b = b - halfDelta
    C, Cvac = _tcorr_none_capacitance(a, b, below, above)
    epseff = _tcorr_Gupta_epseff(a_raw, b_raw, t, C_raw / Cvac_raw)
    C = epseff * Cvac
    return C, Cvac

def _tcorr_Garg_Delta(a, b, t):
    # Microstrip Lines and Slotlines (Ramesh Garg et al., 2013 Artech House)
    # Garg cites ashesh2007 (see below) but writes '53' in the second fitting
    # parameter has instead of '35'. This is probably a typo, but an
    # inconsequential one.
    # Make sure t has the same shape as a.
    t = np.ones_like(a) * t
    good_t = t != 0
    arg1 = np.divide(b, t, where=good_t, out=np.zeros_like(t))
    arg2 = np.divide(4 * np.pi * (b - a), t, where=good_t, out=np.ones_like(t))
    Delta = t / np.pi * (4.098 + (0.9536 + 3.864e-3 * arg1) * np.log(arg2))
    # Make sure Delta is an array so the following assigment also works with scalars.
    Delta = np.asarray(Delta)
    Delta[~good_t] = np.inf
    return Delta

def _tcorr_Garg_capacitance_aux(a, b, below, above, halfDelta):
    # Use the effective a, b only for the upper half.
    lower_C, lower_Cvac = _capacitance_aux(a, b, *zip(*_layerspec(below)))
    a = a + halfDelta
    b = b - halfDelta
    upper_C, upper_Cvac = _capacitance_aux(a, b, *zip(*_layerspec(above)))
    C = lower_C + upper_C
    Cvac = lower_Cvac + upper_Cvac
    return C, Cvac

def _tcorr_Garg_capacitance(a, b, below, above, t):
    return _tcorr_Garg_capacitance_aux(a, b, below, above, _tcorr_Garg_Delta(a, b, t) / 2)

def _tcorr_Ashesh_Delta(a, b, t):
    # Analysis and Design of Symmetric Coplanar Lines with Thick Conductors
    # (C. B. Ashesh, 2007, PhD thesis, Indian Institute of Technology, Kharagpur)
    good_t = t != 0
    arg1 = np.divide(b, t, where=good_t, out=np.zeros_like(t))
    arg2 = np.divide(4 * np.pi * (b - a), t, where=good_t, out=np.ones_like(t))
    Delta = t / np.pi * (4.098 + (0.9356 + 3.864e-3 * arg1) * np.log(arg2))
    # Make sure Delta is an array so the following assigment also works with scalars.
    Delta = np.asarray(Delta)
    Delta[~good_t] = np.inf
    return Delta

def _tcorr_Ashesh_capacitance(a, b, below, above, t):
    return _tcorr_Garg_capacitance_aux(a, b, below, above, _tcorr_Ashesh_Delta(a, b, t) / 2)

def _tcorr_ppc_C_aux(a, b, t, Cfun, epsr_above):
    lmbd = t / (b - a)
    Cvac = 2 * Cfun(lmbd)
    C = epsr_above * Cvac
    return C, Cvac

def _tcorr_ppc_C(a, b, above, t):
    first_h, first_epsr = _layerspec(above)[0]
    if np.any(first_h < t):
        raise ValueError("Conductor thicknesses larger than the first"
                         " dielectric layer are not currently supported.")
    return _tcorr_ppc_C_aux(a, b, t, _ppc_C, first_epsr)

def _tcorr_ppc_capacitance_aux(a, b, below, above, ppc_C, ppc_Cvac):
    lower_C, lower_Cvac = _capacitance_aux(a, b, *zip(*_layerspec(below)))
    upper_C, upper_Cvac = _capacitance_aux(a, b, *zip(*_layerspec(above)))
    C = lower_C + upper_C + ppc_C
    Cvac = lower_Cvac + upper_Cvac + ppc_Cvac
    return C, Cvac

def _tcorr_ppc_capacitance(a, b, below, above, t):
    return _tcorr_ppc_capacitance_aux(a, b, below, above,
                                      *_tcorr_ppc_C(a, b, above, t))

def _tcorr_ppcedge_C_aux(a, b, t, Cfun, epsr_below, epsr_above):
    lmbd = t / (b - a)
    Cvac = 2 * Cfun(lmbd)
    Cvac_ppc = 2 * _ppc_C(lmbd)
    single_edge_contribution = (Cvac - Cvac_ppc) / 2
    C = epsr_above * Cvac + (epsr_below - epsr_above) * single_edge_contribution
    return C, Cvac

def _tcorr_ppcedge_C(a, b, below, above, t):
    first_h_below, first_epsr_below = _layerspec(below)[0]
    if np.any(first_h_below < 2 * (b - a)):
        raise ValueError("Substrate thicknesses smaller than twice the gap size"
                         " are not currently supported.")
    first_h_above, first_epsr_above = _layerspec(above)[0]
    if np.any(first_h_above < t + 2 * (b - a)):
        raise ValueError("Conductor thicknesses larger than the first"
                         " dielectric layer minus twice the gap size"
                         " are not currently supported.")
    return _tcorr_ppcedge_C_aux(a, b, t, _ppcedge_C, first_epsr_below, first_epsr_above)

def _tcorr_ppcedge_capacitance(a, b, below, above, t):
    return _tcorr_ppc_capacitance_aux(a, b, below, above,
                                      *_tcorr_ppcedge_C(a, b, below, above, t))

def _tcorr_Cohn_C(a, b, below, above, t):
    first_h_below, first_epsr_below = _layerspec(below)[0]
    if np.any(first_h_below < 2 * (b - a)):
        raise ValueError("Substrate thicknesses smaller than twice the gap size"
                         " are not currently supported.")
    first_h_above, first_epsr_above = _layerspec(above)[0]
    if np.any(first_h_above < t + 2 * (b - a)):
        raise ValueError("Conductor thicknesses larger than the first"
                         " dielectric layer minus twice the gap size"
                         " are not currently supported.")
    return _tcorr_ppcedge_C_aux(a, b, t, _Cohn_C, first_epsr_below, first_epsr_above)

def _tcorr_Cohn_capacitance(a, b, below, above, t):
    return _tcorr_ppc_capacitance_aux(a, b, below, above,
                                      *_tcorr_Cohn_C(a, b, below, above, t))

def _capacitance(a: ArrayLike, b: ArrayLike,
                 below: LayerSpec = None,
                 above: LayerSpec = None,
                 t: ArrayLike = 0) -> Union[float, np.array]:
    # Make sure everything's an array.
    a = np.asarray(a)
    b = np.asarray(b)
    t = np.asarray(t)
    # Now do the calculation.
    if np.any(t):
        if thickness_correction_method == 'Gupta':
            _capfun = _tcorr_Gupta_capacitance
        elif thickness_correction_method == 'Garg':
            _capfun = _tcorr_Garg_capacitance
        elif thickness_correction_method == 'Ashesh':
            _capfun = _tcorr_Ashesh_capacitance
        elif thickness_correction_method == 'ppc':
            _capfun = _tcorr_ppc_capacitance
        elif thickness_correction_method == 'ppcedge':
            _capfun = _tcorr_ppcedge_capacitance
        elif thickness_correction_method == 'Cohn':
            _capfun = _tcorr_Cohn_capacitance
        else:
            raise ValueError(f"Unknown t correction method {thickness_correction_method!r}.")
        return _capfun(a, b, below, above, t)
    else:
        return _tcorr_none_capacitance(a, b, below, above)

def capacitance(a: ArrayLike, b: ArrayLike,
                below: LayerSpec = None,
                above: LayerSpec = None, *,
                t: ArrayLike = 0) -> Union[float, np.array]:
    """Compute the specific capacitance.

    You can use :func:`.stack_layers` to create values for `below`
    and `above`.

    The unit of the return value will be F / m irrespective of the unit of the
    input lengths (as long as they all have the same unit).

    Arguments:
        a: Distance of the inner edge of the CPW gap from the center line
            (i.e. half the width of the center conductor).
        b: Distance of the outer edge of the CPW gap from the center line
            (i.e. `a` plus the gap width).
        below: The dielectric layers below the CPW.
            Each layer should be given as a tuple containing the height at which
            the layer ends and the relative permittivity of that layer. The last
            layer is considered to be terminated by a metal cover. The default
            corresponds to infinite vacuum.
        above: The dielectric layers above the CPW.
            (In the same format as `layers_below`.)
        t: Metallization thickness of the CPW.

    Returns:
        The CPW's specific capacitance in the same shape as `a` and `b`.

    """
    return _capacitance(a, b, below, above, t)[0]

def characteristics(a: ArrayLike, b: ArrayLike,
                    below: LayerSpec = None,
                    above: LayerSpec = None, *,
                    t: ArrayLike = 0) -> dict:
    """Compute characteristic values.

    You can use :func:`.stack_layers` to create values for `below`
    and `above`.

    The units of the returned values will be SI base units irrespective of the
    unit of the input lengths (as long as they all have the same unit).

    Arguments:
        a: Distance of the inner edge of the CPW gap from the center line
            (i.e. half the width of the center conductor).
        b: Distance of the outer edge of the CPW gap from the center line
            (i.e. `a` plus the gap width).
        below: The dielectric layers below the CPW.
            Each layer should be given as a tuple containing the height at which
            the layer ends and the relative permittivity of that layer. The last
            layer is considered to be terminated by a metal cover. The default
            corresponds to infinite vacuum.
        above: The dielectric layers above the CPW.
            (In the same format as `layers_below`.)
        t: Metallization thickness of the CPW.

    Returns:
        A dict of characteristic values comprising `Z0`, `v_ph`, `eps_eff`,
        `C` and `L`.

    """
    C, Cvac = _capacitance(a, b, below, above, t)
    eps_eff = C / Cvac
    v_ph = c_vac / np.sqrt(eps_eff)
    Z0 = 1 / (C * v_ph)
    L = 1 / (v_ph**2 * C)
    return {'Z0': Z0, 'v_ph': v_ph, 'eps_eff': eps_eff, 'C': C, 'L': L}

def impedance(a: ArrayLike, b: ArrayLike,
              below: LayerSpec = None,
              above: LayerSpec = None, *,
              t: ArrayLike = 0) -> tuple[Union[float, np.array]]:
    """Compute the characteristic impedance.

    You can use :func:`.stack_layers` to create values for `below`
    and `above`.

    The unit of the return value will be Ohm irrespective of the unit of the
    input lengths (as long as they all have the same unit).

    Arguments:
        a: Distance of the inner edge of the CPW gap from the center line
            (i.e. half the width of the center conductor).
        b: Distance of the outer edge of the CPW gap from the center line
            (i.e. `a` plus the gap width).
        below: The dielectric layers below the CPW.
            Each layer should be given as a tuple containing the height at which
            the layer ends and the relative permittivity of that layer. The last
            layer is considered to be terminated by a metal cover. The default
            corresponds to infinite vacuum.
        above: The dielectric layers above the CPW.
            (In the same format as `layers_below`.)
        t: Metallization thickness of the CPW.

    Returns:
        The CPW's characteristic impedance in the same shape as `a` and `b`.

    """
    return characteristics(a, b, below, above, t=t)['Z0']
