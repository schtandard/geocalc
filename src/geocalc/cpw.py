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
from scipy.special import ellipk
from scipy.constants import epsilon_0 as eps_0, c as c_vac
from typing import Union, Literal
from numpy.typing import ArrayLike
from ._util import LayerSpec, _layerspec

__all__ = ['ab2SW', 'SW2ab', 'DW2ab', 'dW2ab',
           'capacitance', 'impedance', 'characteristics']

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
    """Compute the half-plane specific capacitance contributions and its vacuum value.

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
    return C.reshape(np.shape(a)), Cvac_tot.reshape(np.shape(a))

def _capacitance(a: ArrayLike, b: ArrayLike,
                 layers_below: LayerSpec = None,
                 layers_above: LayerSpec = None,
                 t: ArrayLike = 0) -> Union[float, np.array]:
    lower_C, lower_Cvac = _capacitance_aux(a, b, *zip(*_layerspec(layers_below)))
    upper_C, upper_Cvac = _capacitance_aux(a, b, *zip(*_layerspec(layers_above)))
    C = lower_C + upper_C
    Cvac = lower_Cvac + upper_Cvac
    if np.any(t):
        first_h, first_eps = _layerspec(layers_above)[0]
        if np.any(first_h < t):
            raise ValueError("Conductor thicknesses larger than the first"
                             " dielectric layer are not currently supported.")
        Cppc = 2 * eps_0 * t / (b - a)
        Cvac += Cppc
        C += first_eps * Cppc
    return C, Cvac

def capacitance(a: ArrayLike, b: ArrayLike,
                layers_below: LayerSpec = None,
                layers_above: LayerSpec = None, *,
                t: ArrayLike = 0) -> Union[float, np.array]:
    """Compute the specific capacitance.

    You can use :func:`.stack_layers` to create values for `layers_below`
    and `layers_above`.

    The unit of the return value will be F / m irrespective of the unit of the
    input lengths (as long as they all have the same unit).

    Arguments:
        a: Distance of the inner edge of the CPW gap from the center line
            (i.e. half the width of the center conductor).
        b: Distance of the outer edge of the CPW gap from the center line
            (i.e. `a` plus the gap width).
        layers_below: The dielectric layers below the CPW.
            Each layer should be given as a tuple containing the height at which
            the layer ends and the relative permittivity of that layer. The last
            layer is considered to be terminated by a metal cover. The default
            corresponds to infinite vacuum.
        layers_above: The dielectric layers above the CPW.
            (In the same format as `layers_below`.)
        t: Metallization thickness of the CPW.

    Returns:
        The CPW's specific capacitance in the same shape as `a` and `b`.

    """
    return _capacitance(a, b, layers_below, layers_above, t)[0]

def characteristics(a: ArrayLike, b: ArrayLike,
                    layers_below: LayerSpec = None,
                    layers_above: LayerSpec = None, *,
                    t: ArrayLike = 0) -> dict:
    """Compute characteristic values.

    You can use :func:`.stack_layers` to create values for `layers_below`
    and `layers_above`.

    The units of the returned values will be SI base units irrespective of the
    unit of the input lengths (as long as they all have the same unit).

    Arguments:
        a: Distance of the inner edge of the CPW gap from the center line
            (i.e. half the width of the center conductor).
        b: Distance of the outer edge of the CPW gap from the center line
            (i.e. `a` plus the gap width).
        layers_below: The dielectric layers below the CPW.
            Each layer should be given as a tuple containing the height at which
            the layer ends and the relative permittivity of that layer. The last
            layer is considered to be terminated by a metal cover. The default
            corresponds to infinite vacuum.
        layers_above: The dielectric layers above the CPW.
            (In the same format as `layers_below`.)
        t: Metallization thickness of the CPW.

    Returns:
        A dict of characteristic values comprising `Z0`, `v_ph`, `eps_eff`,
        `C` and `L`.

    """
    C, Cvac = _capacitance(a, b, layers_below, layers_above, t)
    eps_eff = C / Cvac
    v_ph = c_vac / np.sqrt(eps_eff)
    Z0 = 1 / (C * v_ph)
    L = 1 / (v_ph**2 * C)
    return {'Z0': Z0, 'v_ph': v_ph, 'eps_eff': eps_eff, 'C': C, 'L': L}

def impedance(a: ArrayLike, b: ArrayLike,
              layers_below: LayerSpec = None,
              layers_above: LayerSpec = None, *,
              t: ArrayLike = 0) -> tuple[Union[float, np.array]]:
    """Compute the characteristic impedance.

    You can use :func:`.stack_layers` to create values for `layers_below`
    and `layers_above`.

    The unit of the return value will be Ohm irrespective of the unit of the
    input lengths (as long as they all have the same unit).

    Arguments:
        a: Distance of the inner edge of the CPW gap from the center line
            (i.e. half the width of the center conductor).
        b: Distance of the outer edge of the CPW gap from the center line
            (i.e. `a` plus the gap width).
        layers_below: The dielectric layers below the CPW.
            Each layer should be given as a tuple containing the height at which
            the layer ends and the relative permittivity of that layer. The last
            layer is considered to be terminated by a metal cover. The default
            corresponds to infinite vacuum.
        layers_above: The dielectric layers above the CPW.
            (In the same format as `layers_below`.)
        t: Metallization thickness of the CPW.

    Returns:
        The CPW's characteristic impedance in the same shape as `a` and `b`.

    """
    return characteristics(a, b, layers_below, layers_above, t=t)['Z0']
