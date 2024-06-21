"""Geocalc utilities."""

from __future__ import annotations
import numpy as np
from typing import Union, Optional, Iterable
from scipy.constants import epsilon_0 as eps_0
from scipy.special import ellipk, ellipe
from scipy.optimize import root_scalar

LayerSpec = Optional[Union[Iterable[tuple], float, int, np.floating, np.integer]]

def _layerspec(layers: LayerSpec) -> Iterable[tuple]:
    if layers is None:
        return [(np.inf, 1)]
    if isinstance(layers, (float, int, np.floating, np.integer)):
        return [(np.inf, layers)]
    return layers

def stack_layers(*layers: Iterable[tuple], start_stack: Iterable[tuple] = None) -> list[tuple]:
    """Stack a number of layers.

    This just shifts the termination coordinate of each layer by the heights
    of previous layers.

    Arguments:
        layers: The layers to stack.
            Each layer should be given as a tuple whose first value represents
            its termination coordinate (i.e. its height).
        start_stack: A stack to put at the bottom of the stack unchanged.

    Returns:
        The stacked layers.

    """
    if start_stack is None:
        abs_layers = []
        height = 0
    else:
        abs_layers = list(start_stack)
        height = abs_layers[-1][0]
    for l in layers:
        height += l[0]
        abs_layers.append((height, *l[1:]))
    return abs_layers

def _ppc_C(lmbd):
    return eps_0 * lmbd

def _ppcedge_C(lmbd):
    # Edge corrections for parallel-plate capacitors
    # (E. Yariv, European Journal of Applied Mathematics 32, 226-241 (2021))
    return eps_0 * (lmbd + (np.log(2 * np.pi * lmbd) + 1) / np.pi)

def _Cohn_m_charfun(m):
    a = (1 + m) / 2 * ellipk(1 - m)
    b = ellipe(1 - m)
    c = 2 * ellipe(m)
    # For m = 1 the fraction (1 - m) * K(m) goes to zero.
    d = np.multiply(1 - m, ellipk(m), where=m != 1, out=np.zeros_like(m, dtype=float))
    return (a - b) / (c - d)

def _Cohn_m_single(lmbd):
    if lmbd == 0:
        return 1
    return root_scalar(lambda m: _Cohn_m_charfun(m) - lmbd, bracket=[0, 1]).root

def _Cohn_m(lmbd):
    lmbd = np.asarray(lmbd)
    if not lmbd.shape:
        # A single value.
        return _Cohn_m_single(lmbd)
    m = [_Cohn_m_single(l) for l in lmbd]
    return np.array(m)

def _Cohn_C_aux(m):
    # For m to 1 the fraction (1 - m) * K(m) goes to zero.
    d = np.multiply(1 - m, ellipk(m), where=m != 1, out=np.zeros_like(m, dtype=float))
    # For m to 0 the fraction (E - d / 2) / m**(1/4) goes to inf.
    arg = np.divide(ellipe(m) - d / 2,
                    np.power(m, 0.25),
                    where=m != 0,
                    out=np.full_like(m, np.inf, dtype=float))
    return 2 / np.pi * np.log(arg)

_Cohn_offset_limit = 2 / np.pi * (1 + np.log(np.pi / 8))
def _Cohn_C(lmbd):
    # Thickness Corrections for Capacitive Obstacles and Strip Conductors
    # (Seymour B. Cohn, IRE Transactions on Microwave Theory and Techniques 8, 638-644 (1960))
    # Large lmdb lead to significant numerical errors in finding m, which gets tiny.
    # For very large lmbd, root_scalar returns a constant value around 1e-12.
    # The issues start appearing around lmbd == pi, at which point the offset
    # from the parallel plate capacitor value is well saturated. We keep some
    # safety distance and use the limiting behavior (Cohn (5)) for all lmbd
    # beyond 2.5.
    out = lmbd + _Cohn_offset_limit
    small = lmbd < 2.5
    # Make sure out is an array so the following assigment also works with scalars.
    out = np.asarray(out)
    out[small] = _Cohn_C_aux(_Cohn_m(lmbd[small]))
    return eps_0 * out
