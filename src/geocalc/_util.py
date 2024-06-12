"""Geocalc utilities."""

from __future__ import annotations
import numpy as np
from typing import Union, Optional, Iterable

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
