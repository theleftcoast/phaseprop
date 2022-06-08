"""Generalized implementation of binary interaction parameter."""

import numpy as np
import dataclasses
import typing


@dataclasses.dataclass
class BinaryInterParm(object):
    """Temperature-dependent binary interaction parameter between two componets or pseudo-components."""
    const: float = 0.0
    linear_t: float = 0.0
    inverse_t: float = 0.0
    ln_t: float = 0.0
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def bip(self, t: float) -> float:
        if t > 0.0:
            return self.const + self.linear_t * t + self.inverse_t / t + self.ln_t * np.log(t)
        else:
            raise ValueError("temperature must be greater than zero.")

