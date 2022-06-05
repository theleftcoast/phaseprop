"""Generalized implementation of binary interactions between components or pseudo-components."""

import numpy as np
import dataclasses
import typing
import comp


@dataclasses.dataclass
class BinaryInterParm(object):
    """Temperature-dependent binary interaction parameter.
        bip_ij(T) = k_ij + a*T + b/T + c*ln(T)
    """
    const: float = 0.0
    linear_t: float = 0.0
    inverse_t: float = 0.0
    ln_t: float = 0.0
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def k_ij(self, t: float) -> float:
        if t > 0.0:
            return self.const + self.linear_t * t + self.inverse_t / t + self.ln_t * np.log(t)
        else:
            raise ValueError("temperature must be greater than zero.")


@dataclasses.dataclass
class BinaryInter():
    """Parameters that characterize the interaction of components or pseudo-components."""
    comp_a: comp.Comp
    comp_b: comp.Comp
    k_ij: typing.Dict[str, BinaryInterParm] = dataclasses.field(default_factory=dict)
