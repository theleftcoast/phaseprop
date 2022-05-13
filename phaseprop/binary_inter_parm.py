"""Generalized implementation of a binary interaction parameter"""

from __future__ import annotations
import numpy as np
import comp


class BinaryInterParm(object):
    """Binary interaction parameter between components or pseudo-components.
        k_ij(T) = k_ij + a*T + b/T + c*ln(T)
    """

    def __init__(self, comp_a=None, comp_b=None, source=None,
                 temp_indep_coef=None, lin_temp_coef=None, inv_temp_coef=None, ln_temp_coef=None):
        if comp_a is None or comp_b is None:
            raise ValueError("comp_a and comp_b must be provided to create an instance of BinaryInterParm.")
        else:
            self.comp_a = comp_a
            self.comp_b = comp_b
            self.source = source
            self.temp_indep_coef = temp_indep_coef
            self.lin_temp_coef = lin_temp_coef
            self.inv_temp_coef = inv_temp_coef
            self.ln_temp_coef = ln_temp_coef

    @property
    def comp_a(self):
        return self._comp_a

    @comp_a.setter
    def comp_a(self, value):
        try:
            self._comp_a
        except AttributeError:
            if isinstance(value, (comp.Comp, comp.PseudoComp)):
                self._comp_a = value
            else:
                raise TypeError("comp_a must be an instance of Comp or PseudoComp.")

    @property
    def comp_b(self):
        return self._comp_b

    @comp_b.setter
    def comp_b(self, value):
        try:
            self._comp_b
        except AttributeError:
            if isinstance(value, (comp.Comp, comp.PseudoComp)):
                self._comp_b = value
            else:
                raise TypeError("comp_b must be an instance of Comp or PseudoComp.")

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        if value is None:
            self._source = value
        elif isinstance(value, str):
            self._source = value
        else:
            raise ValueError("source must be a string.")

    @property
    def temp_indep_coef(self):
        return self._temp_indep_coef

    @temp_indep_coef.setter
    def temp_indep_coef(self, value):
        if value is None:
            self._temp_indep_coef = value
        elif isinstance(value, float):
            self._temp_indep_coef = value
        else:
            raise ValueError("temp_indep_coef must be a float.")

    @property
    def lin_temp_coef(self):
        return self._lin_temp_coef

    @lin_temp_coef.setter
    def lin_temp_coef(self, value):
        if value is None:
            self._lin_temp_coef = value
        elif isinstance(value, float):
            self._lin_temp_coef = value
        else:
            raise ValueError("lin_temp_coef must be a float.")

    @property
    def inv_temp_coef(self):
        return self._inv_temp_coef

    @inv_temp_coef.setter
    def inv_temp_coef(self, value):
        if value is None:
            self._inv_temp_coef = value
        elif isinstance(value, float):
            self._inv_temp_coef = value
        else:
            raise ValueError("inv_temp_coef must be a float.")

    @property
    def ln_temp_coef(self):
        return self._ln_temp_coef

    @ln_temp_coef.setter
    def ln_temp_coef(self, value):
        if value is None:
            self._ln_temp_coef = value
        elif isinstance(value, float):
            self._ln_temp_coef = value
        else:
            raise ValueError("ln_temp_coef must be a float.")

    def __eq__(self, other):
        if isinstance(other, BinaryInterParm):
            aa_bb_eq = self.comp_a == other.comp_a and self.comp_b == other.comp_b
            ab_ba_eq = self.comp_a == other.comp_b and self.comp_b == other.comp_a
            return aa_bb_eq or ab_ba_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.comp_a), hash(self.comp_b)))

    def __str__(self):
        return "Comp A: ({}), Comp B: ({}), k_ij: {}".format(self.comp_a,
                                                             self.comp_b,
                                                             self.k_ij)

    # TODO: Include a defined flag for all other classes.
    def defined(self):
        # A binary interaction parameter is considered defined if any of the coefficients are not none.
        tic = self.temp_indep_coef is not None
        ltc = self.lin_temp_coef is not None
        itc = self.inv_temp_coef is not None
        lntc = self.ln_temp_coef is not None
        return tic or ltc or itc or lntc

    def k_ij(self, t):
        if t <= 0.0:
            raise ValueError("The temperature cannot be less than zero.")

        ti = self.temp_indep_coef if (self.temp_indep_coef is not None) else 0.0
        lt = self.lin_temp_coef * t if (self.lin_temp_coef is not None) else 0.0
        it = self.inv_temp_coef / t if (self.inv_temp_coef is not None) else 0.0
        lnt = self.ln_temp_coef * np.log(t) if (self.ln_temp_coef is not None) else 0.0
        return ti + lt + it + lnt
