"""Physical and project constants.

Attributes
----------
R : float
    Universal gas constant, J/mol.K
EC : float
    Elementary charge, C
KB : float
    Boltzmann's constant, J/K
NA : float
    Avogadro's number, particles/mol
C : float
    Speed of light, m/s
F : float
    Faraday constant, C/mol
PI : float
    Mathematical constant
E0 : float
    Vacuum permittivity, F/m
ROOT_DIR : Path
    Root directory for project

Notes
-----
Constants are taken from Perry's Chemical Engineer's Handbook [1]_.

References
----------
[1] Perry's Chemical Engineers' Handbook; Perry, R. H., Southard, M. Z., Eds.; McGraw-Hill Education: New York, 2019.
"""

from pathlib import Path
import typing
import src.phaseprop.units as units
import src.phaseprop.refs as refs

# Project constants
ROOT_DIR = Path(__file__).resolve().parent


class Const(float):
    """Constant with metadata.

    Parameters
    ----------
    unit : str, optional
        Unit associated with the constant.
    uncertainty : float, optional
        Uncertainty associated with the constant.
    source : str, optional
        Source for the constant (ACS citation format preferred).
    notes : str, optional
        Notes associated with the constant.
    """

    def __new__(
        cls,
        value: float,
        unit: typing.Optional[str] = None,
        uncertainty: typing.Optional[float] = None,
        source: typing.Optional[str] = None,
        notes: typing.Optional[str] = None,
    ):
        return float.__new__(cls, value)

    def __init__(
        self,
        value: float,
        unit: typing.Optional[str] = None,
        uncertainty: typing.Optional[float] = None,
        source: typing.Optional[str] = None,
        notes: typing.Optional[str] = None,
    ):
        float.__init__(value)
        self.unit = unit
        self.uncertainty = uncertainty
        self.source = source
        self.notes = notes

    @property
    def unit(self):
        """str : Source unit for constant."""
        return self._unit

    @unit.setter
    def unit(self, value):
        if value in units.UNITS:
            self._unit = value
            return
        elif value in units.TEMPERATURE:
            self._unit = value
            return
        elif value is None:
            self._unit = value
            return
        else:
            raise ValueError("Unit is not defined.")


# Physical constants
R = Const(value=8.31446261815324, source=refs.dippr)
EC = Const(value=1.602176634 * 10**-19, source=refs.dippr)
KB = Const(value=1.380649 * 10**-23, source=refs.dippr)
NA = Const(value=6.02214076 * 10**23, source=refs.dippr)
C = Const(value=299792458.0, source=refs.dippr)
F = Const(value=96485.33212, source=refs.dippr)
PI = Const(value=3.14159265358979323846, unit="dimensionless", source=refs.dippr)
E0 = Const(value=8.8541878128 * 10**-12, source=refs.dippr)
