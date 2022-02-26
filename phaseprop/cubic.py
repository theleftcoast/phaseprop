"""Cubic equation of state.

Attributes
----------
RK : instance of CubicSpec
    Redlich-Kwong equation of state functional specification.
PR : instance of CubicSpec
    Peng-Robinson equation of state functional specification.
"""
import numpy as np
from eos import EOS
import dataclasses
import typing
import refs


@dataclasses.dataclass
class CubicSpec(object):
    """Constants and functions that define a specific version of the generalized cubic equation of state.

    Parameters
    ----------
    delta_1 : float
        Placeholder...need to define.
    delta_2 : float
        Placeholder...need to define.
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.

    Notes
    -----
    Generalized cubic equation of state functional form and thermodynamic properties are defined in [1]_.

        P = R*T/(v - b) - a*alpha(T)/((v + delta_1*b) * (v + delta_2*b))

    References
    ----------
    [1] Michelsen, M. L.; Mollerup, J. Thermodynamic Models: Fundamental and Computational Aspects, 2nd ed.; Tie-Line
    Publications: Holte, Denmark, 2007.
    """
    delta_1: float
    delta_2: float
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _d_1(self):
        """Intermediate variable used to evaluate critical properties."""
        return self.delta_1 + self.delta_2

    def _d_2(self):
        """Intermediate variable used to evaluate critical properties."""
        return self.delta_1 * self.delta_2

    def _y(self):
        """Intermediate variable used to evaluate critical properties."""
        a = ((1.0 + self.delta_2) * (1.0 + self.delta_1) ** 2.0) ** (1.0 / 3.0)
        b = ((1.0 + self.delta_1) * (1.0 + self.delta_2) ** 2.0) ** (1.0 / 3.0)
        return 1.0 + a + b

    def _gamma_c(self):
        """Intermediate variable used to evaluate critical properties."""
        numerator = 3.0 * self.y() ** 2.0 + 3.0 * self.y() * self.d_1() + self.d_1() ** 2.0 - self.d_2()
        denominator = 3.0 * self.y() + self.d_1() - 1.0
        return numerator / denominator

    def _z_c(self):
        """Critical compressibility factor."""
        return self.y() / (3.0 * self.y() + self.d_1() - 1.0)

    def _z_b(self):
        """Intermediate variable used to evaluate critical properties."""
        return 1.0 / (3.0 * self.y() + self.d_1() - 1.0)

    def _omega_a(self):
        """Constant for ac if the CEOS is constrained to match Tc and Pc."""
        return self.gamma_c() * self.z_b()

    def _omega_b(self):
        """Constant for bc if the CEOS is constrained to match Tc and Pc."""
        return self.z_b()


RK = CubicSpec(delta_1=1.0,
               delta_2=0.0,
               source=refs.rk)

PR = CubicSpec(delta_1=1.0 + 2 ** 0.5,
               delta_2=0.0 - 2 ** 0.5,
               source=refs.pr)


@dataclasses.dataclass
class SoaveAlpha(object):
    """Soave alpha function.

    Parameters
    ----------
    m : float
        Soave model parameter.
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.

    References
    ----------
    [1] Soave, G. Equilibrium constants from a modified Redlich-Kwong equation of state. Fluid Phase Equilb. 1972, 27,
    1197-1203.
    [2] Peng, D. Y.; Robinson, D. B. A New Two-Constant Equation of State. Ind. Eng. Chem. Fundamen. 1976, 15, 59–64.
    """
    m: float
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    @staticmethod
    def srk_m_from_acentric(acentric: float) -> float:
        """Correlation to estimate SRK 'm' for nonpolar compounds."""
        return 0.48 + 1.574 * acentric + 0.176 * acentric ** 2.0

    @staticmethod
    def pr_m_from_acentric(acentric: float) -> float:
        """Correlation to estimate PR 'm' for nonpolar compounds."""
        return 0.37464 + 1.54226 * acentric + 0.26992 * acentric ** 2.0

    def __call__(self, tr: float) -> float:
        """Soave alpha function.

        Parameters
        ----------
        tr : float
            Reduced temperature (defined as Tr = T / Tc).

        Returns
        -------
        float
            Soave's alpha function evaluated at tr.
        """
        return (1.0 + self.m * (1.0 - tr ** 0.5)) ** 2.0


@dataclasses.dataclass
class GasemAlpha(object):
    """Gasem alpha function.

    Parameters
    ----------
    a : float
        Gasem model parameter.
    b : float
        Gasem model parameter.
    c : float
        Gasem model parameter.
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.

    Notes
    -----
    For nonpolar compounds, use a = 2.0 and b = 0.836
    For hydrogen, use a = -4.3 and b = 10.4

    References
    ----------
    [1] Gasem, K. A. M.; Gao, W.; Pan, Z.; Robinson R. L. A modified temperature dependence for the Peng-Robinson
    equation of state. Fluid Phase Equilib. 2001, 181, 113-125.
    """
    a: float
    b: float
    c: float
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    @staticmethod
    def c_from_acentric(acentric: float) -> float:
        """Correlation to estimate Gasem 'c' for nonpolar compounds."""
        return 0.134 + 0.508 * acentric - 0.0467 * acentric ** 2.0

    def __call__(self, tr: float) -> float:
        """Gasem alpha function.

        Parameters
        ----------
        tr : float
            Reduced temperature (defined as Tr = T / Tc).

        Returns
        -------
        float
            Gasem's alpha function evaluated at tr.
        """
        return np.exp((self.a + self.b * tr) * (1.0 - tr ** self.c))


@dataclasses.dataclass
class TwuAlpha(object):
    """Twu alpha function.

    Parameters
    ----------
    l : float
        Twu model parameter.
    m : float
        Twu model parameter.
    n : float
        Twu model parameter.
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.


    References
    ----------
    [1] Twu, C. H.; Bluck, D.; Cunningham, J. R.; Coon, J. E. A cubic equation of state with a new alpha function
    and a new mixing rule. Fluid Phase Equilib. 1991, 69, 33-50.
    [2] Le Guennec, Y.; Privat, R.; Jaubert. J. N. Development of the translated-consistent tc-PR and tc-RK cubic
    equations of state for a safe and accurate prediction of volumetric, energetic and saturation properties of pure
    compounds in the sub- and super-critical domains. Fluid Phase Equilib. 2016, 429, 301-312.
    """
    l: float
    m: float
    n: float
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def __call__(self, tr: float) -> float:
        """Twu's alpha function.

        Parameters
        ----------
        tr : float
            Reduced temperature (defined as Tr = T / Tc).

        Returns
        -------
        float
            Twu's alpha function evaluated at tr.
        """
        return np.exp(self.l * (1.0 - tr ** (self.n * self.m))) * (tr ** (self.n * (self.m - 1.0)))


@dataclasses.dataclass
class SRKParms(object):
    """Soave's Redlich-Kwong equation of state parameters for a single component.

    Parameters
    ----------
    a : float
        Attractive parameter.
    b : float
        Repulsive parameter.
    alpha : SoaveAlpha
        Soave alpha function.
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.

    References
    ----------
    [1] Soave, G. Equilibrium constants from a modified Redlich-Kwong equation of state. Fluid Phase Equilb. 1972, 27,
    1197-1203.
    """
    a: float
    b: float
    alpha: SoaveAlpha
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class CPAParms(object):
    """Cubic Plus Association equation of state parameters for a single component.

    Parameters
    ----------
    a : float
        Attractive parameter.
    b : float
        Repulsive parameter.
    alpha : SoaveAlpha
        Soave alpha function.
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.

    References
    ----------
    [1] Kontogeorgis, G. M.; Voutsas, E. C.; Yakoumis, I. V.; Tassios, D. P. An Equation of State for Associating
    Fluids. Ind. Eng. Chem. Res. 1996, 35, 4310-4318.
    [2] Kontogeorgis, G. M.; Yakoumis, I. V.; Meijer, H.; Hendriks, E.; Moorwood, T. Multicomponent phase equilibrium
    calculations for water–methanol–alkane mixtures. Fluid Phase Equilib. 1999, 158-160, 201-209.
    """
    a: float
    b: float
    alpha: SoaveAlpha
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    @staticmethod
    def a_from_a0(a: float) -> float:
        """Convert a0 to a."""
        # TODO: Build this out further.
        return a

    @staticmethod
    def b_from_b(b: float) -> float:
        # TODO: Build this out further.
        """Convert b to b."""
        return b

    @staticmethod
    def m_from_c1(c1: float) -> float:
        # TODO: Build this out further.
        """Convert c1 to m."""
        return c1


@dataclasses.dataclass
class PRParms(object):
    """Peng-Robinson equation of state parameters for a single component.

    Parameters
    ----------
    a : float
        Attractive parameter.
    b : float
        Repulsive parameter.
    alpha : SoaveAlpha
        Soave alpha function.
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.

    References
    ----------
    [1] Peng, D. Y.; Robinson, D. B. A New Two-Constant Equation of State. Ind. Eng. Chem. Fundamen. 1976, 15, 59–64.
    """
    a: float
    b: float
    alpha: SoaveAlpha
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class GPRParms(object):
    """Gasem's Peng-Robinson equation of state parameters for a single component.

    Parameters
    ----------
    a : float
        Attractive parameter.
    b : float
        Repulsive parameter.
    alpha : SoaveAlpha
        Soave alpha function.
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.

    References
    ----------
    [1] Gasem, K. A. M.; Gao, W.; Pan, Z.; Robinson R. L. A modified temperature dependence for the Peng-Robinson
    equation of state. Fluid Phase Equilib. 2001, 181, 113-125.
    """
    a: float
    b: float
    alpha: GasemAlpha
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class TPRParms(object):
    """Twu's Peng-Robinson equation of state parameters for a single component.

    Parameters
    ----------
    a : float
        Attractive parameter.
    b : float
        Repulsive parameter.
    alpha : SoaveAlpha
        Soave alpha function.
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.

    References
    ----------
    [3] Twu, C. H.; Bluck, D.; Cunningham, J. R.; Coon, J. E. A cubic equation of state with a new alpha function and a
    new mixing rule. Fluid Phase Equilib. 1991, 69, 33-50.
    """
    a: float
    b: float
    alpha: GasemAlpha
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


class CubicPhysInter(object):
    """Physical interactions between components."""


class Cubic(EOS):
    """Generalized implementation of the cubic equation of state.

    Attributes:cubic_spec
        r: Ideal gas constant
        a: Energy parameter
        b: Co-volume parameter
        v: Molar volume, 1/rho
        rho: Molar density, 1/v
        eta: Reduced fluid density, b/4*V

    Methods:
        p: Returns pressure when volume and temperature are specified
        phi: Returns fugacity coefficients
    """

    def __init__(self, cubic_spec=None):
        # Define a cubic equation of state.
        if isinstance(cubic_spec, CubicSpec):
            self.delta_1 = cubic_spec.delta_1
            self.delta_2
            self.alpha = self._soave
        else:
            raise ValueError("Must pass a CubicSpec instance to the constructor.")

        # Lists of component objects, ci, and their concentrations, ni.
        self.ci = []
        self.ni = []

    def p(self, t, v, a, b):
        return self.r * t / (v - b) - a / ((v + self.d1 * b) * (v + self.d2 * b))

    def z(self, t, v, a, b):
        return 1.0 / (1.0 - b / v) - (a / (self.r * t * b)) * (b / v) / (
                    (1.0 + self.d1 * b / v) * (1.0 + self.d2 * b / v))

    def F(self, n, t, V, B, D):
        return -n * g(V, B) - D(t) * f(V, B) / t

    def g(self, V, B):
        return np.log(V - B) - np.log(V)

    def f(self, V, B):
        return np.log((V + self.d1 * B) / (V + self.d2 * B)) / (self.r * B * (self.d1 - self.d2))
