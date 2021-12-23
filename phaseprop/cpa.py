"""Cubic equation of state.

Attributes
----------
PR : instance of CubicSpec
    Peng-Robinson equation of state (Peng and Robinson, [1]_).
GPR : instance of CubicSpec
    Peng-Robinson equation of state with the Gasem alpha function (Gasem et al., [2]_).
TPR : instance of CubicSpec
    Peng-Robinson equation of state with the Twu alpha function (Twu et al., [3]_).
SRK : instance of CubicSpec
    Soave-Redlich-Kwong equation of state (Soave, [4]_).
CPA : instance of CubicSpec
    Cubic-plus-association equation of state (Kontogeorgis, [5, 6]_).

Notes
-----


References
----------
[1] Peng, D. Y.; Robinson, D. B. A New Two-Constant Equation of State. Ind. Eng. Chem. Fundamen. 1976, 15, 59–64.
[2] Gasem, K. A. M.; Gao, W.; Pan, Z.; Robinson R. L. A modified temperature dependence for the Peng-Robinson equation
of state. Fluid Phase Equilib. 2001, 181, 113-125.
[3] Twu, C. H.; Bluck, D.; Cunningham, J. R.; Coon, J. E. A cubic equation of state with a new alpha function and a new
mixing rule. Fluid Phase Equilib. 1991, 69, 33-50.
[4] Soave, G. Equilibrium constants from a modified Redlich-Kwong equation of state. Fluid Phase Equilb. 1972, 27,
1197-1203.
[5] Kontogeorgis, G. M.; Voutsas, E. C.; Yakoumis, I. V.; Tassios, D. P. An Equation of State for Associating Fluids.
Ind. Eng. Chem. Res. 1996, 35, 4310-4318.
[6] Kontogeorgis, G. M.; Yakoumis, I. V.; Meijer, H.; Hendriks, E.; Moorwood, T. Multicomponent phase equilibrium
calculations for water–methanol–alkane mixtures. Fluid Phase Equilib. 1999, 158-160, 201-209.
"""
import numpy as np
from eos import EOS


class CubicSpec(object):
    """Constants and functions that define a specific version of the generalized cubic equation of state.

    Notes
    -----
    Generalized cubic equation of state functional form and thermodynamic properties are from [1]_.

    P = R*T/(v - b) - a*alpha(T)/((v + delta_1*b) * (v + delta_2*b))

    References
    ----------
    [1] Michelsen, M. L.; Mollerup, J. Thermodynamic Models: Fundamental and Computational Aspects, 2nd ed.; Tie-Line
    Publications: Holte, Denmark, 2007.
    """

    def __init__(self, delta_1=None, delta_2=None, alpha=None):
        """
        Parameters
        ----------
        delta_1 : float
        delta_2 : float
        alpha : str
        """
        if delta_1 is None or delta_2 is None or alpha is None:
            raise ValueError("Inputs are not sufficient to define an instance of CubicSpec.")
        else:
            self.delta_1 = delta_1
            self.delta_2 = delta_2
            self.alpha = alpha

    @property
    def delta_1(self):
        """delta_1 parameter for generalized cubic equation of state."""
        return self._delta_1

    @delta_1.setter
    def delta_1(self, value):
        try:
            self._delta_1
        except AttributeError:
            if isinstance(value, float):
                self._delta_1 = value
            else:
                raise ValueError("delta_1 must be a float.")

    @property
    def delta_2(self):
        """delta_2 parameter for generalized cubic equation of state."""
        return self._delta_2

    @delta_2.setter
    def delta_2(self, value):
        try:
            self._delta_2
        except AttributeError:
            if isinstance(value, float):
                self._delta_2 = value
            else:
                raise ValueError("delta_2 must be a float.")

    @property
    def alpha(self):
        """Alpha function for generalized cubic equation of state (either 'soave', 'gasem', or 'twu')."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        try:
            self._alpha
        except AttributeError:
            if value == 'soave':
                self._alpha = self._soave
            elif value == 'gasem':
                self._alpha = self._gasem
            elif value == 'twu':
                self._alpha = self._twu
            else:
                raise ValueError("alpha must be a valid string.")

    def _soave(self, tr, m):
        """Soave alpha function.

        Parameters
        ----------
        tr : float
            Reduced temperature (defined as Tr = T / Tc).
        m : float
            Soave's slope parameter.

        Returns
        -------
        float
            Soave's alpha function evaluated at tr.

        Notes
        -----
        Initial guess for nonpolar compounds: m = 0.48 + 1.574*acentric + 0.176*acentric**2

        References
        ----------
        [1] Soave, G. Equilibrium constants from a modified Redlich-Kwong equation of state. Fluid Phase Equilb. 1972,
        27, 1197-1203.
        """
        return (1.0 + m * (1.0 - tr ** 0.5)) ** 2.0

    def _gasem(self, tr, a, b, c):
        """Gasem alpha function.

        Parameters
        ----------
        tr : float
            Reduced temperature (defined as Tr = T / Tc).
        a : float
            Gasem model constant.
        b : float
            Gasem model constant.
        c : float
            Gasem model constant.

        Returns
        -------
        float
            Gasem's alpha function evaluated at tr.

        Notes
        -----
        Initial guess for nonpolar compounds: a = 2.0, b = 0.836, c = 0.134+0.508*acentric+-0.0467*acentric**2
        For hydrogen: a = -4.3, b = 10.4

        References
        ----------
        [1] Gasem, K. A. M.; Gao, W.; Pan, Z.; Robinson R. L. A modified temperature dependence for the Peng-Robinson
        equation of state. Fluid Phase Equilib. 2001, 181, 113-125.
        """
        return np.exp((a + b * tr) * (1.0 - tr ** c))

    def _twu(self, tr, l, m, n):
        """Twu alpha function.

        Parameters
        ----------
        tr : float
            Reduced temperature (defined as Tr = T / Tc).
        l : float
            Twu model constant.
        m : float
            Twu model constant.
        n : float
            Twu model constant.

        Returns
        -------
        float
            Twu's alpha function evaluated at tr.

        Notes
        -----
        Generic parameters are unavailable.  However, there are large tables of pure component parameters that can be
        used as a starting point for further optimization [2]_.

        References
        ----------
        [1] Twu, C. H.; Bluck, D.; Cunningham, J. R.; Coon, J. E. A cubic equation of state with a new alpha function
        and a new mixing rule. Fluid Phase Equilib. 1991, 69, 33-50.
        [2] Le Guennec, Y.; Privat, R.; Jaubert. J. N. Development of the translated-consistent tc-PR and tc-RK cubic
        equations of state for a safe and accurate prediction of volumetric, energetic and saturation properties of pure
        compounds in the sub- and super-critical domains. Fluid Phase Equilib. 2016, 429, 301-312.
        """
        return np.exp(l * (1.0 - tr ** (n * m))) * (tr ** (n * (m - 1.0)))

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

    def __eq__(self, other):
        if isinstance(other, CubicSpec):
            delta_1_eq = self.delta_1 == other.delta_1
            delta_2_eq = self.delta_2 == other.delta_2
            alpha_eq = self.alpha == other.alpha
            return delta_1_eq and delta_2_eq and alpha_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.delta_1, self.delta_2, self.alpha, self.vol_trans))

    def __str__(self):
        return "delta_1: ({}), delta_2: ({}), alpha: {}".format(self.delta_1,
                                                                self.delta_2,
                                                                self.alpha)


PR = CubicSpec(1.0 + 2 ** 0.5, 0.0 - 2 ** 0.5, 'soave')
GPR = CubicSpec(1.0 + 2 ** 0.5, 0.0 - 2 ** 0.5, 'gasem')
TPR = CubicSpec(1.0 + 2 ** 0.5, 0.0 - 2 ** 0.5, 'twu')
SRK = CubicSpec(1.0, 0.0, 'soave')
CPA = CubicSpec(1.0, 0.0, 'soave')


class CubicParms(object):
    """Cubic equation of state parameters"""


class CubicPhysInter(object):
    """Physical interactions between components."""


class CPA(EOS):
    """Generalized implementation of the Cubic Plus Association equation of state.

    Attributes:
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
