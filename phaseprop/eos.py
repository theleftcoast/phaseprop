"""Multiphase equilibrium and thermophysical property estimation.

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
MASS : dict
    Keys are units for mass and values are the conversion factor for that unit into kilograms (or 'kg')
LENGTH : dict
    Keys are units for length and values are the conversion factor for that unit into meters (or 'm').
AREA : dict
    Keys are units for area and the values are the conversion factor for that unit into square meters (or 'm2')
VOLUME : dict
    Keys are units for volume and values are the conversion factor for that unit into cubic meters (or 'm3')
TEMPERATURE : dict
    Keys are units for temperature and values are lambda functions (func(x) -> t) which convert that unit into kelvin
    (or 'K').
FORCE : dict
    Keys are units for force and values are the conversion factor for that unit into newtons (or 'N').
PRESSURE : dict
    Keys are units for pressure and values are the conversion factor for that unit into pascals (or 'Pa').
DENSITY : dict
    Keys are units for density and values are the conversion factor for that unit into kilograms per cubic meter (or
    'kg/m3').
ENERGY : dict
    Keys are units for energy and values are the conversion factor for that unit into joules (or 'J').
AMOUNT : dict
    Keys are units for amount of substance and values are the conversion factor for that unit into moles (or 'mol').
HEAT_CAPACITY : dict
    Keys are units for heat capacity and values are the conversion factor for that unit into 'J/mol.k'
GS : instance of PCSAFTSpec
    PC-SAFT universal constants (Gross and Sadowski, [2, 3]_).
PR : instance of CubicSpec
    Peng-Robinson equation of state (Peng and Robinson, [4]_).
GPR : instance of CubicSpec
    Peng-Robinson equation of state with the Gasem alpha function (Gasem et al., [5]_).
TPR : instance of CubicSpec
    Peng-Robinson equation of state with the Twu alpha function (Twu et al., [6]_).
SRK : instance of CubicSpec
    Soave-Redlich-Kwong equation of state (Soave, [7]_).
CPA : instance of CubicSpec
    Cubic-plus-association equation of state (Kontogeorgis, [8, 9]_).

Notes
-----
Constants and conversion factors taken from Perry's Chemical Engineer's Handbook [1]_.

References
----------
[1] Perry's Chemical Engineers' Handbook; Perry, R. H., Southard, M. Z., Eds.; McGraw-Hill Education: New York, 2019.
[2] Gross, J; Sadowski, G. Perturbed-Chain SAFT:  An Equation of State Based on a Perturbation Theory for Chain
Molecules. Ind. Eng. Chem. Res. 2001, 40, 1244–1260.
[3] Gross, J.; Sadowski, G. Application of the Perturbed-Chain SAFT Equation of State to Associating Systems. Ind. Eng.
Chem. Res. 2002, 41, 5510-5515.
[4] Peng, D. Y.; Robinson, D. B. A New Two-Constant Equation of State. Ind. Eng. Chem. Fundamen. 1976, 15, 59–64.
[5] Gasem, K. A. M.; Gao, W.; Pan, Z.; Robinson R. L. A modified temperature dependence for the Peng-Robinson equation
of state. Fluid Phase Equilib. 2001, 181, 113-125.
[6] Twu, C. H.; Bluck, D.; Cunningham, J. R.; Coon, J. E. A cubic equation of state with a new alpha function and a new
mixing rule. Fluid Phase Equilib. 1991, 69, 33-50.
[7] Soave, G. Equilibrium constants from a modified Redlich-Kwong equation of state. Fluid Phase Equilb. 1972, 27,
1197-1203.
[8] Kontogeorgis, G. M.; Voutsas, E. C.; Yakoumis, I. V.; Tassios, D. P. An Equation of State for Associating Fluids.
Ind. Eng. Chem. Res. 1996, 35, 4310-4318.
[9] Kontogeorgis, G. M.; Yakoumis, I. V.; Meijer, H.; Hendriks, E.; Moorwood, T. Multicomponent phase equilibrium
calculations for water–methanol–alkane mixtures. Fluid Phase Equilib. 1999, 158-160, 201-209.
"""

import numpy as np

# TODO: Check that value errors and type errors are raised consistently.

R = 8.31446261815324
EC = 1.602176634*10**-19
KB = 1.380649*10**-23
NA = 6.02214076*10**23
C = 299792458.0
F = 96485.33212
PI = 3.14159265358979323846
E0 = 8.8541878128*10**-12

MASS = {'lbm': 0.45359,
        'st': 907.18,
        'lt': 1016.0,
        'mt': 1000.0,
        'g': 0.001,
        'kg': 1.0}

LENGTH = {'ft': 0.3048,
          'in': 0.0254,
          'mi': 1609.344,
          'yd': 0.9144,
          'km': 1000.0,
          'cm': 0.01,
          'mm': 0.001,
          'm': 1.0}

AREA = {'sqft': 0.09290304,
        'ft2': 0.09290304,
        'sqyd': 0.8361274,
        'yd2': 0.8361274,
        'sqin': 0.00064516,
        'in2': 0.00064516,
        'sqcm': 0.0001,
        'cm2': 0.0001,
        'm2': 1.0}

VOLUME = {'cuft': 0.02831685,
          'ft3': 0.02831685,
          'usgal': 0.003785412,
          'ukgal': 0.004546092,
          'bbl': 0.1589873,
          'acre-ft': 1233.482,
          'l': 0.001,
          'm3': 1.0}

TEMPERATURE = {'F': lambda t: (t - 32.0) * 5.0 / 9.0 + 273.15,
               'R': lambda t: t / 1.8,
               'C': lambda t: t + 273.15,
               'K': lambda t: t}

FORCE = {'lbf': 4.448222,
         'dyne': 0.00001,
         'N': 1.0}

PRESSURE = {'psi': 6894.8,
            'atm': 101325.0,
            'mmhg': 133.32,
            'Pa': 1.0}

DENSITY = {'lbm/cuft': 16.01846,
           'lbm/ft3': 16.01846,
           'lbm/usgal': 119.8264,
           'lbm/ukgal': 99.77633,
           'kg/m3': 1.0}

MOLAR_DENSITY = {'kmol/m3': 1000.0,
                 'mol/m3': 1.0}

ENERGY = {'Btu': 1054.4,
          'J': 1.0}

AMOUNT = {'lbmmol': 453.5924,
          'stdm3': 44.6158,
          'stdft3': 1.1953,
          'kmol': 0.001,
          'mol': 1.0}

HEAT_OF_VAPORIZATION = {"J/kmol": 0.001,
                        "J/mol": 1.0}

HEAT_CAPACITY = {'J/kmol.K': 0.0001,
                 'J/mol.K': 1.0}

# Note that temperature is left out of this list because conversion is more than just multiplication by a constant.
UNITS = [MASS, LENGTH, AREA, VOLUME, FORCE, PRESSURE, DENSITY, MOLAR_DENSITY, ENERGY, AMOUNT,
         HEAT_OF_VAPORIZATION, HEAT_CAPACITY]

SI_UNITS = {'kg': MASS,
            'm': LENGTH,
            'm2': AREA,
            'm3': VOLUME,
            'K': TEMPERATURE,
            'N': FORCE,
            'Pa': PRESSURE,
            'kg/m3': DENSITY,
            'mol/m3': MOLAR_DENSITY,
            'J': ENERGY,
            'mol': AMOUNT,
            'J/mol': HEAT_OF_VAPORIZATION,
            'J/mol.K': HEAT_CAPACITY}


def conv_to_si(value, unit):
    """Convert input to corresponding SI unit.

    Handles cases where units are converted by scalar multiplication (i.e. everything but temperature conversion).

    Parameters
    ----------
    value : float, list of float, or tuple of float
        Input value(s) to be converted to corresponding SI unit.
    unit : str
        Unit of the input value.

    Returns
    -------
    float, list of float, or tuple of float
        Value(s) converted to corresponding SI unit.
    """
    if isinstance(value, float):
        for conv_dict in UNITS:
            if unit in conv_dict:
                return value * conv_dict[unit]
        raise ValueError("unit is not defined.")
    elif isinstance(value, (list, tuple)) and all(isinstance(x, float) for x in value):
        for conv_dict in UNITS:
            if unit in conv_dict:
                return [x * conv_dict[unit] for x in value]
        raise ValueError("input_unit is not defined.")
    else:
        raise TypeError("input must be a float, list of floats, or tuple of floats.")


def si_unit(conv_dict=None):
    """Find the SI unit for a given unit conversion dictionary.

    Parameters
    ----------
    conv_dict : dict
        Unit conversion dictionary (keys are units and values are conversion factors for the corresponding SI unit).

    Returns
    -------
    str
        SI unit corresponding to input unit conversion dictionary.
    """
    if not isinstance(conv_dict, dict):
        return TypeError("conv_dict must be a dictionary.")
    elif conv_dict not in UNITS:
        return ValueError("conv_dict must be a pre-defined unit conversion dictionary.")
    else:
        for key, value in SI_UNITS.items():
            if value == conv_dict:
                return key
        return RuntimeError("conv_dict does not have a defined SI value.")


# TODO: Consider eliminating pre-defined EOS in favor of identification using spec objects instead.
# Pre-defined equation of state (EOS) pick lists.
CUBIC_EOS = ['PR', 'GPR', 'TPR', 'SRK', 'CPA']
PC_SAFT_EOS = ['PC-SAFT', 'sPC-SAFT']
PHYS_EOS = CUBIC_EOS + PC_SAFT_EOS
ASSOC_EOS = ['CPA', 'PC-SAFT', 'sPC-SAFT']

# Pre-defined component family pick list.  Pick list are defined based on the the following references:
#
#   (1) Perry's Chemical Engineer's Handbook, 9th ed.
#
#   (2) Tihic, A.; Kontogeorgis, G. M.; von Solms, N.;Michelsen, M. L. Applications of the simplified perturbed-chain
#   SAFT equation of state using an extended parameter table. Fluid Phase Equilib. 2006, 248, 29-43.
COMP_FAM = ['Alkanes', 'Alkenes', 'Alkynes', 'Cycloalkanes' 'Aromatics', 'Polynuclear Aromatics', 'Aldehydes',
            'Ketones', 'Heterocyclics', 'Elements', 'Alcohols', 'Phenols', 'Ethers', 'Acids', 'Esters', 'Amines',
            'Amides', 'Nitriles', 'Nitro Compounds', 'Isocyanates', 'Mercaptans', 'Sulfides',
            'Halogenated Hydrocarbons', 'Silanes', 'Inorganics', 'Multifunctional']


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
        return (1.0 + m * (1.0 - tr**0.5))**2.0

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
        return np.exp((a + b*tr) * (1.0 - tr**c))

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
        return np.exp(l * (1.0 - tr**(n * m))) * (tr**(n * (m - 1.0)))

    def _d_1(self):
        """Intermediate variable used to evaluate critical properties."""
        return self.delta_1 + self.delta_2

    def _d_2(self):
        """Intermediate variable used to evaluate critical properties."""
        return self.delta_1 * self.delta_2

    def _y(self):
        """Intermediate variable used to evaluate critical properties."""
        a = ((1.0 + self.delta_2) * (1.0 + self.delta_1)**2.0)**(1.0 / 3.0)
        b = ((1.0 + self.delta_1) * (1.0 + self.delta_2)**2.0)**(1.0 / 3.0)
        return 1.0 + a + b

    def _gamma_c(self):
        """Intermediate variable used to evaluate critical properties."""
        numerator = 3.0*self.y()**2.0 + 3.0*self.y()*self.d_1() + self.d_1()**2.0 - self.d_2()
        denominator = 3.0*self.y() + self.d_1() - 1.0
        return numerator / denominator

    def _z_c(self):
        """Critical compressibility factor."""
        return self.y() / (3.0*self.y() + self.d_1() - 1.0)

    def _z_b(self):
        """Intermediate variable used to evaluate critical properties."""
        return 1.0 / (3.0*self.y() + self.d_1() - 1.0)

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


class PCSAFTSpec(object):
    """Constants that define a specific version of the PC-SAFT equation of state."""
    def __init__(self, a=None, b=None):
        """
        Parameters
        ----------
        a : np.ndarray
            2D array with shape (7, 3) containing data of 'float' type.
        b : np.ndarray
            2D array with shape (7, 3) containing data of 'float' type.
        """
        if a is None or b is None:
            raise ValueError("Inputs are not sufficient to define an instance of sPCSAFTSpec.")
        else:
            self.a = a
            self.b = b

    @property
    def a(self):
        """Universal model constants for the first-order perturbation dispersion term."""
        return self._a

    @a.setter
    def a(self, value):
        try:
            self._a
        except AttributeError:
            if isinstance(value, np.ndarray) and value.shape == (7, 3):
                self._a = value
            else:
                raise ValueError("a must be an np.ndarray with shape (7, 3).")

    @property
    def b(self):
        """Universal model constants for the second-order perturbation dispersion term."""
        return self._b

    @b.setter
    def b(self, value):
        try:
            self._b
        except AttributeError:
            if isinstance(value, np.ndarray) and value.shape == (7, 3):
                self._b = value
            else:
                raise ValueError("a must be an np.ndarray with shape (7, 3).")

    def __eq__(self, other):
        if isinstance(other, PCSAFTSpec):
            a_eq = np.array_equal(self.a, other.a)
            b_eq = np.array_equal(self.b, other.b)
            return a_eq and b_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.a, self.b))

    def __str__(self):
        return "a: {} \nb: {}".format(self.a, self.b)


# PC-SAFT universal constants (Gross and Sadowski).
gs_a0 = [0.9105631445, -0.3084016918, -0.0906148351]
gs_a1 = [0.6361281449, 0.1860531159, 0.4527842806]
gs_a2 = [2.6861347891, -2.5030047259, 0.5962700728]
gs_a3 = [-26.547362491, 21.419793629, -1.7241829131]
gs_a4 = [97.759208784, -65.25588533, -4.1302112531]
gs_a5 = [-159.59154087, 83.318680481, 13.77663187]
gs_a6 = [91.297774084, -33.74692293, -8.6728470368]
gs_a = np.array([gs_a0, gs_a1, gs_a2, gs_a3, gs_a4, gs_a5, gs_a6])

gs_b0 = [0.7240946941, -0.5755498075, 0.0976883116]
gs_b1 = [2.2382791861, 0.6995095521, -0.2557574982]
gs_b2 = [-4.0025849485, 3.892567339, -9.155856153]
gs_b3 = [-21.003576815, -17.215471648, 20.642075974]
gs_b4 = [26.855641363, 192.67226447, -38.804430052]
gs_b5 = [206.55133841, -161.82646165, 93.626774077]
gs_b6 = [-355.60235612, -165.20769346, -29.666905585]
gs_b = np.array([gs_b0, gs_b1, gs_b2, gs_b3, gs_b4, gs_b5, gs_b6])

GS = PCSAFTSpec(a=gs_a, b=gs_b)


class Comp(object):
    """A pure chemical component."""
    def __init__(self, name=None):
        """
        Parameters
        ----------
        name : str
        """
        # General Attributes
        if name is None:
            raise ValueError("Comp object must be initialized with a name and molecular weight.")
        else:
            # Metadata and constants.
            self.name = name
            self.cas_no = None
            self.formula = None
            self.family = None
            self.mw = None
            self.vdwv = None
            self.vdwa = None
            self.rgyr = None
            self.dipole = None
            self.quadrupole = None
            self.acentric = None
            self.tc = None
            self.pc = None
            self.vc = None
            self.rhoc = None
            self.tt = None
            self.pt = None
            self.bp = None
            self.mp = None
            self.hfus = None
            self.sfus = None
            self.hsub = None
            self.ssub = None
            # TODO:  Implement gibbs energy of formation, entropy, and enthalpy of combustion.

            # Temperature dependent properties.
            self.pvap_s = None
            self.pvap_l = None
            self.hvap_l = None
            self.den_s = None
            self.den_l = None
            self.beta_s = None
            self.beta_l = None
            self.cp_s = None
            self.cp_l = None
            self.cp_ig = None
            self.visc_l = None
            self.visc_ig = None
            self.tcond_l = None
            self.tcond_ig = None
            self.sigma = None

            # TODO: Ensure the cubic equation of state physical terms are built with getter/setter checks for CEOS objects.
            # Cubic EOS physical parameter dictionaries.
            self.srk_phys = {'a': None, 'b': None, 'm': None}
            self.cpa_phys = {'a0': None, 'b': None, 'c1': None}
            self.pr_phys = {'a': None, 'b': None, 'm': None}
            self.gpr_phys = {'a': None, 'b': None, 'a': None, 'b': None, 'c': None}
            self.tpr_phys = {'a': None, 'b': None, 'l': None, 'm': None, 'n': None}

            # SAFT EOS physical parameter dictionaries.
            self.spc_saft_phys = {'m': None, 'sig': None, 'eps': None}
            self.pc_saft_phys = {'m': None, 'sig': None, 'eps': None}

            # SAFT EOS association parameter objects.
            self.assoc_sites = None
            self.cpa_assoc = None
            self.spc_saft_assoc = None
            self.pc_saft_assoc = None

    @property
    def name(self):
        """str : Name of chemical compound.

        Value can only be set whe creating a new Comp instance.
        """
        return self._name

    @name.setter
    def name(self, value):
        try:
            self._name
        except AttributeError:
            if isinstance(value, str):
                self._name = value
            else:
                raise TypeError("name must be a string.")

    @property
    def cas_no(self):
        """str: Chemical Abstracts Service Registry Number."""
        return self._cas_no

    @cas_no.setter
    def cas_no(self, value):
        if isinstance(value, str) or value is None:
            self._cas_no = value
        else:
            raise TypeError("cas_no must be a string.")

    @property
    def formula(self):
        """str : Chemical formula."""
        return self._formula

    @formula.setter
    def formula(self, value):
        if isinstance(value, str) or value is None:
            self._formula = value
        else:
            raise TypeError("formula must be a string.")

    @property
    def family(self):
        """str : Chemical family."""
        return self._family

    @family.setter
    def family(self, value):
        if isinstance(value, str) or value is None:
            if value in COMP_FAM or value is None:
                self._family = value
            else:
                raise ValueError("family must be one of the pre-defined values.")
        else:
            raise TypeError("family must be a string.")

    @property
    def mw(self):
        """float : Molecular weight, g/mol."""
        return self._mw

    @mw.setter
    def mw(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._mw = value
        else:
            raise TypeError("mw must be a positive float.")

    @property
    def vdwv(self):
        """float : Van der Waal's volume, unit TBD."""
        return self._vdwv

    @vdwv.setter
    def vdwv(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._vdwv = value
        else:
            raise TypeError("vdwv must be a positive float.")

    @property
    def vdwa(self):
        """float : Van der Waal's surface area, unit TBD."""
        return self._vdwa

    @vdwa.setter
    def vdwa(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._vdwa = value
        else:
            raise TypeError("vdwa must be a positive float.")

    @property
    def rgyr(self):
        """float : Radius of gyration, unit TBD."""
        return self._rgyr

    @rgyr.setter
    def rgyr(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._rgyr = value
        else:
            raise TypeError("rgyr must be a positive float.")

    @property
    def dipole(self):
        """float : Gas phase dipole moment, unit TBD."""
        return self._dipole

    @dipole.setter
    def dipole(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._dipole = value
        else:
            raise TypeError("dipole must be a positive float.")

    @property
    def quadrupole(self):
        """float : Gas phase quadrupole moment, unit TBD."""
        return self._quadrupole

    @quadrupole.setter
    def quadrupole(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._quadrupole = value
        else:
            raise TypeError("quadrupole must be a positive float.")

    @property
    def acentric(self):
        """float : Acentric factor, dimensionless."""
        return self._acentric

    @acentric.setter
    def acentric(self, value):
        if isinstance(value, float) or value is None:
            self._acentric = value
        else:
            raise TypeError("acentric must be a float.")

    @property
    def tc(self):
        """float : Critical temperature, K."""
        return self._tc

    @tc.setter
    def tc(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._tc = value
        else:
            raise TypeError("tc must be a positive float.")

    @property
    def pc(self):
        """float : Critical pressure, Pa."""
        return self._pc

    @pc.setter
    def pc(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._pc = value
        else:
            raise TypeError("pc must be a positive float.")

    @property
    def vc(self):
        """float : Critical volume, m**3/mol."""
        return self._vc

    @vc.setter
    def vc(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._vc = value
        else:
            raise TypeError("vc must be a positive float.")

    @property
    def rhoc(self):
        """float : Critical density, mol/m**3"""
        return self._rhoc

    @rhoc.setter
    def rhoc(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._rhoc = value
        else:
            raise TypeError("rhoc must be a positive float.")

    @property
    def tt(self):
        """float : Triple point temperature, K."""
        return self._tt

    @tt.setter
    def tt(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._tt = value
        else:
            raise TypeError("tt must be a positive float.")

    @property
    def pt(self):
        """float : Triple point pressure, Pa."""
        return self._pt

    @pt.setter
    def pt(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._pt = value
        else:
            raise TypeError("pt must be a positive float.")

    @property
    def bp(self):
        """float : Boiling point, K."""
        return self._bp

    @bp.setter
    def bp(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._bp = value
        else:
            raise TypeError("bp must be a positive float.")

    @property
    def mp(self):
        """float : Melting point, K."""
        return self._mp

    @mp.setter
    def mp(self, value):
        if (isinstance(value, float) and value >= 0.0) or value is None:
            self._mp = value
        else:
            raise TypeError("mp must be a positive float.")

    @property
    def hfus(self):
        """float : Enthalpy of solid-liquid fusion, J/mol."""
        return self._hfus

    @hfus.setter
    def hfus(self, value):
        if isinstance(value, float) or value is None:
            self._hfus = value
        else:
            raise TypeError("comp object hfus must be a float.")

    @property
    def sfus(self):
        """float : Entropy of solid-liquid fusion, J/mol.K"""
        return self._sfus

    @sfus.setter
    def sfus(self, value):
        if isinstance(value, float) or value is None:
            self._sfus = value
        else:
            raise TypeError("sfus must be a float.")

    @property
    def hsub(self):
        """float : Enthalpy of sublimation, J/mol."""
        return self._hsub

    @hsub.setter
    def hsub(self, value):
        if isinstance(value, float) or value is None:
            self._hsub = value
        else:
            raise TypeError("hsub must be a float.")

    @property
    def ssub(self):
        """float : Entropy of sublimation, J/mol.K"""
        return self._ssub

    @ssub.setter
    def ssub(self, value):
        if isinstance(value, float) or value is None:
            self._ssub = value
        else:
            raise TypeError("ssub must be a float.")

    @property
    def pvap_s(self):
        """float : Solid vapor pressure (i.e. sublimation pressure), Pa."""
        return self._pvap_s

    @pvap_s.setter
    def pvap_s(self, value):
        if isinstance(value, Corel) or value is None:
            self._pvap_s = value
        else:
            raise TypeError("pvap_s must be an instance of Corel.")

    @property
    def pvap_l(self):
        """float : Liquid vapor pressure, Pa."""
        return self._pvap_l

    @pvap_l.setter
    def pvap_l(self, value):
        if isinstance(value, Corel) or value is None:
            self._pvap_l = value
        else:
            raise TypeError("pvap_l must be an instance of Corel.")

    @property
    def hvap_l(self):
        """float : Enthalpy of saturated liquid vaporization, J/mol."""
        return self._hvap_l

    @hvap_l.setter
    def hvap_l(self, value):
        if isinstance(value, Corel) or value is None:
            self._hvap_l = value
        else:
            raise TypeError("hvap_l must be an instance of Corel.")

    @property
    def den_s(self):
        """float : Solid density, kg/m3 or mol/m3."""
        return self._den_s

    @den_s.setter
    def den_s(self, value):
        if isinstance(value, Corel) or value is None:
            self._den_s = value
        else:
            raise TypeError("den_S must be an instance of Corel.")

    @property
    def den_l(self):
        """float : Liquid density, kg/m3 or mol/m3."""
        return self._den_l

    @den_l.setter
    def den_l(self, value):
        if isinstance(value, Corel) or value is None:
            self._den_l = value
        else:
            raise TypeError("den_l must be an instance of Corel.")

    @property
    def beta_s(self):
        """float : Isothermal solid compressibility, Unit TBD.

        beta = -(1/V)*(dV/dP)"""
        return self._beta_s

    @beta_s.setter
    def beta_s(self, value):
        if isinstance(value, Corel) or value is None:
            self._beta_s = value
        else:
            raise TypeError("beta_s must be an instance of Corel.")

    @property
    def beta_l(self):
        """float : Isothermal liquid compressibility, Unit TBD.

        beta = -(1/V)*(dV/dP)"""
        return self._beta_l

    @beta_l.setter
    def beta_l(self, value):
        if isinstance(value, Corel) or value is None:
            self._beta_l = value
        else:
            raise TypeError("beta_l must be an instance of Corel.")

    @property
    def cp_s(self):
        """float : Solid heat capacity, J/mol.K."""
        return self._cp_s

    @cp_s.setter
    def cp_s(self, value):
        if isinstance(value, Corel) or value is None:
            self._cp_s = value
        else:
            raise TypeError("cp_s must be an instance of Corel.")

    @property
    def cp_l(self):
        """float : Saturated liquid heat capacity, J/mol.K."""
        return self._cp_l

    @cp_l.setter
    def cp_l(self, value):
        if isinstance(value, Corel) or value is None:
            self._cp_l = value
        else:
            raise TypeError("cp_l must be an instance of Corel.")

    @property
    def cp_ig(self):
        """float : Ideal gas heat capacity, J/mol.K."""
        return self._cp_ig

    @cp_ig.setter
    def cp_ig(self, value):
        if isinstance(value, Corel) or value is None:
            self._cp_ig = value
        else:
            raise TypeError("cp_ig must be an instance of Corel.")

    @property
    def visc_l(self):
        """float : Saturated liquid viscosity, unit TBD."""
        return self._visc_l

    @visc_l.setter
    def visc_l(self, value):
        if isinstance(value, Corel) or value is None:
            self._visc_l = value
        else:
            raise TypeError("visc_l must be an instance of Corel.")

    @property
    def visc_ig(self):
        """float : Ideal gas viscosity, unit TBD."""
        return self._visc_ig

    @visc_ig.setter
    def visc_ig(self, value):
        if isinstance(value, Corel) or value is None:
            self._visc_ig = value
        else:
            raise TypeError("visc_ig must be an instance of Corel.")

    @property
    def tcond_l(self):
        """float : Saturated liquid thermal conductivity, unit TBD."""
        return self._tcond_l

    @tcond_l.setter
    def tcond_l(self, value):
        if isinstance(value, Corel) or value is None:
            self._tcond_l = value
        else:
            raise TypeError("tcond_l must be an instance of Corel.")

    @property
    def tcond_ig(self):
        """float : Ideal gas thermal conductivity, unit TBD."""
        return self._tcond_ig

    @tcond_ig.setter
    def tcond_ig(self, value):
        if isinstance(value, Corel) or value is None:
            self._tcond_ig = value
        else:
            raise TypeError("tcond_ig must be an instance of Corel.")

    @property
    def sigma(self):
        """float : Surface tension, unit TBD."""
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if isinstance(value, Corel) or value is None:
            self._sigma = value
        else:
            raise TypeError("sigma must be an instance of Corel.")

    def k_wilson(self, p=None, t=None):
        """Wilson's equilibrium ratio (Ki = yi / xi).

        Parameters
        ----------
        p : float
            Pressure, Pa.
        t : float
            Temperature, K.

        Returns
        -------
        float
            Wilson's K-factor.
        """
        if isinstance(p, float) and isinstance(t, float) and p > 0.0 and t > 0.0:
            return (self._pc / p) * np.exp(5.37 * (1.0 + self._acentric) * (1.0 - self._tc / t))
        else:
            raise RuntimeError("p and t must both be positive floats.")

    def density(self, p=None, t=None, phase=None, spec=None):
        """Pure component liquid and/or solid density estimation.

        Parameters
        ----------
        p : float
            Pressure, K.
        t : float
            Temperature, K.
        phase : str
            Phase specification (either 's' or 'l').

        Returns
        -------
        float
            Density for specified condition.
        """
        # TODO: Purpose is to combine isothermal compressibility and density equations into one function.
        # TODO: Purpose is to evaluate both solid and/or liquid phase density and return whichever is stable.
        return

    # TODO: Improve interface by making these checks part of getter/setter methods?
    def _check_assoc_sites(self):
        # Check to ensure there are no duplicate association sites.
        if self.assoc_sites is not None:
            if not isinstance(self.assoc_sites, (list, tuple)):
                raise TypeError("Comp objects must be a list of unique assoc_site objects.")
            elif len(self.assoc_sites) != len(set(self.assoc_sites)):
                raise ValueError("Comp objects must be a list of unique assoc_site objects.")
            else:
                pass

    def _check_assoc_parameters(self):
        # Check to ensure association parameter are lists of AssocSiteInter objects.
        if self.assoc_sites is not None:
            if self.cpa_assoc is not None:
                for asi in self.cpa_assoc:
                    if not isinstance(asi, AssocSiteInter):
                        raise TypeError("Association parameter lists must contain AssocSiteInter objects.")

    def __eq__(self, other):
        if isinstance(other, Comp):
            name_eq = self.name == other.name
            return name_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "Name: {}, CAS No.: {}".format(self.name, self.cas_no)


class PseudoComp(object):
    """A pseudo-component (polymer, distillation cut, or asphaltene).

    Notes
    -----
    # TODO: Improve agreement bewteeen Comp and PseudoComp classes. Currently a skeleton implementation.
    # TODO: Add Riazi correlations to estimate Tc, Pc, Vc, Rhoc from mw, sg, NBP.
    """

    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
        """
        # General Attributes
        self.name = name
        self.mw = None
        self.sg = None
        self.nbp = None
        self.acentric = None
        self.tc = None
        self.pc = None
        self.vc = None
        self.rhoc = None

        # EOS Physical Attributes
        self.srk_phys = None
        self.pr_phys = None
        self.gpr_phys = None
        self.cpa_phys = None
        self.spc_saft_phys = None

        # EOS Association Attributes
        self.assoc_sites = None
        self.cpa_assoc = None
        self.spc_saft_assoc = None

    @property
    def name(self):
        """str : Name of pseudo-component.

        Value can only be set whe creating a new Comp instance.
        """
        return self._name

    @name.setter
    def name(self, value):
        try:
            self._name
        except AttributeError:
            self._name = value

    def __eq__(self, other):
        if isinstance(other, PseudoComp):
            name_eq = self.name == other.name
            return name_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "Name: {}, MW: {}, SG: {}, NBP: {}".format(self.name, self.mw, self.sg, self.nbp)


class CompSet(object):
    """Set of components or pseudo-components.

    Notes
    -----
    """
    def __init__(self, comps=None):
        """
        Parameters
        ----------
        comps : list or tuple of Comp objects
        """
        if comps is None:
            raise ValueError("comps must be provided to create an instance of CompSet.")
        else:
            self.comps = comps

    @property
    def comps(self):
        """list or tuple of Comp objects : A collection of Comp objects.

        Values can only be set whe creating a new CompSet instance.
        """
        return self._comps

    @comps.setter
    def comps(self, value):
        try:
            self._comps
        except AttributeError:
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    raise ValueError("comps must contain at least one Comp object.")
                elif len(value) != len(set(value)):
                    raise ValueError("comps cannot contain duplicate Comp objects.")
                elif all(isinstance(item, (Comp, PseudoComp)) for item in value):
                    self._comps = value
                else:
                    raise TypeError("comps must contain only Comp or PseudoComp objects.")
            else:
                raise TypeError("comps must be a list or tuple.")

    @property
    def size(self):
        """int : The number of Comp or PseudoComp objects in 'comps'."""
        return len(self._comps)

    @property
    def can_associate(self):
        """list of bool : Boolean indicating if Comp or PseudoComp objects in 'comps' can associate."""
        result = []
        for comp in self._comps:
            if comp.assoc_sites is not None:
                result.append(True)
            else:
                result.append(False)
        return result

    @property
    def mw(self):
        """list of float or None : Molecular weight for each Comp or PseudoComp objects in 'comps'.

        Returns None if any Comp or PseudoComp object is missing molecular weight."""
        result = []
        for comp in self._comps:
            if comp.mw is None:
                return None
            else:
                result.append(comp.mw)
        return np.array(result)

    def __eq__(self, other):
        if isinstance(other, CompSet):
            return self.comps == other.comps
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._comps)

    def __str__(self):
        result = []
        for comp in self._comps:
            result.append(comp.name)
        return ",".join(tuple(result))


class Callback(list):
    """Utility for managing callback execution.

    References
    ----------
    [1] http://web.archive.org/web/20060612061259/http://www.suttoncourtenay.org.uk/duncan/accu/pythonpatterns.html
    """
    def __init__(self):
        self._delegates = []

    @property
    def delegates(self):
        """list of callable : List of callback functions."""
        return self._delegates

    def add(self, callback):
        """Add a function to delegates.

        Parameters
        ----------
        callback : callable
            Function to be added to delegates.
        """
        if callable(callback):
            self._delegates.append(callback)

    def remove(self, callback):
        """Remove a function from delegates.

        Parameters
        ----------
        callback : callable
            Function to be removed from delegates.
        """
        if callable(callback):
            for i, d in enumerate(self._delegates):
                if d == callback:
                    del self._delegates[i]

    def fire(self, *args, **kwargs):
        """Execute all callback functions in delegates.

        Parameters
        ----------
        *args : list
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        for d in self._delegates:
            d(*args, **kwargs)


class State(object):
    """Intensive PVT and association state of a system.

    Methods
    -------
    set(p, vm, t)
        Set specification variables and trigger update for other dependent variables and objects.
    is_consistent(other)
        Check if two State objects have the same state definition.
    align_mixture()
        Sets State definition for all Phases to the State definition of the corresponding Mixture.
    defined()
        Check to see if instance is fully defined.

    Notes
    -----
    TODO: Generalize this class a bit?  A state can conceptually exist independently of a system.
    """
    def __init__(self, system=None, spec=None):
            """
            Parameters
            ----------
            system : Phase or Mixture object
            spec : str
            """
            if system is None:
                raise RuntimeError("Must pass an instance of Phase or Mixture to the constructor.")
            else:
                self.system = system
                self.spec = spec
                self._callback = Callback()
                self._p = None
                self._vm = None
                self._t = None
                self._xai = None

    @property
    def system(self):
        """Phase or Mixture object : The phase or mixture associated with a State object.

        Can only be set whe creating a new State instance."""
        return self._system

    @system.setter
    def system(self, value):
        try:
            self._system
        except AttributeError:
            if isinstance(value, (Phase, Mixture)):
                self._system = value
            else:
                raise TypeError("phase must be an instance of Phase or Mixture.")

    @property
    def spec(self):
        """str : Specification of either the 'PT' or 'VT' variable pair that specifies the system.

        Two state variables must be specified to define the PVT state of a system.  However, not all combinations are
        valid or supported. Only the most common pressure-temperature (PT) and volume-temperature(VT) specifications
        are currently implemented.
        """
        return self._spec

    @spec.setter
    def spec(self, value):
        if value is None:
            self._spec = None
            self._reset()
        elif value in ['PT', 'VT']:
            self._spec = value
            self._reset()
        else:
            raise ValueError("spec must be either PT or VT.")

    @property
    def callback(self):
        """Callback object : Manage functions from other classes that should be called if State changes."""
        return self._callback

    @property
    def p(self):
        """float : Pressure, Pa."""
        return self._p

    @property
    def vm(self):
        """float : Molar volume, m**3/mol."""
        return self._vm

    @property
    def t(self):
        """float : Temperature, K."""
        return self._t

    @property
    def xai(self):
        """list of lists : Fraction of sites not bonded for association sites in mixture.

        xai is a list of lists with the following structure:

            xai = [[xa for each site in Comp 1], [xa for each site in Comp 2], ..., [xa for each site in Comp n]]

        The length of xai is equal to the length of the associated CompSet.  The length of each element corresponds
        to the length of the assoc_sites attribute of each Comp in the associated CompSet.  If there are no association
        sites for a compound, then the corresponding element must have a truth value of False (which is represented
        by None this library for simplicity). If there are association sites for a compound, then the fraction of sites
        not bonded can take on a value of 0.0 <= xa <= 1.0.
        """
        return self._xai

    def set(self, p=None, vm=None, t=None):
        """Set specification variables and trigger update for other dependent variables and objects.

        This method takes 'p' & 't' and calculates 'vm' (for spec='PT') or takes 'vm' & 't' and calculates 'p' (for
        spec='VT'). Calculation of the associated dependent variable is done by the equation of state that is part of
        the associated system.  If the equation of state accounts for intermolecular association, then the resulting
        free site fraction estimate will be loaded into 'xai'. If the equation of state does not account for
        intermolecular association, then 'None' will be loaded into 'xai'.

        This method also manages callbacks to other objects that must be synchronized if State changes.

        Parameters
        ----------
        p : float or None
            Pressure, Pa.
        vm : float or None
            Molar volume, m**3/mol.
        t : float or None
            Temperature, K.
        """
        if self._spec is None:
            raise RuntimeError("spec must be initialized prior to setting p, vm, or t.")

        # Evaluate if each input is correctly defined.
        p_def = p is not None and isinstance(p, float) and p > 0.0
        vm_def = vm is not None and isinstance(vm, float) and vm > 0.0
        t_def = t is not None and isinstance(t, float) and t > 0.0

        # Set state variables.
        if self._spec == 'PT':
            # Check if state variables are correctly defined and proceed with update if so.
            if p_def:
                self._p = p
            if t_def:
                self._t = t

            # Update volume.
            if self._p is not None and self._t is not None and self._vm is not None:
                # Use prior volume as initial guess (time savings in vast majority of cases).
                # TODO: Not implemented yet. Need to define limits on how 'close' the prior state must be to justify
                #  using the prior value. A conservative suggestion is that 'p' and 't' are unchanged and the norm of
                #  the new and prior composition vector is less than some cutoff value (i.e. the composition only
                #  changed a little bit).  The norm of the composition change could be passed to State by the Compos
                #  callback.  This approach caters to the most common case where performance is critical...the 'PT
                #  Flash' calculation.  In this circumstance, 'p' and 't' remain unchanged with relatively small updates
                #  to composition at each iteration step.
                self._pt_vol_update(spec='full')
            elif self._p is not None and self._t is not None:
                # Using both liquid and vapor guesses. Choose volume with minimum gibbs energy.
                self._pt_vol_update(spec='full')
        elif self.spec == 'VT':
            # Check if state variables are correctly defined and proceed with update if so.
            if vm_def:
                self._vm = vm
            if t_def:
                self._t = t
            if self._vm is not None and self._t is not None:
                # TODO:  Implement pressure update.  VT specifiation and pressure update is low-priority and can wait
                #  until later.
                return # TBD
            else:
                raise ValueError("vm and t must be positive floats.")
        else:
            raise RuntimeError("State specification not valid.")

    def _pt_vol_update(self, spec='full'):
        if spec == 'full' and self.defined():
            # Update volume using default liquid and vapor initial guesses.
            # TODO: Add functionality to pass back xai with vol_solver.
            vm_liq, gr_liq = self._system.eos.vol_solver(p=self._p, t=self._t, ni=self._system.compos.xi,
                                                         root='liquid')
            vm_vap, gr_vap = self._system.eos.vol_solver(p=self._p, t=self._t, ni=self._system.compos.xi,
                                                         root='vapor')
            # Choose volume with minimum residual gibbs energy.
            if gr_liq < gr_vap:
                self._vm = vm_liq
            else:
                self._vm = vm_vap
            return
        if spec == 'prior' and self.defined():
            # Update volume using prior value as initial guess.
            vm, gr = self._system.eos.vol_solver(p=self._p, t=self._.t, ni=self._system.compos.xi,
                                                 root='liquid', prior=self._vm)
            self._vm = vm
            return
        else:
            raise ValueError("spec must be either full or prior.")

    def _reset(self):
        self._p = None
        self._vm = None
        self._t = None
        self._xai = None

    def is_consistent(self, other):
        """Check if two State objects have the same state definition.

        This method does not check if two State objects are completely identical (every attribute is the same).  Rather,
        it checks if the state spec (either 'PT' or 'VT') and the corresponding specification variables ('p' & 't' or
        'vm' & 't') are identical.  This is useful because State objects may have the same definition, but different
        values for dependent variables (due to different equation of state methods) or no values for dependent variables
        (if no equation of state was specified).

        Parameters
        ----------
        other : State object
            The State object to be compared with 'self'.

        Returns
        -------
        bool
            True if 'other' is consistent with 'self'.
        """
        if isinstance(other, State):
            spec_eq = self._spec == other.spec
            if self._spec == 'PT':
                p_eq = self._p == other.p
                t_eq = self._t == other.t
                return spec_eq and p_eq and t_eq
            elif self._spec == 'VT':
                vm_eq = self._vm == other.vm
                t_eq = self._t == other.t
                return spec_eq and vm_eq and t_eq
            else:
                return False
        return False

    def align_mixture(self):
        """Sets State definition for all Phases to the State definition of the corresponding Mixture."""
        if isinstance(self._system, Mixture):
            for p in self._system.phases:
                if not self.is_consistent(p):
                    p.state.spec = self._spec
                    if self._spec == 'PT':
                        p.state.set(p=self._p, t=self._t)
                    elif self._spec == 'VT':
                        p.state.set(vm=self._vm, t=self._t)

    def defined(self):
        """Check to see if instance is fully defined.

        Returns
        -------
        bool
            True if instance is fully defined.
        """
        if self.spec == 'PT':
            p_def = self.p is not None
            t_def = self.t is not None
            return p_def and t_def
        elif self.spec == 'VT':
            vm_def = self.vm is not None
            t_def = self.t is not None
            return vm_def and t_def
        else:
            return False

    def __eq__(self, other):
        if isinstance(other, State):
            spec_eq = self.spec == other.spec
            p_eq = self.p == other.p
            t_eq = self.t == other.t
            vm_eq = self.vm == other.vm
            return spec_eq and p_eq and t_eq and vm_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.spec, self.p, self.vm, self.t))


class Extent(object):
    """Extent of a system (phase or mixture)."""
    def __init__(self, system=None):
        """
        Parameters
        ----------
        system : Phase or Mixture object
        """
        if system is None:
            raise ValueError("Must pass an instance of Phase or Mixture to the constructor.")
        else:
            self.system = system
            self.callback = Callback()
            self._ni = None
            self._n = None
            self._mi = None
            self._m = None
            self._v = None

    @property
    def system(self):
        """Phase or Mixture object : The phase or mixture associated with a State object.

        system can only be set whe creating a new State instance."""
        return self._system

    @system.setter
    def system(self, value):
        try:
            self._system
        except AttributeError:
            if isinstance(value, (Phase, Mixture)):
                self._system = value
            else:
                raise TypeError("system must be an instance of Phase or Mixture.")

    @property
    def ni(self):
        # Moles of each component.
        return self._ni

    @property
    def n(self):
        # Total moles.
        return self._n

    @property
    def mi(self):
        # Mass of each component (in kg).
        return self._mi

    @property
    def m(self):
        # Total mass (in kg).
        return self._m

    @property
    def v(self):
        # Units of volume are m**3.
        return self._v

    def set(self, n=None, m=None, v=None):
        # TODO: Add logic to update child phase extents if this is a mixture.
        # TODO: Add logic to callback parent mixture if this is a child phase.
        n_def = n is not None and isinstance(n, float) and n > 0.0
        m_def = m is not None and isinstance(m, float) and m > 0.0
        v_def = v is not None and isinstance(v, float) and v > 0.0

        if self._system.compos is None:
            compos_xi_def = False
            compos_wi_def = False
        else:
            compos_xi_def = self._system.compos.defined(spec='xi')
            compos_wi_def = self._system.compos.defined(spec='wi')

        if self._system.state is None:
            vm_def = False
        else:
            vm_def = self._state.vm is not None

        if n_def and not compos_xi_def:
            self._reset()
            self._n = n
        elif n_def and compos_xi_def:
            self._reset()
            self._xi_n_spec(n=n)
        elif m_def and not compos_wi_def:
            self._reset()
            self._m = m
        elif m_def and compos_wi_def:
            self._reset()
            self._wi_m_spec(m=m)
        elif v_def and not vm_def:
            self._reset()
            self._v = v
        elif v_def and vm_def:
            self._reset()
            self._vm_v_spec(v)
        else:
            raise RuntimeError("Invalid combination of specification variables.")

    def _xi_n_spec(self, n=None):
        if isinstance(n, float) and n >= 0.0:
            self._n = n
            self._ni = self._system.compos.xi * n
            if self._system.comps.mw is not None:
                # Convert grams into kilograms.
                self._mi = (self._ni * self._system.comps.mw) / 1000.0
                self._m = np.sum(self._mi)
        else:
            raise ValueError("n must be a positive float.")

    def _wi_m_spec(self, m=None):
        if isinstance(m, float) and m >= 0.0:
            self._m = m
            self._mi = self._system.compos.wi * m
            if self._system.comps.mw is not None:
                # Convert grams into kilograms.
                self._ni = 1000.0 * (self._mi / self._system.comps.mw)
                self._n = np.sum(self._ni)
        else:
            raise TypeError("m must be a positive float.")

    def _vm_v_spec(self, v=None):
        if isinstance(v, float) and v >= 0.0:
            raise TypeError("v must be a positive float.")
        else:
            self._n = v / self._system.state.vm
            if self._system.compos.defined(spec='xi'):
                self._ni = self._system.compos.xi * self._n
            if self._system.comps.mw is not None:
                # Convert grams into kilograms.
                self._mi = (self._ni * self._system.comps.mw) / 1000.0
                self._m = np.sum(self._mi)

    def _reset(self):
        self._ni = None
        self._n = None
        self._mi = None
        self._m = None
        self._v = None

    def _compos_callback(self):
        # Execute if Compos was updated.
        return

    def _mixture_extent_callback(self):
        # Execute if Mixture extent was updated.
        return

    def __eq__(self, other):
        if isinstance(other, Extent):
            system_eq = self.system == other.system
            ni_eq = self.ni == other.ni
            n_eq = self.n == other.n
            mi_eq = self.mi == other.mi
            m_eq = self.m == other.m
            v_eq = self.v == other.v
            return system_eq and ni_eq and n_eq and mi_eq and m_eq and v_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.system, self.ni, self.n, self.mi, self.m, self.v))

    def defined(self, spec=None):
        if spec is None or spec == 'full':
            if self._n is not None and self._m is not None and self._v is not None:
                return True
            else:
                return False
        elif spec == 'm':
            if self._m is not None:
                return True
            else:
                return False
        elif spec == 'n':
            if self._n is not None:
                return True
            else:
                return False
        elif spec == 'v':
            if self._v is not None:
                return True
            else:
                return False
        else:
            return False


class Compos(object):
    """Intensive composition of a system (phase or mixture).

    TODO: Add uncertainty for each compos. Could be used similarly to a Var for storing experimental data points.
    TODO: Generalize?  Does not have to be part of a system.  Could leave the initialization check for 'None' out and
     implement some safeguards elsewhere in the class.
    """
    def __init__(self, system=None):
        """
        Parameters
        ----------
        system : Phase or Mixture object
        """
        if system is None:
            raise ValueError("Must pass an instance of Phase or Mixture to the constructor.")
        else:
            self.system = system
            self._callback = Callback()
            self._xi = None
            self._wi = None
            if isinstance(system, Phase):
                self._phase_frac = None

    @property
    def system(self):
        """Phase or Mixture object : The phase or mixture associated with a Compos object.

        Can only be set whe creating a new State instance."""
        return self._system

    @system.setter
    def system(self, value):
        try:
            self._system
        except AttributeError:
            if isinstance(value, (Phase, Mixture)):
                self._system = value
            else:
                raise TypeError("phase must be an instance of Phase or Mixture.")

    @property
    def callback(self):
        """Callback object : Manage functions from other classes that should be called if state changes."""
        return self._callback

    @property
    def xi(self):
        """np.ndarray : Mole fraction of each component in phase or mixture."""
        return self._xi

    @property
    def wi(self):
        """np.ndarray : Mass fraction of each component in phase or mixture."""
        return self._wi

    @property
    def phase_frac(self):
        """float : Mole fraction of phase in a mixture.

        Only defined for Phases (not Mixtures) and is used extensively in phase equilibrium calculations.
        """
        return self._phase_frac

    @phase_frac.setter
    def phase_frac(self, value):
        if isinstance(self._system, Phase):
            if value is None:
                self._phase_frac = None
            elif isinstance(value, float) and 0.0 <= value <= 1.0:
                self._phase_frac = value
            else:
                raise TypeError("phase_frac must be a float between zero and one.")
        else:
            raise RuntimeError("phase_frac is only defined for Phases.")

    def set(self, xi=None, wi=None):
        """Set composition and trigger update for other dependent variables and objects.

        Parameters
        ----------
        xi : list, tuple, or np.ndarray
            Mole fraction for each component or pseudo-component in the associated CompSet.
        wi : list, tuple, or np.ndarray
            Mass fraction for each component or pseudo-component in the associated CompSet.
        """
        xi_def = xi is not None
        wi_def = wi is not None

        if xi_def is True and wi_def is False:
            self._reset()
            self._xi_spec(xi=xi)
        elif wi_def is True and xi_def is False:
            self._reset()
            self._wi_spec(wi=wi)
        else:
            raise ValueError("Must specify either xi or wi.")

    def _xi_spec(self, xi=None):
        if not isinstance(xi, (list, tuple, np.ndarray)):
            raise TypeError("xi must be a list, tuple, or np.ndarray.")
        elif not all(isinstance(i, (float, np.floating)) for i in xi):
            raise TypeError("xi can only contain floats.")
        elif not all(i >= 0.0 for i in xi):
            raise ValueError("xi can only contain positive floats.")
        elif len(xi) != self._system.comps.size:
            raise ValueError("xi must be the same size as the associated CompSet.")
        else:
            self._xi = np.array(xi) / np.sum(np.array(xi))
            if self._system.comps.mw is None:
                self._wi = None
                self._callback.fire()
            else:
                self._wi = (self._xi * self._system.comps.mw) / np.sum(self._xi * self._system.comps.mw)
                self._callback.fire()

    def _wi_spec(self, wi=None):
        if not isinstance(wi, (list, tuple, np.ndarray)):
            raise TypeError("wi must be a list, tuple, or np.ndarray.")
        elif not all(isinstance(i, (float, np.floating)) for i in wi):
            raise TypeError("wi can only contain floats.")
        elif not all(i >= 0.0 for i in wi):
            raise ValueError("wi can only contain positive floats.")
        elif len(wi) != self._system.comps.size:
            raise ValueError("wi must be the same size as the associated CompSet.")
        else:
            self._wi = np.array(wi) / np.sum(np.array(wi))
            if self._system.comps.mw is None:
                self._xi = None
                self._callback.fire()
            else:
                self._xi = (self._wi / self._system.comps.mw) / np.sum(self._wi / self._system.comps.mw)
                self._callback.fire()

    def _reset(self):
        self._xi = None
        self._wi = None

    def defined(self):
        """Check to see if instance is fully defined.

        Returns
        -------
        bool
            True if instance is fully defined.
        """
        if self._xi is not None and self._wi is not None:
            return True
        else:
            return False

    def __eq__(self, other):
        if isinstance(other, Compos):
            system_eq = self.system == other.system
            xi_eq = np.array_equal(self.xi, other.xi)
            wi_eq = np.array_equal(self.wi, other.wi)
            return system_eq and xi_eq and wi_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.system, self.xi, self.wi))


class Corel(object):
    """Pure component temperature dependent property."""
    def __init__(self, a=None, b=None, c=None, d=None, e=None, f=None, g=None, eq_id=None,
                 source_t_min=None, source_t_max=None, source_rmse=None, source_mae=None, source_mape=None,
                 source_t_unit=None, source_unit=None, source=None, notes=None):
        """
        Parameters
        ----------
        a : float or None
        b : float or None
        c : float or None
        d : float or None
        e : float or None
        f : float or None
        g : float or None
        t_min : float or None
        t_max : float or None
        eq_id : int
        source : str or None
        rmse : float or None
        mae : float or None
        mape : float or None
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g

        # Property correlations.
        self._correlation = {1: lambda t: np.exp(self._a +
                                                 self._b / t +
                                                 self._c * np.log(t) +
                                                 self._d * t ** self._e),
                             2: lambda t: self._a / self._b ** (1.0 + (1.0 - t / self._c) ** self._d),
                             3: lambda t: self._a +
                                          self._b * self._tau(t, self._f) ** 0.35 +
                                          self._c * self._tau(t, self._f) ** (2.0 / 3.0) +
                                          self._d * self._tau(t, self._f) +
                                          self._e * self._tau(t, self._f) ** (4.0 / 3.0),
                             4: lambda t: self._a * (1.0 - self._tr(t, self._f)) ** (self._b +
                                                                                     self._c*self._tr(t, self._f) +
                                                                                     self._d*self._tr(t, self._f)**2.0 +
                                                                                     self._e*self._tr(t, self._f)**3.0),
                             5: lambda t: self._a +
                                          self._b * t +
                                          self._c * t ** 2.0 +
                                          self._d * t ** 3.0 +
                                          self._e * t ** 4.0,
                             6: lambda t: (self._a ** 2.0) / self._tau(t, self._e) +
                                           self._b -
                                           2.0 * self._a * self._c * self._tau(t, self._e) -
                                           self._a * self._d * self._tau(t, self._e) ** 2.0 -
                                           (self._c ** 2.0) * (self._tau(t, self._e) ** 3.0) / 3.0 -
                                           (self._c * self._d * self._tau(t, self._e) ** 4.0) / 2.0 -
                                           (self._d ** 2.0) * (self._tau(t, self._e) ** 5.0) / 5.0,
                             7: lambda t: self._a +
                                          self._b * ((self._c / t) / np.sinh(self._c / t)) ** 2.0 +
                                          self._d * ((self._e / t) / np.cosh(self._e / t)) ** 2.0,
                             8: lambda t: (self._a * t ** self._b) / (1.0 + self._c / t + self._d / t ** 2.0)}

        # Property correlation derivatives evaluated using Wolfram Alpha.
        self._derivative = {1: None,
                            2: None,
                            3: None,
                            4: None,
                            5: None,
                            6: None,
                            7: lambda t: 2.0 * ((self._b * self._c ** 2.0) *
                                                (self._c / np.tanh(self._c / t) - t) *
                                                ((1.0 / np.sinh(self._c / t)) ** 2.0) +
                                                (self._d * self._e ** 2.0) *
                                                (self._e * np.tanh(self._e / t) - t) *
                                                ((1.0 / np.cosh(self._e / t)) ** 2.0)) / (t ** 4.0),
                            8: None}

        # Property correlation integrals evaluated using Wolfram Alpha.
        self._integral = {1: None,
                          2: None,
                          3: None,
                          4: None,
                          5: None,
                          6: None,
                          7: lambda t: self._a * t +
                                       self._b * self._c / np.tanh(self._c / t) -
                                       self._d * self._e * np.tanh(self._e / t),
                          8: None}

        # Required parameters for each correlating equation. Keys are eq_id and value lists specify if parameters a-g
        # are required to evaluate the equation.
        self._req_parms = {1: [True, True, True, True, True, False, False],
                           2: [True, True, True, True, False, False, False],
                           3: [True, True, True, True, True, True, False],
                           4: [True, True, True, True, True, True, False],
                           5: [True, True, True, True, True, False, False],
                           6: [True, True, True, True, True, False, False],
                           7: [True, True, True, True, True, False, False],
                           8: [True, True, True, True, False, False, False]}

        self.eq_id = eq_id
        self.source_t_min = None # Initialize to None before assigning any values.
        self.source_t_max = None # Initialize to None before assigning any values.
        self.source_t_min = source_t_min
        self.source_t_max = source_t_max
        self.source_rmse = source_rmse
        self.source_mae = source_mae
        self.source_mape = source_mape
        self.source_t_unit = source_t_unit
        self.source_unit = source_unit
        self.source = source
        self.notes = notes


    @property
    def a(self):
        """float or None : Correlation constant."""
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            self._a = value
        elif isinstance(value, float):
            self._a = value
        else:
            raise TypeError("a must be a float.")

    @property
    def b(self):
        """float or None : Correlation constant."""
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            self._b = value
        elif isinstance(value, float):
            self._b = value
        else:
            raise TypeError("b must be a float.")

    @property
    def c(self):
        """float or None : Correlation constant."""
        return self._c

    @c.setter
    def c(self, value):
        if value is None:
            self._c = value
        elif isinstance(value, float):
            self._c = value
        else:
            raise TypeError("c must be a float.")

    @property
    def d(self):
        """float or None : Correlation constant."""
        return self._d

    @d.setter
    def d(self, value):
        if value is None:
            self._d = value
        elif isinstance(value, float):
            self._d = value
        else:
            raise TypeError("d must be a float.")

    @property
    def e(self):
        """float or None : Correlation constant."""
        return self._e

    @e.setter
    def e(self, value):
        if value is None:
            self._e = value
        elif isinstance(value, float):
            self._e = value
        else:
            raise TypeError("e must be a float.")

    @property
    def f(self):
        """float or None : Correlation constant."""
        return self._f

    @f.setter
    def f(self, value):
        if value is None:
            self._f = value
        elif isinstance(value, float):
            self._f = value
        else:
            raise TypeError("f must be a float.")

    @property
    def g(self):
        """float or None : Correlation constant."""
        return self._g

    @g.setter
    def g(self, value):
        if value is None:
            self._g = value
        elif isinstance(value, float):
            self._g = value
        else:
            raise TypeError("g must be a float.")

    @property
    def eq_id(self):
        """int : Equation identification number."""
        return self._eq_id

    @eq_id.setter
    def eq_id(self, value):
        if value is None:
            self._eq_id = None
        elif isinstance(value, int):
            if value in self._req_parms:
                self._eq_id = value
            else:
                raise ValueError("eq_id must correspond to a pre-defined equation.")
        else:
            raise TypeError("eq_id must be an int.")

    @property
    def source_t_min(self):
        """float or None : Minimum temperature in source temperature unit."""
        return self._source_t_min

    @source_t_min.setter
    def source_t_min(self, value):
        if value is None:
            self._source_t_min = value
        elif isinstance(value, float):
            if self._source_t_max is None:
                self._source_t_min = value
            elif self._source_t_max > value:
                self._source_t_min = value
            else:
                raise ValueError("source_t_min must be less than t_max.")
        else:
            raise TypeError("source_t_min must be a float.")

    @property
    def t_min(self):
        """float or None : Minimum temperature, K."""
        if self._source_t_min is None:
            return None
        else:
            return TEMPERATURE[self._source_t_unit](self._source_t_min)

    @property
    def source_t_max(self):
        """float or None : Maximum temperature in source temperature unit."""
        return self._source_t_max

    @source_t_max.setter
    def source_t_max(self, value):
        if value is None:
            self._source_t_max = value
        elif isinstance(value, float):
            if self._source_t_min is None:
                self._source_t_max = value
            elif self._source_t_min < value:
                self._source_t_max = value
            else:
                raise ValueError("t_max must be greater than t_min.")
        else:
            raise TypeError("t_max must be a float.")

    @property
    def t_max(self):
        """float or None : Minimum temperature, K."""
        if self._source_t_max is None:
            return None
        else:
            return TEMPERATURE[self._source_t_unit](self._source_t_max)

    @property
    def source_rmse(self):
        """float : Root mean squared error for correlated property in source unit.

        Quadratic measure of average magnitude of error without considering direction.  Useful for uncertainty
        propagation analysis (notice the functional form is similar to the standard deviation).

        mae = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5
        """
        return self._source_rmse

    @source_rmse.setter
    def source_rmse(self, value):
        if value is None:
            self._source_rmse = value
        elif isinstance(value, float):
            self._source_rmse = value
        else:
            raise TypeError("source_rmse must be a float.")

    @property
    def rmse(self):
        """float : Root mean squared error for correlated property in SI unit.

        Quadratic measure of average magnitude of error without considering direction.  Useful for uncertainty
        propagation analysis (notice the functional form is similar to the standard deviation).

        mae = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5
        """
        if self._source_rmse is None or self._source_unit is None:
            return None
        else:
            return conv_to_si(self._source_rmse, self._source_unit)

    @property
    def source_mae(self):
        """float : Mean absolute error for correlated property in source unit.

        Measure of average magnitude of error without considering direction.

        mae = (1/n) * sum_i(abs(yi_meas - yi_model))
        """
        return self._source_mae

    @source_mae.setter
    def source_mae(self, value):
        if value is None:
            self._source_mae = value
        elif isinstance(value, float):
            self._source_mae = value
        else:
            raise TypeError("source_mae must be a float.")

    @property
    def mae(self):
        """float : Mean absolute error for correlated property in SI unit.

        Measure of average magnitude of error without considering direction.

        mae = (1/n) * sum_i(abs(yi_meas - yi_model))
        """
        if self._source_mae is None or self._source_unit is None:
            return None
        else:
            return conv_to_si(self._source_mae, self._source_unit)

    @property
    def source_mape(self):
        """float : Mean absolute percentage error for correlated property in source unit.

        Measure of relative average magnitude of error without considering direction.

        mae = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))
        """
        return self._source_mape

    @source_mape.setter
    def source_mape(self, value):
        if value is None:
            self._source_mape = value
        elif isinstance(value, float):
            self._source_mape = value
        else:
            raise TypeError("source_mae must be a float.")

    @property
    def mape(self):
        """float : Mean absolute percentage error for correlated property in source unit.

        Measure of relative average magnitude of error without considering direction.

        mae = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))
        """
        return self._source_mape

    @property
    def source_t_unit(self):
        """str : Source temperature unit for correlated property."""
        return self._source_t_unit

    @source_t_unit.setter
    def source_t_unit(self, value):
        if isinstance(value, str):
            if value in TEMPERATURE:
                self._source_t_unit = value
                return
            raise ValueError("source_t_unit is not defined.")
        else:
            raise TypeError("source_t_unit must be a string.")

    @property
    def source_unit(self):
        """str : Source unit for correlated property."""
        return self._source_unit

    @source_unit.setter
    def source_unit(self, value):
        if isinstance(value, str):
            for conv_dict in UNITS:
                if value in conv_dict:
                    self._source_unit = value
                    return
            raise ValueError("source_unit is not defined.")
        else:
            raise TypeError("source_unit must be a string.")

    @property
    def unit(self):
        """str : SI unit for correlated property."""
        for conv_dict in UNITS:
            if self._source_unit in conv_dict:
                return si_unit(conv_dict)
        raise ValueError("source_unit is not defined.")

    @property
    def source(self):
        """str : Source for the correlation (ACS citation format preferred)."""
        return self._source

    @source.setter
    def source(self, value):
        if value is None:
            self._source = value
        elif isinstance(value, str):
            self._source = value
        else:
            raise TypeError("source must be a string.")

    @property
    def notes(self):
        """str : Notes associated with the correlation."""
        return self._notes

    @notes.setter
    def notes(self, value):
        if value is None:
            self._notes = value
        elif isinstance(value, str):
            self._notes = value
        else:
            raise TypeError("notes must be a string.")

    def _t_conv(self, t, source_unit):
        """Convert temperature from Kelvin to source_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[source_unit](t)

    def _tr(self, t, tc):
        return t / tc

    def _tau(self, t, tc):
        return 1.0 - self._tr(t, tc)

    def _t_in_range(self, t):
        if self.t_min <= t <= self.t_max:
            return True
        elif self.t_min is None and t <= self.t_max:
            return True
        elif self.t_max is None and t >= self.t_min:
            return True
        elif self.t_min is None and self.t_max is None:
            return True
        else:
            return False

    def defined(self):
        """Check to see if instance is fully defined.

        Returns
        -------
        bool
            True if instance is fully defined.
        """
        if self._eq_id is not None:
            a_def = self._a is not None
            b_def = self._b is not None
            c_def = self._c is not None
            d_def = self._d is not None
            e_def = self._e is not None
            f_def = self._f is not None
            g_def = self._g is not None
            parms_def = [a_def, b_def, c_def, d_def, e_def, f_def, g_def]
            req_parms_def = parms_def == self._req_parms[self._eq_id]
            source_unit_def = self._source_unit is not None
            correlation_def = self._correlation[self._eq_id] is not None
            return req_parms_def and source_unit_def and correlation_def
        else:
            return False

    def derivative(self, t=None):
        """Derivative of the correlation.

        Parameters
        ----------
        t : float
            Temperature

        Returns
        -------
        float
            Derivative evaluated at 't'.
        """
        if not isinstance(t, float):
            raise TypeError("t must be a float.")
        elif not self.defined():
            raise RuntimeError("Corel instance not fully defined.")
        elif self._derivative[self._eq_id] is None:
            raise RuntimeError("Derivative is not defined.")
        elif not self._t_in_range(t=t):
            raise ValueError("t must be inside the range defined by t_min and t_max.")
        else:
            return conv_to_si(self._derivative[self._eq_id](self._t_conv(t, self._source_t_unit)), self._source_unit)

    def integral(self, t1=None, t2=None):
        """Integral of the correlation.

        Parameters
        ----------
        t1 : float
            Lower temperature
        t2 : float
            Upper temperature

        Returns
        -------
        float
            Integral evaluated over the interval 't1' to 't2'.
        """
        if not isinstance(t1, float):
            raise TypeError("t1 must be a float.")
        elif not isinstance(t2, float):
            raise TypeError("t2 must be a float.")
        elif not self.defined():
            raise RuntimeError("Corel instance not fully defined.")
        elif self._integral[self._eq_id] is None:
            raise RuntimeError("Integral is not defined.")
        elif not self._t_in_range(t=t):
            raise ValueError("t must be inside the range defined by t_min and t_max.")
        else:
            return conv_to_si(self._integral[self._eq_id](self._t_conv(t2, self._source_t_unit)) -
                              self._integral[self._eq_id](self._t_conv(t1, self._source_t_unit)), self._source_unit)

    def __call__(self, t=None):
        """Evaluate the correlation.

        Patameters
        ----------
        t : float
            Temperature, K.

        Returns
        -------
        float
            Correlation evaluated at 't'.
        """
        if not isinstance(t, float):
            raise TypeError("t must be a float.")
        elif not self.defined():
            raise RuntimeError("Corel instance not fully defined.")
        elif not self._t_in_range(t=t):
            raise ValueError("t must be inside the range defined by t_min and t_max.")
        else:
            return conv_to_si(self._correlation[self._eq_id](self._t_conv(t, self._source_t_unit)), self._source_unit)

    def __eq__(self, other):
        if isinstance(other, Corel):
            a_eq = self._a == other.a
            b_eq = self._b == other.b
            c_eq = self._c == other.c
            d_eq = self._d == other.d
            e_eq = self._e == other.e
            f_eq = self._f == other.f
            g_eq = self._g == other.g
            t_min_eq = self._source_t_min == other.source_t_min
            t_max_eq = self._source_t_max == other.source_t_max
            eq_id_eq = self._eq_id == other.eq_id
            unit_eq = self._source_unit == other.source_unit
            t_unit_eq = self._source_t_unit == other.sourc_t_unit
            return a_eq and b_eq and c_eq and d_eq and e_eq and f_eq and g_eq and \
                   t_min_eq and t_max_eq and eq_id_eq and unit_eq and t_unit_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self._a, self._b, self._c, self._d, self._e, self._f, self._g,
                     self._t_min, self._t_max, self._eq_id))


class Var(object):
    """Variable with metadata.

    Notes
    -----
    TODO: Implement a float with units, uncertainty, source, and other useful metadata (similar to Corel class).
    """


class Props(object):
    """Thermodynamic and transport properties of a system (phase or mixture)."""
    def __init__(self, system=None):
        """
        Parameters
        ----------
        system : Phase or Mixture object
        """
        if system is None:
            raise ValueError("Must pass an instance of Phase or Mixture to the constructor.")
        else:
            self.system = system
            self._a_r = None
            self._a_ig = None
            self._u_r = None
            self._u_ig = None
            self._h_r = None
            self._h_ig = None
            self._g_r = None
            self._g_ig = None
            self._s_r = None
            self._s_ig = None
            self._cv_r = None
            self._cv_ig = None
            self._cp_r = None
            self._cp_ig = None
            self._c = None
            self._z = None
            # TODO: Isothermal compressibility
            # TODO: Isentropic compressibility
            # TODO: Joule-Thompson coefficient
            self._phi = None
            self._ln_phi = None
            self._dln_phi_dt = None
            self._dln_phi_dp = None
            self._dln_phi_dni = None
            self._dp_dt = None
            self._dp_dv = None
            self._dp_dni = None
            self._dv_dni = None
            # TODO: Ideal gas viscosity
            # TODO: Ideal gas thermal conductivity
            # TODO: Viscosity
            # TODO: Thermal conductivity

    @property
    def system(self):
        """Phase or Mixture instance : The phase or mixture associated with a Prop object.

        Can only be set whe creating a new Comp instance."""
        return self._system

    @system.setter
    def system(self, value):
        try:
            self._system
        except AttributeError:
            if isinstance(value, (Phase, Mixture)):
                self._system = value
            else:
                raise TypeError("phase must be an instance of Phase or Mixture.")

    @property
    def mw(self):
        """float or None : Molecular weight, g/mol.
        
        TODO: Remove mw from Comps?  It is nice to have MW available in comps.  Remove from Props?
        """
        if self._system.comps.mw is None or self._system.compos.xi is None:
            return None
        else:
            return np.sum(self._system.comps.mw * self._system.compos.xi)

    @property
    def rho_m(self):
        """float or None : Mass density, kg/m3"""
        if self.mw is None or self._system.state.vm is None:
            return None
        else:
            return self.mw / (1000.0 * self._system.state.vm)

    @property
    def a(self):
        """float or None : Helmholtz energy, ??"""
        if isinstance(self._a_r, float) and isinstance(self._a_ig, float):
            return self._a_r + self._a_ig
        else:
            return None

    @property
    def a_r(self):
        """float or None : Residual Helmholtz energy, ??"""
        return self._a_r

    @property
    def a_ig(self):
        """float or None : Ideal gas Helmholtz energy, ??"""
        return self._a_ig

    @property
    def u(self):
        """float or None : Internal energy, ??"""
        if isinstance(self._u_r, float) and isinstance(self._u_ig, float):
            return self._u_r + self._u_ig
        else:
            return None

    @property
    def u_r(self):
        """float or None : Residual internal energy, ??"""
        return self._u_r

    @property
    def u_ig(self):
        """float or None : Ideal gas internal energy, ??"""
        return self._u_ig

    @property
    def h(self):
        """float or None : Enthalpy, ??"""
        if isinstance(self._h_r, float) and isinstance(self._h_ig, float):
            return self._h_r + self._h_ig
        else:
            return None

    @property
    def h_r(self):
        """float or None : Residual enthalpy, ??"""
        return self._h_r

    @property
    def h_ig(self):
        """float or None : Ideal gas enthalpy, ??"""
        return self._h_ig

    @property
    def g(self):
        """float or None : Gibbs energy, ??"""
        if isinstance(self._g_r, float) and isinstance(self._g_ig, float):
            return self._g_r + self._g_ig
        else:
            return None

    @property
    def g_r(self):
        """float or None : Residual gibbs energy, ??"""
        return self._g_r

    @property
    def g_ig(self):
        """float or None : Ideal gas gibbs energy, ??"""
        return self._g_ig

    @property
    def s(self):
        """float or None : Entropy, ??"""
        if isinstance(self._s_r, float) and isinstance(self._s_ig, float):
            return self._s_r + self._s_ig
        else:
            return None

    @property
    def s_r(self):
        """float or None : Residual entropy, ??"""
        return self._s_r

    @property
    def s_ig(self):
        """float or None : Ideal gas entropy, ??"""
        return self._s_ig

    @property
    def cv(self):
        """float or None : Isochoric heat capacity, ??"""
        if isinstance(self._cv_r, float) and isinstance(self._cv_ig, float):
            return self._cv_r + self._cv_ig
        else:
            return None

    @property
    def cv_r(self):
        """float or None : Residual isochoric heat capacity, ??"""
        return self._cv_r

    @property
    def cv_ig(self):
        """float or None : Ideal gas isochoric heat capacity, ??"""
        return self._cv_ig

    @property
    def cp(self):
        """float or None : Isobaric heat capacity, ??"""
        if isinstance(self._cp_r, float) and isinstance(self._cp_ig, float):
            return self._cp_r + self._cp_ig
        else:
            return None

    @property
    def cp_r(self):
        """float or None : Residual isobaric heat capacity, ??"""
        return self._cp_r

    @property
    def cp_ig(self):
        """float or None : Ideal gas isobaric heat capacity, ??"""
        return self._cp_ig

    @property
    def z(self):
        """float or None : Compressibility factor, dimensionless"""
        return self._z

    @property
    def c(self):
        """float or None : Speed of sound, m/s"""
        return self._c

    @property
    def phi(self):
        """np.ndarray or None : Fugacity coefficient for each component in corresponding CompSet."""
        return self._phi

    @property
    def ln_phi(self):
        """np.ndarray or None : Natural log of the fugacity coefficient for each component in corresponding CompSet."""
        return self._ln_phi

    @property
    def dln_phi_dt(self):
        return self._dln_phi_dt

    @property
    def dln_phi_dp(self):
        return self._dln_phi_dp

    @property
    def dln_phi_dni(self):
        return self._dln_phi_dni

    @property
    def dp_dt(self):
        """float or None : Derivative of pressure with respect to temperature, Pa/K."""
        return self._dp_dt

    @property
    def dp_dv(self):
        """float or None : Derivative of pressure with respect to volume, ??."""
        return self._dp_dv

    @property
    def dp_dni(self):
        """float or None : Derivative of pressure with respect to mole numbers, ??."""
        return self._dp_dni

    @property
    def dv_dni(self):
        """float or None : Derivative of volume with respect to mole numbers, ??."""
        return self._dv_dni

    def _update_ig(self):
        return

    def _update_r(self):
        return

    def _update_successive_substitution_flash(self):
        return

    def _update_gibbs_minimization_flash(self):
        return

    def _update_phase_envelope(self):
        return

    def _update_all(self):
        self._update_ig()
        self._update_r()
        self._update_successive_substitution_flash()
        self._update_gibbs_minimization_flash()
        self._update_phase_envelope()

    def _reset_all(self):
        self._a_r = None
        self._a_ig = None
        self._u_r = None
        self._u_ig = None
        self._h_r = None
        self._h_ig = None
        self._g_r = None
        self._g_ig = None
        self._s_r = None
        self._s_ig = None
        self._cv_r = None
        self._cv_ig = None
        self._cp_r = None
        self._cp_ig = None
        self._c = None
        self._z = None
        self._phi = None
        self._ln_phi = None
        self._dln_phi_dt = None
        self._dln_phi_dp = None
        self._dln_phi_dni = None
        self._dp_dt = None
        self._dp_dv = None
        self._dp_dni = None
        self._dv_dni = None


class Phase(object):
    """Pure phase with uniform properties.

    TODO: Consider generalizing. A Phase object doesn't have to have an EOS and can be used to store experimental data.
    """
    def __init__(self, comps=None, eos=None):
        """
        Parameters
        ----------
        comps : CompSet object
        eos : sPCSAFT or CPA object
        """
        if comps is None:
            raise ValueError("Must pass an instance of CompSet to the constructor.")
        else:
            self.comps = comps
            self.eos = eos
            self.state = State(system=self)
            self.compos = Compos(system=self)
            self.extent = Extent(system=self)
            self.props = Props(system=self)

    @property
    def comps(self):
        """CompSet object : Set of components or pseudo-components associated with Phase."""
        return self._comps

    @comps.setter
    def comps(self, value):
        try:
            self._comps
        except AttributeError:
            if isinstance(value, CompSet):
                self._comps = value
            else:
                raise TypeError("comps must be an instance of CompSet.")

    @property
    def eos(self):
        """sPCSAFT or CPA object or None : Equation of state associated with Phase."""
        return self._eos

    @eos.setter
    def eos(self, value):
        if value is None:
            self._eos = None
        elif isinstance(value, (sPCSAFT, CPA)):
            self._eos = value
        else:
            raise TypeError("eos must be an instance of sPC-SAFT or CPA.")

    @property
    def state(self):
        """State object : State of the Phase."""
        return self._state

    @state.setter
    def state(self, value):
        try:
            self._state
        except AttributeError:
            if isinstance(value, State):
                self._state = value
                self._state.callback.add(self._state_callback)
            else:
                raise TypeError("state must be an instance of State.")
    @property
    def compos(self):
        """Compos object : Intensive composition of the Phase."""
        return self._compos

    @compos.setter
    def compos(self, value):
        try:
            self._compos
        except AttributeError:
            if isinstance(value, Compos):
                self._compos = value
                self._compos.callback.add(self._compos_callback)
            else:
                raise TypeError("compos must be an instance of Compos.")

    @property
    def extent(self):
        """Extent object : Extensive composition of the Phase."""
        return self._extent

    @extent.setter
    def extent(self, value):
        try:
            self._extent
        except AttributeError:
            if isinstance(value, Extent):
                self._extent = value
            else:
                raise TypeError("extent must be an instance of Extent.")

    @property
    def props(self):
        """Props object : Thermodynamic and transport properties associated with Phase."""
        return self._props

    @props.setter
    def props(self, value):
        try:
            self._props
        except AttributeError:
            if isinstance(value, Props):
                self._props = value
            else:
                raise TypeError("props must be an instance of Props")

    def _compos_callback(self):
        # Execute if Compos was updated.
        # print("Compos was called back.")
        return

    def _state_callback(self, spec=None):
        # Execute if State was updated.
        # print("State was called back. Spec = {}".format(spec))
        return

    def _vt_pres_update(self, spec='full'):
        if spec == 'full' and self.defined():
            # Update volume using both liquid and vapor guesses.  Choose volume with minimum gibbs energy.
            return
        else:
            raise ValueError("spec must be either full or prior.")

    def defined(self):
        """Check to see if the instance is fully defined (all values available for EOS evaluation)

        Returns
        -------
        bool
            True if instance is fully defined.
        """
        comps_def = self.comps is not None
        compos_def = self.compos is not None and self.compos.defined()
        state_def = self.state is not None and self.state.defined()
        eos_def = self.eos is not None
        return comps_def and compos_def and state_def and eos_def

    def __eq__(self, other):
        if isinstance(other, Phase):
            comps_eq = self.comps == other.comps
            compos_eq = self.compos == other.compos
            state_eq = self.state == other.state
            eos_eq = True
            return comps_eq and compos_eq and state_eq and eos_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(())


class Mixture(object):
    """Mixture of phases."""
    def __init__(self, comps=None, state=None, eos=None):
        """
        Parameters
        ----------
        comps : CompSet object
        state : State object or None
        eos : sPCSAFT or CPA object or None
        """
        if comps is None:
            raise ValueError("Must pass an instance of CompSet to the constructor.")
        else:
            self.comps = comps
            self.state = state
            self.eos = eos
            self.compos = Compos(system=self)
            self.extent = Extent(system=self)
            self.props = Props(system=self)
            self._phases = []
            self._trial_phases = []

    @property
    def comps(self):
        """CompSet object : Set of components or pseudo-components associated with Mixture."""
        return self._comps

    @comps.setter
    def comps(self, value):
        try:
            self._comps
        except AttributeError:
            if isinstance(value, CompSet):
                self._comps = value
            else:
                raise TypeError("comps must be an instance of CompSet.")

    @property
    def state(self):
        """State object : State of the Mixture."""
        return self._state

    @state.setter
    def state(self, value):
        try:
            self._state
        except AttributeError:
            if value is None:
                self._state = State(system=self)
            elif isinstance(value, State):
                self._state = value
            else:
                raise TypeError("state must be an instance of State.")

    @property
    def eos(self):
        """sPCSAFT or CPA object or None : Equation of state associated with Mixture."""
        return self._eos

    @eos.setter
    def eos(self, value):
        if value is None:
            self._eos = None
        elif isinstance(value, (sPCSAFT, CPA)):
            self._eos = value
        else:
            raise TypeError("eos must be an instance of sPC-SAFT or CPA.")

    @property
    def compos(self):
        """Compos object : Intensive composition of the Mixture."""
        return self._compos

    @compos.setter
    def compos(self, value):
        try:
            self._compos
        except AttributeError:
            if isinstance(value, Compos):
                self._compos = value
            else:
                raise TypeError("compos must be an instance of Compos.")

    @property
    def extent(self):
        """Extent object : Extensive composition of the Mixture."""
        return self._extent

    @extent.setter
    def extent(self, value):
        try:
            self._extent
        except AttributeError:
            if isinstance(value, Extent):
                self._extent = value
            else:
                raise TypeError("extent must be an instance of Extent.")

    @property
    def props(self):
        """Props object : Thermodynamic and transport properties associated with the Mixture."""
        return self._props

    @props.setter
    def props(self, value):
        try:
            self._props
        except AttributeError:
            if isinstance(value, Props):
                self._props = value
            else:
                raise TypeError("props must be an instance of Props")

    @property
    def phases(self):
        """list of Phase objects : Phases in the mixture."""
        return self._phases

    @phases.setter
    def phases(self, value):
        try:
            self._phases
        except AttributeError:
            if isinstance(value, list):
                self._phases = value
            else:
                raise TypeError("phases must be a list.")

    def _add_phase(self, phase=None):
        if phase is None:
            p = Phase(comps=self._comps, state=self._state, eos=self._eos)
            self._phases.append(p)
        elif isinstance(phase, Phase):
            self._phases.append(phase)
        else:
            raise TypeError("phase must be an instance of Phase.")

    def _remove_phase(self, phase=None):
        if isinstance(phase, Phase):
            for i, p in enumerate(self._phases):
                if p == phase:
                    del self._phases[i]

    def _e(self, beta=None, phi=None):
        """E-vector for Michelsen's alternative flash.

        For a mixture containing 'i' components and 'k' phases, the E-vector is defined as follows:

            e[i] = sum_over_phases(beta[k] / phi[i, k])

        Parameters
        ----------
        beta : np.ndarray
            Phase fraction vector with shape (k, 1).
        phi : np.ndarray
            Fugacity coefficient matrix with shape (i, k)

        Returns
        -------
        np.ndarray
            E-vector with shape (i, 1).
        """

    def _q(self, z=None, beta=None, e=None):
        """Objective function for Michelsen's alternative flash.

        For a mixture containing 'i' components and 'j' phases, the Q-function is defined as follows:

            Q = sum_over_phases(beta[j]) - sum_over_components(z[i] * ln(e[i]))

        Parameters
        ----------
        z : np.ndarray
            Mixture composition vector with shape (i, 1).
        beta : np.ndarray
            Phase fraction vector with shape (j, 1).
        e : np.ndarray
            E-vector with shape (i, 1)

        Returns
        -------
        float
            Michelsen's Q-function.
        """
        return

    def _g(self, z=None, phi=None, e=None):
        """Gradient for Michelsen's alternative flash.

        For a mixture containing 'i' components and 'j' phases, the gradient is defined as follows:

            g[j] = 1.0 - sum_over_components(z[i] / (e[i] * phi[i, j]))

        Parameters
        ----------
        z : np.ndarray
            Mixture composition vector with shape (i, 1).
        phi : np.ndarray
            Fugacity coefficient matrix with shape (i, j).
        e : np.ndarray
            E-vector with shape (i, 1)

        Returns
        -------
        np.ndarray
            Gradient vector with shape (j, 1).
        """
        return

    def _h(self):
        """Hessian for Michelsen's alternative flash.

        For a mixture containing 'i' components and 'j' phases, the gradient is defined as follows:

            h[j, k] =

        Parameters
        ----------
        z : np.ndarray
            Mixture composition vector with shape (i, 1).
        phi : np.ndarray
            Fugacity coefficient matrix with shape (i, j).
        e : np.ndarray
            E-vector with shape (i, 1)

        Returns
        -------
        np.ndarray
            Gradient vector with shape (j, 1).
        """
        return

    def _pt_flash(self):
        """Driver for PT flash routine."""
        return

    def _vt_flash(self):
        """Driver for VT flash routine (to be developed later)."""
        return


class AssocSite(object):
    """Association site."""
    def __init__(self, comp=None, site=None, type=None, desc=None):
        if comp is None or site is None or type is None:
            raise ValueError("comp, site, and type must be provided to create an instance of AssocSite.")
        else:
            self.comp = comp
            self.site = site
            self.type = type
            self.desc = desc

    @property
    def comp(self):
        return self._comp

    @comp.setter
    def comp(self, value):
        try:
            self._comp
        except AttributeError:
            if isinstance(value, Comp):
                self._comp = value
            else:
                raise TypeError("Component must be an instance of Comp.")

    @property
    def site(self):
        return self._site

    @site.setter
    def site(self, value):
        try:
            self._site
        except AttributeError:
            if isinstance(value, str):
                self._site = value
            else:
                raise TypeError("Site must be a string.")

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        try:
            self._type
        except AttributeError:
            if value in ['ed', 'ea', 'glue', 'pi_stack']:
                self._type = value
            else:
                raise ValueError("Site type not valid.")

    def __eq__(self, other):
        if isinstance(other, AssocSite):
            comp_eq = self.comp == other.comp
            site_eq = self.site == other.site
            type_eq = self.type == other.type
            return comp_eq and site_eq and type_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.comp.name, self.site, self.type))

    def __str__(self):
        return "{}, {}, {}".format(self.comp.name, self.site, self.type)

    def can_interact(self, other):
        if isinstance(other, AssocSite):
            if self.type == 'ea' and other.type == 'ed':
                return True
            elif self.type == 'ed' and other.type == 'ea':
                return True
            elif self.type == 'glue' and other.type == 'glue':
                return True
            elif self.type == 'pi_stack' and other.type == 'pi_stack':
                return True
            else:
                return False
        return False


class AssocSiteInter(object):
    """Interaction between two association sites."""
    def __init__(self, site_a=None, site_b=None, eos=None, source=None,
                 assoc_energy=None, assoc_vol=None):
        if site_a is None or site_b is None or eos is None:
            raise ValueError("site_a, site_b, and eos must be provided to create an instance of AssocSite.")
        elif not isinstance(site_a, AssocSite) or not isinstance(site_b, AssocSite):
            raise ValueError("Association sites must be an instance of AssocSite.")
        elif not site_a.can_interact(site_b):
            raise ValueError("Association sites must be able to interact.")
        else:
            self.site_a = site_a
            self.site_b = site_b
            self.eos = eos
            self.source = source
            self.assoc_energy = assoc_energy
            self.assoc_vol = assoc_vol

    @property
    def site_a(self):
        return self._site_a

    @site_a.setter
    def site_a(self, value):
        try:
            self._site_a
        except AttributeError:
            self._site_a = value

    @property
    def site_b(self):
        return self._site_b

    @site_b.setter
    def site_b(self, value):
        try:
            self._site_b
        except AttributeError:
            self._site_b = value

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, value):
        try:
            self._eos
        except AttributeError:
            # TODO: Convert to checking for instance of CPA or sPC-SAFT.
            if value in ASSOC_EOS:
                self._eos = value
            else:
                raise ValueError("eos is not valid.")

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
    def assoc_energy(self):
        return self._assoc_energy

    @assoc_energy.setter
    def assoc_energy(self, value):
        if value is None:
            self._assoc_energy = value
        elif isinstance(value, float) and value >= 0.0:
            self._assoc_energy = value
        else:
            raise ValueError("assoc_energy must be a positive float.")

    @property
    def assoc_vol(self):
        return self._assoc_vol

    @assoc_vol.setter
    def assoc_vol(self, value):
        if value is None:
            self._assoc_vol = value
        elif isinstance(value, float) and value >= 0.0:
            self._assoc_vol = value
        else:
            raise ValueError("assoc_vol must be a positive float.")

    def __eq__(self, other):
        if isinstance(other, AssocSiteInter):
            aa_bb_eq = self.site_a == other.site_a and self.site_b == other.site_b
            ab_ba_eq = self.site_a == other.site_b and self.site_b == other.site_a
            eos_eq = self.eos == other.eos
            return (aa_bb_eq or ab_ba_eq) and eos_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.site_a), hash(self.site_b), self.eos))

    def __str__(self):
        return "Site A: ({}), Site B: ({}), EOS: {}, Source: {}, Volume: {}, Energy: {}".format(self.site_a,
                                                                                                self.site_b,
                                                                                                self.eos,
                                                                                                self.source,
                                                                                                self.assoc_vol,
                                                                                                self.assoc_energy)


class AssocInter(object):
    """Association interactions between multiple components."""
    def __init__(self, comps=None, eos=None, adj_parm_spec='none'):
        if comps is None or eos is None:
            raise ValueError("comps and eos must be provided to create an instance of AssocSite.")
        self.comps = comps
        self.eos = eos
        self.assoc_sites = self._create_assoc_sites()
        self.assoc_site_inter = self._create_assoc_site_inters()
        self._adj_assoc_energy_members = []
        self._adj_assoc_vol_members = []
        self.set_adj_parms(adj_parm_spec=adj_parm_spec)

    @property
    def comps(self):
        return self._comps

    @comps.setter
    def comps(self, value):
        try:
            self._comps
        except AttributeError:
            if isinstance(value, CompSet):
                self._comps = value
            else:
                raise TypeError("comps must be an instance of CompSet.")

    @property
    def eos(self):
        # TODO: consider dropping the EOS spec...what is it used for anyway?
        return self._eos

    @eos.setter
    def eos(self, value):
        try:
            self._eos
        except AttributeError:
            if value in ASSOC_EOS:
                self._eos = value
            else:
                raise ValueError("eos is not valid.")

    @property
    def assoc_sites(self):
        return self._assoc_sites

    @assoc_sites.setter
    def assoc_sites(self, value):
        try:
            self._assoc_sites
        except AttributeError:
            self._assoc_sites = value

    @property
    def assoc_site_inter(self):
        return self._assoc_site_inter

    @assoc_site_inter.setter
    def assoc_site_inter(self, value):
        try:
            self._assoc_site_inter
        except AttributeError:
            self._assoc_site_inter = value

    @property
    def adj_assoc_energy_members(self):
        return self._adj_assoc_energy_members

    @property
    def adj_assoc_energy(self):
        ae = []
        for asi in self._adj_assoc_energy_members:
            ae.append(asi.assoc_energy)
        return ae

    @adj_assoc_energy.setter
    def adj_assoc_energy(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_assoc_energy_members):
            for i, asi in enumerate(self._adj_assoc_energy_members):
                if isinstance(value[i], float):
                    asi.assoc_energy = value[i]
                else:
                    raise ValueError("adj_assoc_energy values must be floats.")
        else:
            raise TypeError("adj_assoc_energy must be a list or tuple with {} elements."
                            .format(len(self._adj_assoc_energy_members)))

    @property
    def adj_assoc_vol_members(self):
        return self._adj_assoc_vol_members

    @property
    def adj_assoc_vol(self):
        av = []
        for asi in self._adj_assoc_vol_members:
            av.append(asi.assoc_vol)
        return av

    @adj_assoc_vol.setter
    def adj_assoc_vol(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_assoc_vol_members):
            for i, asi in enumerate(self._adj_assoc_vol_members):
                if isinstance(value[i], float):
                    asi.assoc_vol = value[i]
                else:
                    raise ValueError("adj_assoc_vol values must be floats.")
        else:
            raise TypeError("adj_assoc_vol must be a list or tuple with {} elements."
                            .format(len(self._adj_assoc_vol_members)))

    def set_adj_parms(self, adj_parm_spec, incl_pure_comp=False):
        # Define which parameters are adjustable for each AssocSiteInter in assoc_site_inter.  'incl_pure_comp' changes
        # whether or not pure component assoc_vol or assoc_energy are adjustable in multicomponent mixtures (note that
        # pure component assoc_vol or assoc_energy are always adjustable in a single component mixture). Reset all
        # 'member' lists to empty lists as a default every time this method is called.
        self._adj_assoc_energy_members = []
        self._adj_assoc_vol_members = []
        if adj_parm_spec == 'all':
            self._adj_assoc_energy_members = self._assoc_site_inter[:]
            self._adj_assoc_vol_members = self._assoc_site_inter[:]
        elif adj_parm_spec == 'none':
            pass
        elif adj_parm_spec == 'assoc_energy':
            if incl_pure_comp is False and len(self._comps.comps) > 1:
                for asi in self._assoc_site_inter:
                    if asi.site_a.comp != asi.site_b.comp:
                        self._adj_assoc_energy_members.append(asi)
            else:
                self._adj_assoc_vol_members = self._assoc_site_inter[:]
        elif adj_parm_spec == 'assoc_vol':
            if incl_pure_comp is False and len(self._comps.comps) > 1:
                for asi in self._assoc_site_inter:
                    if asi.site_a.comp != asi.site_b.comp:
                        self._adj_assoc_vol_members.append(asi)
            else:
                self._adj_assoc_vol_members = self._assoc_site_inter[:]
        else:
            raise ValueError("adj_parm_spec not valid.")

    # TODO: This already implicitly does not allow a Comp to have two association sites.  However, it would be good to
    #  build this check directly into the specification for Comp and give the user a warning if there is a problem.
    def _create_assoc_sites(self):
        # Create list of all sites.
        result = []
        for comp in self._comps.comps:
            if comp.assoc_sites is not None:
                for site in comp.assoc_sites:
                    if site not in result:
                        result.append(site)
        return result

    def _create_assoc_site_inters(self):
        # Create list of all possible site-to-site interactions as AssocSiteInter objects.
        result = []
        for comp_i in self._comps.comps:
            for comp_j in self._comps.comps:
                if comp_i.assoc_sites is not None and comp_j.assoc_sites is not None:
                    for site_a in comp_i.assoc_sites:
                        for site_b in comp_j.assoc_sites:
                            if site_a.can_interact(site_b):
                                new = AssocSiteInter(site_a, site_b, self._eos)
                                if new not in result:
                                    result.append(new)
        return result

    def load_pure_comp(self):
        # Load source, assoc_vol, and assoc_energy from Comp objects.
        for comp in self._comps.comps:
            if comp.assoc_sites is not None:
                if self._eos == 'CPA':
                    self.load_site_inter(comp.cpa_assoc)
                elif self._eos == 'sPC-SAFT':
                    self.load_site_inter(comp.spc_saft_assoc)
                elif self._eos == 'PC-SAFT':
                    self.load_site_inter(comp.pc_saft_assoc)
                else:
                    raise ValueError("eos not valid.")

    def load_site_inter(self, input):
        # Load source, assoc_vol, and assoc_energy from list of AssocSiteInter ('asi') objects.
        if isinstance(input, AssocSiteInter):
            for _asi in self._assoc_site_inter:
                if input == _asi:
                    _asi.source = input.source
                    _asi.assoc_vol = input.assoc_vol
                    _asi.assoc_energy = input.assoc_energy
        elif isinstance(input, AssocInter):
            for asi in input.assoc_site_inter:
                for _asi in self._assoc_site_inter:
                    if asi == _asi:
                        _asi.source = asi.source
                        _asi.assoc_vol = asi.assoc_vol
                        _asi.assoc_energy = asi.assoc_energy
        elif isinstance(input, list):
            for item in input:
                if isinstance(item, AssocSiteInter):
                    for _asi in self._assoc_site_inter:
                        if _asi == item:
                            _asi.source = item.source
                            _asi.assoc_vol = item.assoc_vol
                            _asi.assoc_energy = item.assoc_energy
                else:
                    raise TypeError("Must pass a list of AssocSiteInter objects.")
        else:
            raise TypeError("Input not AssocSiteInter object, AssocInter object, or list of AssocSiteInter objects.")

    def assoc_energy(self):
        # Build array of association energy for all site combinations.
        ae = np.zeros((len(self.assoc_sites), len(self.assoc_sites)))
        for site in self._assoc_site_inter:
            i = self._assoc_sites.index(site.site_a)
            j = self._assoc_sites.index(site.site_b)
            ae[i, j] = site.assoc_energy
            ae[j, i] = site.assoc_energy
        return ae

    def assoc_vol(self):
        # Build array of association volumes for all site combinations.
        av = np.zeros((len(self.assoc_sites), len(self.assoc_sites)))
        for site in self._assoc_site_inter:
            i = self._assoc_sites.index(site.site_a)
            j = self._assoc_sites.index(site.site_b)
            av[i, j] = site.assoc_vol
            av[j, i] = site.assoc_vol
        return av

    def delta(self, g=None, b=None, d=None, t=None):
        # Build array of association strengths
        # g is the radial distribution function
        # Accociation energy is epsilon/KB.
        # np.multiply(self._assoc_vol(), self._assoc_energy())
        # t_dep_term = np.exp(self._assoc_energy()/(KB * t)) - 1.0
        if self._eos == 'CPA':
            if isinstance(g, float) and isinstance(b, np.ndarray) and isinstance(t, float):
                return np.exp(self.assoc_energy()/t) - 1.0
            else:
                raise TypeError('g and t must be floats and b_ij must be an np.ndarray')
        if self._eos in ['sPC-SAFT', 'PC-SAFT']:
            if isinstance(g, float) and isinstance(d, np.ndarray) and isinstance(t, float):
                return np.exp(self.assoc_energy()/t) - 1.0
            else:
                raise TypeError('g and t must be floats and d_ij must be an np.ndarray')

    def __eq__(self, other):
        if isinstance(other, AssocInter):
            comps_eq = self.comps == other.comps
            assoc_energy_eq = np.array_equal(self.assoc_energy(), other.assoc_energy())
            assoc_vol_eq = np.array_equal(self.assoc_vol(), other.assoc_vol())
            return comps_eq and assoc_energy_eq and assoc_vol_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.comps, self.assoc_energy(), self.assoc_vol()))


class BinaryInterParm(object):
    """Binary interaction parameter between components or pseudo-components.
        k_ij(T) = k_ij + a*T + b/T + c*ln(T)
    """
    def __init__(self, comp_a=None, comp_b=None, phys_eos_spec=None, eos=None, source=None,
                 temp_indep_coef=None, lin_temp_coef=None, inv_temp_coef=None, ln_temp_coef=None):
        if comp_a is None or comp_b is None or eos is None:
            raise ValueError("comp_a, comp_b, phys_eos_spec, and eos must be provided to create an instance of "
                             "BinaryInterParm.")
        else:
            self.comp_a = comp_a
            self.comp_b = comp_b
            self.phys_eos_spec = phys_eos_spec
            self.eos = eos
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
            if isinstance(value, (Comp, PseudoComp)):
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
            if isinstance(value, (Comp, PseudoComp)):
                self._comp_b = value
            else:
                raise TypeError("comp_b must be an instance of Comp or PseudoComp.")

    @property
    def phys_eos_spec(self):
        return self._phys_eos_spec

    @phys_eos_spec.setter
    def phys_eos_spec(self, value):
        try:
            self._phys_eos_spec
        except AttributeError:
            if isinstance(value, (PCSAFTSpec, CubicSpec)):
                self._phys_eos_spec = value
            else:
                raise TypeError("phys_eos_spec must be an instance of PCSAFTSpec or CubicSpec.")

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, value):
        try:
            self._eos
        except AttributeError:
            if value in PHYS_EOS:
                self._eos = value
            else:
                raise ValueError("eos not valid.")

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
            pes_eq = self.phys_eos_spec == other.phys_eos_spec
            eos_eq = self.eos == other.eos
            return (aa_bb_eq or ab_ba_eq) and pes_eq and eos_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.comp_a), hash(self.comp_b), self.eos))

    def __str__(self):
        # TODO: Add phys_eos_spec somehow.
        return "Comp A: ({}), Comp B: ({}), EOS: {}, k_ij: {}".format(self.comp_a,
                                                                      self.comp_b,
                                                                      self.eos,
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


class CubicParms(object):
    """Cubic equation of state parameters"""


class PCSAFTParms(object):
    def __init__(self, comp=None, pc_saft_spec=None, source=None,
                 seg_num=None, seg_diam=None, disp_energy=None, ck_const=0.12):
        if comp is None or pc_saft_spec is None:
            raise ValueError("comp and pc_saft_spec must be provided to create an instance of PCSAFTParms.")
        else:
            self.comp = comp
            self.pc_saft_spec = pc_saft_spec
            self.source = source
            self.seg_num = seg_num
            self.seg_diam = seg_diam
            self.disp_energy = disp_energy
            self.ck_const = ck_const

    @property
    def comp(self):
        return self._comp

    @comp.setter
    def comp(self, value):
        try:
            self._comp
        except AttributeError:
            if isinstance(value, (Comp, PseudoComp)):
                self._comp = value
            else:
                raise TypeError("comp must be an instance of Comp or PseudoComp.")

    @property
    def pc_saft_spec(self):
        return self._pc_saft_spec

    @pc_saft_spec.setter
    def pc_saft_spec(self, value):
        try:
            self._pc_saft_spec
        except AttributeError:
            if isinstance(value, PCSAFTSpec):
                self._pc_saft_spec = value
            else:
                raise TypeError("pc_saft_spec must be an instance of PCSAFTSpec.")

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
    def seg_num(self):
        return self._seg_num

    @seg_num.setter
    def seg_num(self, value):
        if value is None:
            self._seg_num = value
        elif isinstance(value, float):
            self._seg_num = value
        else:
            raise ValueError("seg_num must be a float.")

    @property
    def seg_diam(self):
        return self._seg_diam

    @seg_diam.setter
    def seg_diam(self, value):
        if value is None:
            self._seg_diam = value
        elif isinstance(value, float):
            self._seg_diam = value
        else:
            raise ValueError("seg_diam must be a float.")

    @property
    def disp_energy(self):
        return self._disp_energy

    @disp_energy.setter
    def disp_energy(self, value):
        if value is None:
            self._disp_energy = value
        elif isinstance(value, float):
            self._disp_energy = value
        else:
            raise ValueError("disp_energy must be a float.")

    @property
    def ck_const(self):
        # Chen and Kreglewski temperature-dependent integral parameter is 0.12 for nearly all compounds. However, it can
        # be set to 0.241 for hydrogen (see eq 2-6 in de Villers' PhD thesis and eq 2-10 in Tihic's PhD thesis). This
        # is a useful correction for the quantum gases (hydrogen and helium).
        return self._ck_const

    @ck_const.setter
    def ck_const(self, value):
        if value is None:
            self._ck_const = value
        elif isinstance(value, float):
            self._ck_const = value
        else:
            raise ValueError("ck_const must be a float.")

    def __eq__(self, other):
        if isinstance(other, PCSAFTParms):
            comp_eq = self.comp == other.comp
            spec_eq = self.pc_saft_spec == other.pc_saft_spec
            return comp_eq and spec_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.comp), hash(self.pc_saft_spec)))

    def defined(self):
        sn = self.seg_num is not None
        sd = self.seg_diam is not None
        de = self.disp_energy is not None
        ck = self.ck_const is not None
        return sn and sd and de and ck

    def init_parms(self, comp_family='alkane'):
        # Initialize pure component parameters as a starting point for optimization routines.
        #
        # PC-SAFT parameter correlations taken from the following sources:
        #   (1) Tihic, A.; Kontogeorgis, G.M.; von Solms, N.; Michelsen, M.L. Applications of the simplified perturbed-
        #   chain SAFT equation of state using an extended parameter table. Fluid Phase Equilib. 2006, 248, 29-43.
        #
        # PC-SAFT parameters for n-hexane taken from the following source:
        #   (2) Gross, J.; Sadowski, G. Perturbed-Chain SAFT: An Equation of State Based on a Perturbation Theory for
        #   Chain Molecules.  Ind. Eng. Chem. Res. 2001, 40, 1244–1260.
        #
        # TODO: Define parameter estimation correlations based on component family.
        if self._comp.mw is not None:
            self._seg_num = 0.0249 * self._comp.mw + 0.9711
            self._seg_diam = ((1.6947 * self._comp.mw + 23.27) / self._seg_num) ** 0.33333
            self._disp_energy = (6.5446 * self._comp.mw + 177.92) / self._seg_num
        else:
            self._seg_num = 3.0576
            self._seg_diam = 3.7983
            self._disp_energy = 236.77


class PCSAFTPhysInter(object):
    """Physical interactions between components."""
    def __init__(self, comps=None, eos=None, pc_saft_spec=None, adj_pure_comp_spec=None, adj_binary_spec=None):
        if comps is None or eos is None or pc_saft_spec is None:
            raise ValueError("comps, eos, and pc_saft_spec must be provided to create an instance of PCSAFTPhysInter.")
        self.comps = comps
        self.eos = eos
        self.pc_saft_spec = pc_saft_spec
        self.pure_comp_parms = self._create_pure_comp_parms()
        self.binary_parms = self._create_binary_parms()
        self._adj_seg_num_members = []
        self._adj_seg_diam_members = []
        self._adj_disp_energy_members = []
        self._adj_temp_indep_coef_members = []
        self._adj_lin_temp_coef_members = []
        self._adj_inv_temp_coef_members = []
        self._adj_ln_temp_coef_members = []
        self.set_adj_pure_comp_parms(adj_pure_comp_spec=adj_pure_comp_spec)
        self.set_adj_binary_parms(adj_binary_spec=adj_binary_spec)

    @property
    def comps(self):
        return self._comps

    @comps.setter
    def comps(self, value):
        try:
            self._comps
        except AttributeError:
            if not isinstance(value, CompSet):
                raise TypeError("Component must be an instance of CompSet.")
            self._comps = value

    @property
    def eos(self):
        # TODO: consider droping the EOS spec...what is it used for anyway?
        return self._eos

    @eos.setter
    def eos(self, value):
        try:
            self._eos
        except AttributeError:
            if value in PC_SAFT_EOS:
                self._eos = value
            else:
                raise ValueError("eos not valid.")

    @property
    def pc_saft_spec(self):
        return self._pc_saft_spec

    @pc_saft_spec.setter
    def pc_saft_spec(self, value):
        try:
            self._pc_saft_spec
        except AttributeError:
            if isinstance(value, PCSAFTSpec):
                self._pc_saft_spec = value
            else:
                raise TypeError("pc_saft_spec not an instance of PCSAFTSpec.")

    @property
    def pure_comp_parms(self):
        return self._pure_comp_parms

    @pure_comp_parms.setter
    def pure_comp_parms(self, value):
        try:
            self._pure_comp_parms
        except AttributeError:
            self._pure_comp_parms = value

    @property
    def binary_parms(self):
        return self._binary_parms

    @binary_parms.setter
    def binary_parms(self, value):
        try:
            self._binary_parms
        except AttributeError:
            self._binary_parms = value

    @property
    def adj_seg_num_members(self):
        return self._adj_seg_num_members

    @property
    def adj_seg_num(self):
        values = []
        for member in self._adj_seg_num_members:
            values.append(member.seg_num)
        return values

    @adj_seg_num.setter
    def adj_seg_num(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_seg_num_members):
            for i, member in enumerate(self._adj_seg_num_members):
                if isinstance(value[i], float):
                    member.seg_num = value[i]
                else:
                    raise ValueError("adj_seg_num values must be floats.")
        else:
            raise TypeError("adj_seg_num must be a list or tuple with {} elements."
                            .format(len(self._adj_seg_num_members)))

    @property
    def adj_seg_diam_members(self):
        return self._adj_seg_diam_members

    @property
    def adj_seg_diam(self):
        values = []
        for member in self._adj_seg_diam_members:
            values.append(member.seg_diam)
        return values

    @adj_seg_diam.setter
    def adj_seg_diam(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_seg_diam_members):
            for i, member in enumerate(self._adj_seg_diam_members):
                if isinstance(value[i], float):
                    member.seg_diam = value[i]
                else:
                    raise ValueError("adj_seg_diam values must be floats.")
        else:
            raise TypeError("adj_seg_diam must be a list or tuple with {} elements."
                            .format(len(self._adj_seg_diam_members)))

    @property
    def adj_disp_energy_members(self):
        return self._adj_disp_energy_members

    @property
    def adj_disp_energy(self):
        values = []
        for member in self._adj_disp_energy_members:
            values.append(member.disp_energy)
        return values

    @adj_disp_energy.setter
    def adj_disp_energy(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_disp_energy_members):
            for i, member in enumerate(self._adj_disp_energy_members):
                if isinstance(value[i], float):
                    member.disp_energy = value[i]
                else:
                    raise ValueError("adj_disp_energy values must be floats.")
        else:
            raise TypeError("adj_disp_energy must be a list or tuple with {} elements."
                            .format(len(self._adj_disp_energy_members)))

    @property
    def adj_temp_indep_coef_members(self):
        return self._adj_temp_indep_coef_members

    @property
    def adj_temp_indep_coef(self):
        values = []
        for member in self._adj_temp_indep_coef_members:
            values.append(member.temp_indep_coef)
        return values

    @adj_temp_indep_coef.setter
    def adj_temp_indep_coef(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_temp_indep_coef_members):
            for i, member in enumerate(self._adj_temp_indep_coef_members):
                if isinstance(value[i], float):
                    member.temp_indep_coef = value[i]
                else:
                    raise ValueError("adj_temp_indep_coef values must be floats.")
        else:
            raise TypeError("adj_temp_indep_coef must be a list or tuple with {} elements."
                            .format(len(self._adj_temp_indep_coef_members)))

    @property
    def adj_lin_temp_coef_members(self):
        return self._adj_lin_temp_coef_members

    @property
    def adj_lin_temp_coef(self):
        values = []
        for member in self._adj_lin_temp_coef_members:
            values.append(member.lin_temp_coef)
        return values

    @adj_lin_temp_coef.setter
    def adj_lin_temp_coef(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_lin_temp_coef_members):
            for i, member in enumerate(self._adj_lin_temp_coef_members):
                if isinstance(value[i], float):
                    member.lin_temp_coef = value[i]
                else:
                    raise ValueError("adj_lin_temp_coef values must be floats.")
        else:
            raise TypeError("adj_lin_temp_coef must be a list or tuple with {} elements."
                            .format(len(self._adj_lin_temp_coef_members)))

    @property
    def adj_inv_temp_coef_members(self):
        return self._adj_inv_temp_coef_members

    @property
    def adj_inv_temp_coef(self):
        values = []
        for member in self._adj_inv_temp_coef_members:
            values.append(member.inv_temp_coef)
        return values

    @adj_inv_temp_coef.setter
    def adj_inv_temp_coef(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_inv_temp_coef_members):
            for i, member in enumerate(self._adj_inv_temp_coef_members):
                if isinstance(value[i], float):
                    member.inv_temp_coef = value[i]
                else:
                    raise ValueError("adj_inv_temp_coef values must be floats.")
        else:
            raise TypeError("adj_inv_temp_coef must be a list or tuple with {} elements."
                            .format(len(self._adj_inv_temp_coef_members)))

    @property
    def adj_ln_temp_coef_members(self):
        return self._adj_ln_temp_coef_members

    @property
    def adj_ln_temp_coef(self):
        values = []
        for member in self._adj_ln_temp_coef_members:
            values.append(member.ln_temp_coef)
        return values

    @adj_ln_temp_coef.setter
    def adj_ln_temp_coef(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_ln_temp_coef_members):
            for i, member in enumerate(self._adj_ln_temp_coef_members):
                if isinstance(value[i], float):
                    member.ln_temp_coef = value[i]
                else:
                    raise ValueError("adj_ln_temp_coef values must be floats.")
        else:
            raise TypeError("adj_ln_temp_coef must be a list or tuple with {} elements."
                            .format(len(self._adj_ln_temp_coef_members)))

    @property
    # TODO: Stylistic point.  Make sure a _ prepends any hidden class attribute abbreviations.
    def seg_num(self):
        sn = []
        for _pcp in self._pure_comp_parms:
            sn.append(_pcp.seg_num)
        return np.array(sn)

    @property
    def seg_diam(self):
        sd = []
        for _pcp in self._pure_comp_parms:
            sd.append(_pcp.seg_diam)
        return np.array(sd)

    @property
    def disp_energy(self):
        de = []
        for _pcp in self._pure_comp_parms:
            de.append(_pcp.disp_energy)
        return np.array(de)

    def set_adj_pure_comp_parms(self, adj_pure_comp_spec):
        # Define which parameters are adjustable in each PCSAFTParms in pure_comp_parms. Reset all 'member' lists to
        # empty lists as a default every time this method is called.
        self._adj_seg_num_members = []
        self._adj_seg_diam_members = []
        self._adj_disp_energy_members = []
        if adj_pure_comp_spec is None or adj_pure_comp_spec == 'none':
            pass
        elif adj_pure_comp_spec == 'all':
            self._adj_seg_num_members = self._pure_comp_parms[:]
            self._adj_seg_diam_members = self._pure_comp_parms[:]
            self._adj_disp_energy_members = self._pure_comp_parms[:]
        elif adj_pure_comp_spec == 'seg_num':
            self._adj_seg_num_members = self._pure_comp_parms[:]
        elif adj_pure_comp_spec == 'seg_diam':
            self._adj_seg_diam_members = self._pure_comp_parms[:]
        elif adj_pure_comp_spec == 'disp_energy':
            self._adj_disp_energy_members = self._pure_comp_parms[:]
        else:
            raise ValueError("adj_pure_comp_spec not valid.")

    def set_adj_binary_parms(self, adj_binary_spec):
        # Create lists that define which parameters are adjustable in each BinaryInterParm (bip) in binary_parms. Reset
        # all 'member' lists to empty lists as a default every time this method is called.
        self._adj_temp_indep_coef_members = []
        self._adj_lin_temp_coef_members = []
        self._adj_inv_temp_coef_members = []
        self._adj_ln_temp_coef_members = []
        if adj_binary_spec is None or adj_binary_spec == 'none':
            pass
        elif adj_binary_spec == 'all':
            for bip in self._binary_parms:
                # If an interaction parameter has not been initialized, then assume it is a temperature independent
                # interaction parameter and initialize the temp_indep_coef to 0.0.
                if not bip.defined():
                    bip.temp_indep_coef = 0.0
                # Only allow parameters to be adjusted for mixed-component binaries.
                if bip.comp_a != bip.comp_b:
                    # Coefficients can be adjustable if they have been initialized (changed from the 'None' default).
                    if bip.temp_indep_coef is not None:
                        self._adj_temp_indep_coef_members.append(bip)
                    if bip.lin_temp_coef is not None:
                        self._adj_lin_temp_coef_members.append(bip)
                    if bip.inv_temp_coef is not None:
                        self._adj_inv_temp_coef_members.append(bip)
                    if bip.ln_temp_coef is not None:
                        self._adj_ln_temp_coef_members.append(bip)
        else:
            raise ValueError("adj_binary_spec not valid.")

    def _create_pure_comp_parms(self):
        result = []
        for comp in self._comps.comps:
            new = PCSAFTParms(comp=comp, pc_saft_spec=self._pc_saft_spec)
            if new not in result:
                result.append(new)
        return result

    def load_pure_comp_parms(self):
        # Load equation of state parameters for all Comp objects. The absence of a PCSAFTParms object in a Comp object
        # triggers the init_parms procedure in PCSAFTParms to generate a first-pass estimate as a basis for further
        # parameter optimization.
        for pcp in self._pure_comp_parms:
            if self._eos == 'sPC-SAFT':
                if isinstance(pcp.comp.spc_saft_phys, PCSAFTParms) and pcp.comp.spc_saft_phys.defined():
                    pcp.seg_num = pcp.comp.spc_saft_phys.seg_num
                    pcp.seg_diam = pcp.comp.spc_saft_phys.seg_diam
                    pcp.disp_energy = pcp.comp.spc_saft_phys.disp_energy
                elif pcp.comp.spc_saft_phys is None:
                    pcp.init_parms()
                else:
                    raise TypeError("spc_saft_phys attribute for Comp object must be a PCSAFTParms object.")
            elif self._eos == 'PC-SAFT':
                if isinstance(pcp.comp.pc_saft_phys, PCSAFTParms) and pcp.comp.pc_saft_phys.defined():
                    pcp.seg_num = pcp.comp.pc_saft_phys.seg_num
                    pcp.seg_diam = pcp.comp.pc_saft_phys.seg_diam
                    pcp.disp_energy = pcp.comp.pc_saft_phys.disp_energy
                elif pcp.comp.pc_saft_phys is None:
                    pcp.init_parms()
                else:
                    raise TypeError("pc_saft_phys attribute for Comp object must be a PCSAFTParms object.")
            else:
                raise ValueError("eos not valid.")

    def _create_binary_parms(self):
        # Create list of all possible component-to-component interactions as BinaryInterParm objects.
        # TODO: Rethink the definition of a bip or assoc term.  Is it associatied with a physical or assciation term.
        #  Or is it assocated with an equation of state which is a collection of terms?
        result = []
        for comp_i in self._comps.comps:
            for comp_j in self._comps.comps:
                if comp_i == comp_j:
                    # TODO: Add keywords for all function arguments for clarity.
                    new = BinaryInterParm(comp_i, comp_j,
                                          self._pc_saft_spec, self._eos,
                                          "pure component", temp_indep_coef=0.0)
                    if new not in result:
                        result.append(new)
                else:
                    new = BinaryInterParm(comp_i, comp_j,
                                          self._pc_saft_spec, self._eos)
                    if new not in result:
                        result.append(new)
        return result

    def load_binary_parms(self, input):
        # Load coefficients and source from BinaryInterParm (bip) objects.
        if isinstance(input, BinaryInterParm):
            for _bp in self._binary_parms:
                if input == _bp:
                    _bp.temp_indep_coef = input.temp_indep_coef
                    _bp.lin_temp_coef = input.lin_temp_coef
                    _bp.inv_temp_coef = input.inv_temp_coef
                    _bp.ln_temp_coef = input.ln_temp_coef
                    _bp.source = input.source
        elif isinstance(input, PCSAFTPhysInter):
            for bp in input.binary_parms:
                for _bp in self._binary_parms:
                    if bp == _bp:
                        _bp.temp_indep_coef = bp.temp_indep_coef
                        _bp.lin_temp_coef = bp.lin_temp_coef
                        _bp.inv_temp_coef = bp.inv_temp_coef
                        _bp.ln_temp_coef = bp.ln_temp_coef
                        _bp.source = bp.source
        elif isinstance(input, list):
            for item in input:
                if isinstance(item, BinaryInterParm):
                    for _bp in self._binary_parms:
                        if item == _bp:
                            _bp.temp_indep_coef = item.temp_indep_coef
                            _bp.lin_temp_coef = item.lin_temp_coef
                            _bp.inv_temp_coef = item.inv_temp_coef
                            _bp.ln_temp_coef = item.ln_temp_coef
                            _bp.source = item.source
                else:
                    raise TypeError("Must pass a list of BinaryInterParm objects.")
        else:
            raise TypeError("Input not BinaryInterParm object, PhysInter object, or list of BinaryInterParm objects.")

    def k_ij(self, t=None):
        # Build array of interaction parameters for all comp combinations.
        n = len(self._comps.comps)
        k_ij = np.zeros((n, n))
        if t is not None and isinstance(t, float):
            for bp in self._binary_parms:
                i = self._comps.comps.index(bp.comp_a)
                j = self._comps.comps.index(bp.comp_b)
                k_ij[i, j] = bp.k_ij(t=t)
                k_ij[j, i] = bp.k_ij(t=t)
        else:
            raise ValueError("temperature must be a float.")
        return k_ij

    def __eq__(self, other):
        if isinstance(other, PCSAFTPhysInter):
            comps_eq = self.comps == other.comps
            spec_eq = self.pc_saft_spec == other.pc_saft_spec
            seg_num_eq = np.array_equal(self.seg_num, other.seg_num)
            seg_diam_eq = np.array_equal(self.seg_diam, other.seg_diam)
            disp_energy_eq = np.array_equal(self.disp_energy, other.disp_energy)
            kij_a_eq = np.array_equal(self.k_ij(298.15), other.k_ij(298.15))
            kij_b_eq = np.array_equal(self.k_ij(398.15), other.k_ij(398.15))
            return comps_eq and spec_eq and seg_num_eq and seg_diam_eq and disp_energy_eq and kij_a_eq and kij_b_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.comps, self.pc_saft_spec, self.seg_num, self.seg_diam, self.disp_energy, self.k_ij(298.15)))


class EOS(object):
    """Base class implementation of an equation of state (EOS).

    Notes
    -----
    All thermodynamic properties are calculable as partial derivatives of the residual Helmholtz function.  This is
    fortunate as many popular equations of state are constructed as a series of residual Helmholtz terms.

    Ar(T, V, ni, xai) = A(T, V, ni, xai) - Aig(T, V, ni, xai)

    Note that this is the Helmholtz energy (Ar) rather than the reduced Helmholtz energy (ar).  Multiply 'ar' by the
    total moles to obtain 'Ar'. It is often easier in practice to take derivatives of the reduced residual Helmholtz
    function (F) which has units of mole.

    F = Ar(T, V, ni, xai)/(R*T)

    The methods in this class estimate thermodynamic properties by estimating reduced residual Helmholtz derivatives
    numerically. These numeric derivatives can be replaced with analytic derivatives by overriding parent class methods
    in any child classes. Viewed in this way, this class is intended to be a base class (or a template) used for
    implementing any EOS. Numeric derivatives and can be used to check user-developed analytic derivatives (useful for
    implementing unit testing).  Multivariate finite difference formulas taken from [1]_.

    All thermodynamic property expressions are available (along with derivations and context) in [2]_.

    References
    ----------
    [1] “Finite difference.” Wikipedia, Wikimedia Foundation, 25 Sept 2021, en.wikipedia.org/wiki/Finite_difference
    [2] Michelsen, M. L.; Mollerup, J. Thermodynamic Models: Fundamental and Computational Aspects, 2nd ed.; Tie-Line
    Publications: Holte, Denmark, 2007.
    """

    _step_size = {'f_v': 10.0 ** -6.0,
                  'f_t': 10.0 ** -6.0,
                  'f_i': 10.0 ** -6.0,
                  'f_ai': 10.0 ** -6.0}

    def _n(self, ni):
        # Dimension is moles.
        return np.sum(ni)

    def _f(self, t, v, ni, xai):
        # This function MUST be defined in any child class that inherits the EOS base class.
        pass

    def _func_t_num(self, func, t, v, ni, xai, dt):
        # Finite difference first partial derivative of func(t, v, ni, xai) with respect to 't'.
        dt = t * dt
        return (func(t + dt, v, ni, xai) - func(t - dt, v, ni, xai)) / (2.0 * dt)

    def _f_t(self, t, v, ni, xai):
        return self._func_t_num(self._f, t, v, ni, xai, self._step_size['f_t'])

    def _func_tt_num(self, func, t, v, ni, xai, dt):
        # Finite difference second partial derivative of func(t, v, ni, xai) with respect to 't'.
        dt = t * dt
        return (func(t + dt, v, ni, xai) - 2.0 * func(t, v, ni, xai) + func(t - dt, v, ni, xai)) / (dt ** 2.0)

    def _f_tt(self, t, v, ni, xai):
        return self._func_tt_num(self._f, t, v, ni, xai, self._step_size['f_t'])

    def _func_v_num(self, func, t, v, ni, xai, dv):
        # Finite difference first partial derivative of func(t, v, ni, xai) with respect to 'v'.
        dv = v * dv
        return (func(t, v + dv, ni, xai) - func(t, v - dv, ni, xai)) / (2.0 * dv)

    def _f_v(self, t, v, ni, xai):
        return self._func_v_num(self._f, t, v, ni, xai, self._step_size['f_v'])

    def _func_vv_num(self, func, t, v, ni, xai, dv):
        # Finite difference second partial derivative of func(t, v, ni, xai) with respect to 'v'.
        dv = v * dv
        return (func(t, v + dv, ni, xai) - 2.0 * func(t, v, ni, xai) + func(t, v - dv, ni, xai)) / (dv ** 2.0)

    def _f_vv(self, t, v, ni, xai):
        return self._func_vv_num(self._f, t, v, ni, xai, self._step_size['f_v'])

    def _func_tv_num(self, func, t, v, ni, xai, dt, dv):
        # Finite difference second partial derivative of func(t, v, ni, xai) with respect to 't' and 'v'.
        dv = v * dv
        dt = t * dt
        return (func(t + dt, v + dv, ni, xai) - func(t + dt, v - dv, ni, xai)
                - func(t - dt, v + dv, ni, xai) + func(t - dt, v - dv, ni, xai)) / (4.0 * dt * dv)

    def _f_tv(self, t, v, ni, xai):
        return self._func_tv_num(self._f, t, v, ni, xai, self._step_size['f_t'], self._step_size['f_v'])

    def _func_i_grad_num(self, func, t, v, ni, xai, di):
        # Finite difference first partial derivative of scalar func(t, v, ni, xai) with respect to 'i'.  This
        # implementation mirrors << https://rh8liuqy.github.io/Finite_Difference.html >>.  'ni' is a vector of mole
        # numbers for each comp in the associated CompSet.  Return object is a np.array representing the Gradient.
        ni = np.array(ni, dtype=float)
        di = np.sum(ni) * di
        n = len(ni)
        result = np.zeros(n)
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            f1 = func(t, v, ni + di * ei, xai)
            f2 = func(t, v, ni - di * ei, xai)
            result[i] = (f1 - f2) / (2.0 * di)
        result = result.reshape(n, 1)
        return result

    def _f_i(self, t, v, ni, xai):
        return self._func_i_grad_num(self._f, t, v, ni, xai, self._step_size['f_i'])

    def _func_ij_hess_num(self, func, t, v, ni, xai, di):
        # Finite difference second partial derivative of scalar func(t, v, ni, xai) with respect to 'i'.  This
        # implementation mirrors << https://rh8liuqy.github.io/Finite_Difference.html >>.  'ni' is a vector of mole
        # numbers for each comp in the associated CompSet.  Return object is a np.array representing the Hessian.
        ni = np.array(ni, dtype=float)
        di = np.sum(ni) * di
        n = len(ni)
        result = np.matrix(np.zeros(n * n))
        result = result.reshape(n, n)
        for i in range(n):
            for j in range(n):
                ei = np.zeros(n)
                ei[i] = 1.0
                ej = np.zeros(n)
                ej[j] = 1.0
                f1 = func(t, v, ni + di * ei + di * ej, xai)
                f2 = func(t, v, ni + di * ei - di * ej, xai)
                f3 = func(t, v, ni - di * ei + di * ej, xai)
                f4 = func(t, v, ni - di * ei - di * ej, xai)
                result[i, j] = (f1 - f2 - f3 + f4) / (4.0 * di * di)
        return result

    def _f_ij(self, t, v, ni, xai):
        return self._func_ij_hess_num(self._f, t, v, ni, xai, self._step_size['f_i'])

    def _f_vi(self, t, v, ni, xai):
        return self._func_i_grad_num(self._f_v, t, v, ni, xai, self._step_size['f_i'])

    def _f_ti(self, t, v, ni,  xai):
        return self._func_i_grad_num(self._f_t, t, v, ni, xai, self._step_size['f_i'])

    def _ar(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.6
        # Ar(T, V, n) = R*T*F
        return R * t * self._f(t, v, ni, xai)

    def _p(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.7
        # P = -R*T*(dF/dV) + n*R*T/V
        return -R * t * self._f_v(t, v, ni, xai) + self._n(ni) * R * t / v

    def _z(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.8
        # Z = P*V/(n*R*T)
        return self._p(t, v, ni, xai) * v / (self._n(ni) * R * t)

    def _p_v(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.9
        # dP/dV = -R*T*(d2F/dV2) - n*R*T/V**2
        return -R * t * self._f_vv(t, v, ni, xai) - self._n(ni) * R * t / (v ** 2.0)

    def _p_t(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.10
        # dP/dT = -R*T*(d2F/dTdV) + P/T
        return -R * t * self._f_tv(t, v, ni, xai) + self._p(t, v, ni, xai) / t

    def _p_i(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.11
        # dP/dni = -R*T*(d2F/dVdni) + R*T/V
        return -R * t * self._f_vi(t, v, ni, xai) + R * t / v

    def _v_i(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.12
        # dV/dni = -(dP/dni)/(dP/dV)
        return -self._p_i(t, v, ni, xai) / self._p_v(t, v, ni, xai)

    def _ln_phi(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.13
        # ln(phi_i) = (dF/dni) - ln(Z)
        # Note: ln(phi_i) estimated for all comps in the associated CompSet and returned as an np.array.
        return self._f_i(t, v, ni, xai) - np.log(self._z(t, v, ni, xai))

    def _ln_phi_t(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.14
        # dln(phi_i)/dT = (d2F/dTdni) + 1.0/T - (dV/dni)*(dP/dT)/(R*T)
        return self._f_ti(t, v, ni, xai) + 1.0 / t - self._v_i(t, v, ni, xai) * self._p_t(t, v, ni, xai) / (R * t)

    def _ln_phi_p(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.15
        # dln(phi_i)/dP = (dV/dni)/(R*T) - 1.0/P
        return self._v_i(t, v, ni, xai) / (R * t) - 1.0 / self._p(t, v, ni, xai)

    def _func_i_jac_num(self, func, t, v, ni, xai, di):
        # Finite difference first partial derivative of vector func(t, v, ni) with respect to 'i'.  This implementation
        # mirrors << https://rh8liuqy.github.io/Finite_Difference.html >>.  'ni' is a vector of mole numbers for each
        # comp in the associated CompSet.  Return object is a np.array representing the Jacobian.
        ni = np.array(ni, dtype=float)
        di = np.sum(ni) * di
        n_row = len(func(t, v, ni, xai))
        n_col = len(ni)
        result = np.zeros(n_row * n_col)
        result = result.reshape(n_row, n_col)
        for i in range(n_col):
            ei = np.zeros(n_col)
            ei[i] = 1.0
            f1 = func(t, v, ni + di * ei, xai)
            f2 = func(t, v, ni - di * ei, xai)
            result[:,i] = (f1 - f2) / (2.0 * di)
        return result

    def _ln_phi_j(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.16 (numerically implemented here)
        #
        # This implementation estimates _ln_phi_j by numerically estimating the Jacobian of _ln_phi (a vector function).
        # _ln_phi is often analytically implemented in many equations of state.  This means that estimating _ln_phi_j by
        # numerical evaluation of the Jacobian of _ln_phi involves only first-order central difference approximations.
        # Estimating _ln_phi_j using Equation 2.16 involves estimating _f_ij using second-order central difference
        # approximations which is expected to degrade accuracy to some degree.
        return self._func_i_jac_num(self._ln_phi, t, v, ni, xai, self._step_size['f_i'])

    def _sr(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.17
        # Sr(T, V, n) = -R*T*(dF/dT) - R*F
        return -R * t * self._f_t(t, v, ni, xai) - R * self._f(t, v, ni, xai)

    def _cvr(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.18
        # Cvr(T, V, n) = -R*(T**2.0)*(d2F/dT2) - 2.0*R*T*(dF/dT)
        return -R * (t ** 2.0) * self._f_tt(t, v, ni, xai) - 2.0 * R * t * self._f_t(t, v, ni, xai)

    def _cpr(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.19
        # Cpr(T, V, n) = -T*((dP/dT)**2)/(dP/dV) - n*R + Cvr(T, V, n)
        return -t * (self._p_t(t, v, ni, xai) ** 2.0) / self._p_v(t, v, ni, xai) - self._n(ni) * R + \
               self._cvr(t, v, ni, xai)

    def _hr(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.20
        # Hr(T, P, n) = Ar(T, V, n) + T*Sr(T, V, n) + P*V - n*R*T
        return self._ar(t, v, ni, xai) + t * self._sr(t, v, ni, xai) + self._p(t, v, ni, xai) * v - self._n(ni) * R * t

    def _gr(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.21
        # Gr(T, P, n) = Ar(T, V, n) + P*V - n*R*T - n*R*T*ln(Z)
        return self._ar(t, v, ni, xai) + self._p(t, v, ni, xai) * v - self._n(ni) * R * t - \
               self._n(ni) * R * t * np.log(self._z(t, v, ni, xai))


class Assoc(EOS):
    """Base class implementation of a general associating term (EOS).

    Notes
    -----
    The association term is implemented using the Q-function approach proposed by Michelsen and Hendriks [1]_. The value
    of Q at a stationary point is:

    Qsp = Ar(T, V, ni, xai)/(R*T) = F

    Thermodynamic properties can be estimated by taking derivatives of Qsp.  In all of these expressions, xai is the
    fraction of 'a' sites on molecule 'i' that do not form bonds with other active sites.  xai is a list of lists where
    the structure is as follows:

    xai = [[xa for  each site in Comp 1], [xa for each site in Comp 2], ..., [xa for each site in Comp n]]

    The length of xai is equal to the length of the associated CompSet (i.e. xai and ni have the same length).  The
    length of each xai element corresponds to the length of the assoc_sites attribute of each Comp in the associated
    CompSet.  If there are no sites on a molecule, then the corresponding element's truth value must be False (which is
    represented by None this library for simplicity).

    References
    -----
    [1] Michelsen, M. L.; Hendriks, E. M. Physical properties from association models. Fluid Phase Equilib. 2001, 180,
    165-174.
    [2] Kontogeorgis, G. M.; Folas, G. K. Thermodynamic Models for Industrial Applications, John Wiley & Sons, Ltd:
    West Sussex, UK, 2010.
    [3] de Villiers, A. J. Evaluation and improvement of the sPC-SAFT equation of state for complex mixtures. Ph.D.
    Dissertation, Stellenbosch University, Stellenbosch, South Africa, 2011.
    """

    def _qsp(self, t, v, ni, xai):
        return

    def _qsp_v(self, t, v, ni, xai):
        return

    def _qsp_t(self, t, v, ni, xai):
        return

    def _qsp_i(self, t, v, ni, xai):
        return

    def _ln_g_v(self, t, v, ni):
        return

    def _ln_g_i(self, t, v, ni):
        # This function MUST be defined in any child class that inherits the Association base class.
        return


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
        return self.r*t/(v-b)-a/((v+self.d1*b)*(v+self.d2*b))

    def z(self, t, v, a, b):
        return 1.0/(1.0-b/v)-(a/(self.r*t*b))*(b/v)/((1.0+self.d1*b/v)*(1.0+self.d2*b/v))

    def F(self, n, t, V, B, D):
        return -n * g(V, B) - D(t) * f(V, B) / t

    def g(self, V, B):
        return np.log(V - B) - np.log(V)

    def f(self, V, B):
        return np.log((V + self.d1 * B) / (V + self.d2 * B)) / (self.r * B * (self.d1 - self.d2))


class sPCSAFT(EOS):
    """sPC-SAFT equation of state for a phase."""
    def __init__(self, phys_inter=None, assoc_inter=None):
        if phys_inter is None:
            raise ValueError("Must pass a PCSAFTPhysInter object to the constructor.")
        else:
            self.phys_inter = phys_inter
            self.assoc_inter = assoc_inter

    @property
    def phys_inter(self):
        return self._phys_inter

    @phys_inter.setter
    def phys_inter(self, value):
        try:
            self._phys_inter
        except AttributeError:
            if isinstance(value, PCSAFTPhysInter):
                self._phys_inter = value
            else:
                raise TypeError("Must pass a PCSAFTPhysInter object to the constructor.")

    @property
    def assoc_inter(self):
        return self._assoc_inter

    @assoc_inter.setter
    def assoc_inter(self, value):
        try:
            self._assoc_inter
        except AttributeError:
            if isinstance(value, AssocInter) or value is None:
                self._assoc_inter = value
            else:
                raise TypeError("Must pass a AssocInter object to the constructor.")

    def vol_solver(self, p=None, t=None, ni=None, xai=None, root=None, prior=None, eps=10**-2, max_iter=100):
        """Solve for volume when temperature, mole numbers, and pressure are specified."""
        # Check inputs for consistency.
        if not isinstance(p, float) and p > 0.0:
            raise TypeError("p must be a positive float.")
        elif not isinstance(t, float) and t > 0.0:
            raise TypeError("t must be a positive float.")
        elif not isinstance(ni, (list, tuple, np.ndarray)):
            raise TypeError("ni must be a list, tuple, or np.ndarray.")
        elif not all(isinstance(i, (float, np.floating)) for i in ni):
            raise TypeError("ni can only contain floats.")
        elif not all(i >= 0.0 for i in ni):
            raise ValueError("ni can only contain positive floats.")
        elif not isinstance(eps, float) and eps > 0.0:
            raise TypeError("eps must be a positive float.")
        elif not isinstance(max_iter, int) and max_iter > 0:
            raise TypeError("max_iter must be a positive integer.")

        # Newton's Method iterations solving for pressure by adjusting volume. Used Newton's method structure presented
        # in <<https://secure.math.ubc.ca/~pwalls/math-python/roots-optimization/newton/>>.  Convergence is safeguarded
        # by including modifications suggested in Michelsen, M. L. Robust and Efficient Solution Procedures for
        # Association Models. Ind. Eng. Chem. Res., 45, 2006.

        # Equations to solve.
        def func(t, v, ni, xai):
            return self._p(t, v, ni, xai) - p

        def func_v(t, v, ni, xai):
            # TODO: Consider re-defining with numeric drivative of P.
            # TODO: Need to define function call in terms of xai.
            return self._p_v(t, v, ni, xai) * self._v_from_eta_eta(t, self._eta(t, v, ni), ni)

        # Define reduced density limits based on physics of the problem.
        eta_min = 0.0
        eta_max = 0.74047
        eta_abs_min = 0.0
        eta_abs_max = 0.74047

        # Initial reduced density guess based on user specification.
        if root == 'liquid':
            eta = 0.5
        elif root == 'vapor':
            eta = 0.01
        elif root == 'prior':
            if isinstance(prior, float) and prior > 0.0:
                eta = self._eta(t, prior, ni)
            else:
                raise ValueError("prior must be a positive float.")
        else:
            raise ValueError("root specification must be either liquid, vapor, or prior.")

        for n in range(0, max_iter):
            # Convert reduced density (iteration variable) into volume (practical variable for EOS evaluation).
            print("eta: {}".format(eta))
            v = self._v_from_eta(t, eta, ni)
            # Evaluate f and check for convergence.
            _func = func(t, v, ni, xai)
            if abs(_func) < eps:
                return v, self._gr(t, v, ni, xai)
            # Update iteration variable limits.
            if _func > 0.0:
                eta_max = eta
            else:
                eta_min = eta
            # Evaluate the derivative of f and check for zero denominator. If zero denominator encountered, then
            # estimate new reduced density using a limit bisection and proceed to the next iteration.
            _func_v = func_v(t, v, ni, xai)
            if _func_v == 0.0:
                eta = (eta_min + eta_max) / 2.0
                continue
            # Estimate new reduced density using Newtons Method.
            eta = eta - _func / _func_v
            # Replace reduced density with a limit bisection if Newton's correction fell outside the limits.
            if not eta_min < eta < eta_max:
                eta = (eta_min + eta_max) / 2.0
            # Raise error if reduced density exceeds eta_abs_max.
            if eta <= eta_abs_min or eta >= eta_abs_max:
                raise RuntimeError("Exceeded density limits.")
        raise RuntimeError("Exceeded maximum iterations. No solution found.")

    def _v_from_eta(self, t, eta, ni):
        # Dimension is m**3
        #
        # Note that eta cannot exceed ~0.74048 (closest packing of segments).  The argument 'v' has units of [m3] which
        # is converted to units of [Å**3] by multiplying by  10.0 ** 30.0.
        return PI * NA * self._d3(t, ni) * np.sum(ni * self.phys_inter.seg_num) / (6.0 * eta * 10.0 ** 30.0)

    def _v_from_eta_eta(self, t, eta, ni):
        # Dimension is m**3
        #
        # Note that eta cannot exceed ~0.74048 (closest packing of segments).  The argument 'v' has units of [m3] which
        # is converted to units of [Å**3] by multiplying by  10.0 ** 30.0.
        return - PI * NA * self._d3(t, ni) * np.sum(ni * self.phys_inter.seg_num) / (6.0 * (eta ** 2.0) * 10.0 ** 30.0)

    # This implementation of sPC-SAFT follows Appendix F in de Villers, A. J. Evaluation and improvement of the sPC-SAFT
    # equation of state for complex mixtures. PhD Thesis. Stellenbosch University. 2011.

    def _m(self, ni):
        # Dimensionless.
        # de Villers (2011), Equation F-2
        return np.sum(ni * self.phys_inter.seg_num) / self._n(ni)

    def _m_i(self, ni):
        # de Villers (2011), Equation F-8
        return self.phys_inter.seg_num / self._n(ni) - np.sum(ni * self.phys_inter.seg_num) / (self._n(ni) ** 2.0)

    def _di(self, t):
        # Dimension is Å.
        # de Villers (2011), Equation F-6
        # TODO: Make the 0.12 an adjustable parameter, maybe ci. Must be complete before implementing temp derivatives.
        # TODO: Integrate ck_const into PCSAFTPHYSINTER class.
        return self.phys_inter.seg_diam * (1.0 - 0.12 * np.exp(-3.0 * self.phys_inter.disp_energy / t))

    def _d(self, t, ni):
        # Dimension is Å
        # de Villers (2011), Equation F-5
        return self._d3(t, ni) ** (1.0 / 3.0)

    def _d3(self, t, ni):
        # Dimension is Å**3
        # de Villers (2011), Equation F-5
        return np.sum(ni * self.phys_inter.seg_num * self._di(t) ** 3.0) / np.sum(ni * self.phys_inter.seg_num)

    def _d_i(self, t, ni):
        # de Villers (2011), Equation F-10
        return (self.phys_inter.seg_num * self._di(t) ** 3.0 / np.sum(ni * self.phys_inter.seg_num) -
                self.phys_inter.seg_num * np.sum(ni * self.phys_inter.seg_num * self._di(t) ** 3.0) /
                np.sum(ni * self.phys_inter.seg_num) ** 2.0) / (3.0 * self._d(t, ni) ** 2.0)

    def _rho(self, v, ni):
        # Dimension is 1/Å**3.
        # de Villers (2011), Equation F-57
        #
        # Number density of molecules (and not molar density).  Number density is formally defined as the number of
        # molecules per unit volume.  This means that the units for ρ = N / V are really [count / volume].  The "count"
        # is formally treated as dimensionless.  This can be understood by comparing alternative formulations of the
        # ideal gas law and evaluating units.  This representation of rho in "statistical mechanics" units can be tricky
        # for chemists and engineers.
        #
        #    PV = NkT -> P = kT(N/V) -> P = kTρ -> [Pa] = [J/K][K][?] -> [kg/m.s2] = [kg.m2/s2.K][K][?] -> [] = [m3][?]
        return NA * self._n(ni) / self._v_a3(v)

    def _v_a3(self, v):
        # Dimension is Å**3.
        # Function not a part of original de Villers (2011) implementation and was added for convenience.
        #
        # The argument 'v' has units of [m**3] which is converted to units of [Å**3] by multiplying by  10.0 ** 30.0.
        return v * (10.0 ** 30.0)

    def _eta(self, t, v, ni):
        # Dimensionless.
        # de Villers (2011), Equation F-4
        return PI * NA * self._d3(t, ni) * np.sum(ni * self.phys_inter.seg_num) / (6.0 * self._v_a3(v))

    def _eta_v(self, t, v, ni):
        # de Villers (2011), Equation F-16
        #
        # Note that the conversion factor from [m3] to [Å**3] exists here but not in de Villers. This is due to the
        # conversion in this implementation being separated into _v_a3 and application of the chain rule to _v_a3.
        return -(10.0 ** 30.0) * self._eta(t, v, ni) / self._v_a3(v)

    def _eta_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._eta, t, v, ni, None, self._step_size['f_i'])

    def _eta_i(self, t, v, ni):
        # de Villers (2011), Equation F-9
        return (NA * PI / (6.0 * self._v_a3(v))) * (self._d3(t, ni) * self.phys_inter.seg_num +
                                                    3.0 * (self._d(t, ni) ** 2.0) * self._d_i(t, ni) *
                                                    np.sum(ni * self.phys_inter.seg_num))

    def _fhs(self, t, v, ni):
        # Dimensionless.
        # de Villers (2011), Equation F-3
        return self._m(ni) * self._n(ni) * ((4.0 * self._eta(t, v, ni) - 3.0 * self._eta(t, v, ni) ** 2.0) /
                                            ((1.0 - self._eta(t, v, ni)) ** 2.0))

    def _fhs_v(self, t, v, ni):
        # Dimensionless.
        # de Villers (2011), Equation F-15
        return 2.0 * self._m(ni) * self._n(ni) * self._eta_v(t, v, ni) * ((4.0 * self._eta(t, v, ni) -
                                                                           3.0 * self._eta(t, v, ni) ** 2.0) /
                                                                          ((1.0 - self._eta(t, v, ni)) ** 3.0)) + \
               self._m(ni) * self._n(ni) * ((4.0 * self._eta_v(t, v, ni) -
                                             6.0 * self._eta(t, v, ni) * self._eta_v(t, v, ni)) /
                                            ((1.0 - self._eta(t, v, ni)) ** 2.0))

    def _fhs_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._fhs, t, v, ni, None, self._step_size['f_v'])

    def _fhs_i(self, t, v, ni):
        # de Villers (2011), Equation F-7
        return self._m_i(ni) * self._n(ni) * ((4.0 * self._eta(t, v, ni) - 3.0 * self._eta(t, v, ni) ** 2.0) /
                                                   ((1.0 - self._eta(t, v, ni)) ** 2.0)) + \
               self._m(ni) * ((4.0 * self._eta(t, v, ni) - 3.0 * self._eta(t, v, ni) ** 2.0) /
                              ((1.0 - self._eta(t, v, ni)) ** 2.0)) + \
               2.0 * self._m(ni) * self._n(ni) * self._eta_i(t, v, ni) * ((4.0 * self._eta(t, v, ni) -
                                                                                3.0 * self._eta(t, v, ni) ** 2.0) /
                                                                               ((1.0 - self._eta(t, v, ni)) ** 3.0)) + \
               self._m(ni) * self._n(ni) * ((4.0 * self._eta_i(t, v, ni) -
                                             6.0 * self._eta(t, v, ni) * self._eta_i(t, v, ni)) /
                                            ((1.0 - self._eta(t, v, ni)) ** 2.0))

    def _ghs(self, t, v, ni):
        # Dimensionless.
        # de Villers (2011), Equation F-37
        return (2.0 - self._eta(t, v, ni)) / (2.0 * (1.0 - self._eta(t, v, ni)) ** 3.0)

    def _ghs_v(self, t, v, ni):
        # de Villers (2011), Equation F-43
        return self._eta_v(t, v, ni) * (5.0 - 2.0 * self._eta(t, v, ni)) / (2.0 * (1.0 - self._eta(t, v, ni)) ** 4.0)

    def _ghs_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._ghs, t, v, ni, None, self._step_size['f_v'])

    def _ghs_i(self, t, v, ni):
        # de Villers (2011), Equation F-39
        return self._eta_i(t, v, ni) * (-1.0 / (2.0 * ((1.0 - self._eta(t, v, ni)) ** 3.0)) +
                                             3.0 * (2.0 - self._eta(t, v, ni)) /
                                             (2.0 * ((1.0 - self._eta(t, v, ni)) ** 4.0)))

    def _fhc(self, t, v, ni):
        # de Villers (2011), Equation F-36
        return np.log(self._ghs(t, v, ni)) * np.sum(ni * (1.0 - self.phys_inter.seg_num))

    def _fhc_v(self, t, v, ni):
        # de Villers (2011), Equation F-42
        return self._ghs_v(t, v, ni) * np.sum(ni * (1.0 - self.phys_inter.seg_num)) / self._ghs(t, v, ni)

    def _fhc_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._fhc, t, v, ni, None, self._step_size['f_v'])

    def _fhc_i(self, t, v, ni):
        # de Villers (2011), Equation F-38
        return (1.0 - self.phys_inter.seg_num) * np.log(self._ghs(t, v, ni)) + \
               np.sum(ni * (1.0 - self.phys_inter.seg_num)) * self._ghs_i(t, v, ni) / self._ghs(t, v, ni)

    def _ai(self, i, ni):
        # de Villers (2011), Equation F-69
        return self.phys_inter.pc_saft_spec.a[i, 0] + \
               self.phys_inter.pc_saft_spec.a[i, 1] * (self._m(ni) - 1.0) / self._m(ni) + \
               self.phys_inter.pc_saft_spec.a[i, 2] * (self._m(ni) - 1.0) * (self._m(ni) - 2.0) / (self._m(ni) ** 2.0)

    def _ai_i(self, i, ni):
        # de Villers (2011), F-76
        return self._m_i(ni) * (-4.0 * self.phys_inter.pc_saft_spec.a[i, 2] +
                                     self._m(ni) * (self.phys_inter.pc_saft_spec.a[i, 1] +
                                                    3.0 * self.phys_inter.pc_saft_spec.a[i, 2])) / self._m(ni) ** 3.0

    def _bi(self, i, ni):
        # de Villers (2011), Equation F-70
        return self.phys_inter.pc_saft_spec.b[i, 0] + \
               self.phys_inter.pc_saft_spec.b[i, 1] * (self._m(ni) - 1.0) / self._m(ni) + \
               self.phys_inter.pc_saft_spec.b[i, 2] * (self._m(ni) - 1.0) * (self._m(ni) - 2.0) / (self._m(ni) ** 2.0)

    def _bi_i(self, i, ni):
        # de Villers (2011), F-83
        return self._m_i(ni) * (-4.0 * self.phys_inter.pc_saft_spec.b[i, 2] +
                                     self._m(ni) * (self.phys_inter.pc_saft_spec.b[i, 1] +
                                                    3.0 * self.phys_inter.pc_saft_spec.b[i, 2])) / self._m(ni) ** 3.0

    def _i1(self, t, v, ni):
        # de Villers (2011), Equation F-64
        return np.sum(np.array([self._ai(i, ni) * self._eta(t, v, ni) ** i for i in range(7)]))

    def _i1_v(self, t, v, ni):
        # de Villers (2011), Equation F-96
        return self._eta_v(t, v, ni) * np.sum(np.array([i * self._ai(i, ni) * self._eta(t, v, ni) ** (i - 1.0)
                                                        for i in range(7)]))

    def _i1_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._i1, t, v, ni, None, self._step_size['f_v'])

    def _i1_i(self, t, v, ni):
        # de Villers (2011), F-75
        # TODO: Not fully tested yet.  Need to make sure axis=0 is really right.
        return np.sum(np.array([i * self._ai(i, ni) * self._eta_i(t, v, ni) * self._eta(t, v, ni) ** (i - 1) +
                                self._ai_i(i, ni) * self._eta(t, v, ni) ** i
                                for i in range(7)]), axis=0)

    def _i2(self, t, v, ni):
        # de Villers (2011), Equation F-65
        return np.sum(np.array([self._bi(i, ni) * self._eta(t, v, ni) ** i for i in range(7)]))

    def _i2_v(self, t, v, ni):
        # de Villers (2011), Equation F-101
        return self._eta_v(t, v, ni) * np.sum(np.array([i * self._bi(i, ni) * self._eta(t, v, ni) ** (i - 1.0)
                                                        for i in range(7)]))

    def _i2_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._i2, t, v, ni, None, self._step_size['f_v'])

    def _i2_i(self, t, v, ni):
        # de Villers (2011, F-87
        # TODO: Not fully tested yet.  Need to make sure axis=0 is really right.
        return np.sum(np.array([i * self._bi(i, ni) * self._eta_i(t, v, ni) * self._eta(t, v, ni) ** (i - 1) +
                                self._bi_i(i, ni) * self._eta(t, v, ni) ** i
                                for i in range(7)]), axis=0)

    def _epsij(self, t):
        # de Villers (2011), by definition
        return np.sqrt(np.outer(self.phys_inter.disp_energy, self.phys_inter.disp_energy)) * \
               (1.0 - self.phys_inter.k_ij(t))

    def _sigij(self):
        # de Villers (2011), by definition
        return 0.5 * np.add.outer(self.phys_inter.seg_diam, np.transpose(self.phys_inter.seg_diam))

    def _m2eps1sig3(self, t, ni):
        # de Villers (2011), Equation F-61
        return np.sum(np.outer(ni, ni) *
                      np.outer(self.phys_inter.seg_num, self.phys_inter.seg_num) *
                      np.power(self._epsij(t) / t, 1.0) *
                      np.power(self._sigij(), 3.0)) / (self._n(ni) ** 2.0)

    def _m2eps1sig3_i(self, t, ni):
        # de Villers (2011), Equation F-74
        # TODO: Not fully tested yet.  THis one is tricky.  Summation of array multiplication.  axis=1 makes this work.
        return 2.0 * self.phys_inter.seg_num * np.sum(ni *
                                                      self.phys_inter.seg_num *
                                                      np.power(self._epsij(t) / t, 1.0) *
                                                      np.power(self._sigij(), 3.0), axis=1) / (self._n(ni) ** 2.0) - \
               2.0 * np.sum(np.outer(ni, ni) *
                            np.outer(self.phys_inter.seg_num, self.phys_inter.seg_num) *
                            np.power(self._epsij(t) / t, 1.0) *
                            np.power(self._sigij(), 3.0)) / (self._n(ni) ** 3.0)

    def _m2eps2sig3(self, t, ni):
        # de Villers (2011), Equation F-63
        return np.sum(np.outer(ni, ni) *
                      np.outer(self.phys_inter.seg_num, self.phys_inter.seg_num) *
                      np.power(self._epsij(t) / t, 2.0) *
                      np.power(self._sigij(), 3.0)) / (self._n(ni) ** 2.0)

    def _m2eps2sig3_i(self, t, ni):
        # de Villers (2011), Equation F-79
        # TODO: Not fully tested yet.  THis one is tricky.  Summation of array multiplication.  axis=1 makes this work.
        return 2.0 * self.phys_inter.seg_num * np.sum(ni *
                                                      self.phys_inter.seg_num *
                                                      np.power(self._epsij(t) / t, 2.0) *
                                                      np.power(self._sigij(), 3.0), axis=1) / (self._n(ni) ** 2.0) - \
               2.0 * np.sum(np.outer(ni, ni) *
                            np.outer(self.phys_inter.seg_num, self.phys_inter.seg_num) *
                            np.power(self._epsij(t) / t, 2.0) *
                            np.power(self._sigij(), 3.0)) / (self._n(ni) ** 3.0)

    def _c0(self, t, v, ni):
        # de Villers (2011), Equation F-66
        return (1.0 + self._m(ni) * (8.0 * self._eta(t, v, ni) - 2.0 * self._eta(t, v, ni) ** 2.0) /
                ((1.0 - self._eta(t, v, ni)) ** 4.0) +
                (1.0 - self._m(ni)) * (20.0 * self._eta(t, v, ni) - 27.0 * self._eta(t, v, ni) ** 2.0 + 12.0 *
                                       self._eta(t, v, ni) ** 3.0 - 2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0)))

    def _c0_v(self, t, v, ni):
        # de Villers (2011), Equation F-99
        return (2.0 * (1.0 - self._m(ni)) * self._eta_v(t, v, ni) *
                (20.0 * self._eta(t, v, ni) - 27.0 * self._eta(t, v, ni) ** 2.0 + 12.0 * self._eta(t, v, ni) ** 3.0 -
                 2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 3.0))) + \
               (2.0 * (1.0 - self._m(ni)) * self._eta_v(t, v, ni) *
                (20.0 * self._eta(t, v, ni) - 27.0 * self._eta(t, v, ni) ** 2.0 + 12.0 * self._eta(t, v, ni) ** 3.0 -
                 2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 3.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0))) + \
               (4.0 * self._m(ni) * self._eta_v(t, v, ni) * (8.0 * self._eta(t, v, ni) -
                                                             2.0 * self._eta(t, v, ni) ** 2.0) /
                ((1.0 - self._eta(t, v, ni)) ** 5.0)) + \
               (self._m(ni) * self._eta_v(t, v, ni) * (8.0 - 4.0 * self._eta(t, v, ni)) /
                ((1.0 - self._eta(t, v, ni)) ** 4.0)) + \
               ((1.0 - self._m(ni)) * self._eta_v(t, v, ni) * (20.0 - 54.0 * self._eta(t, v, ni) +
                                                               36.0 * self._eta(t, v, ni) ** 2.0 -
                                                               8.0 * self._eta(t, v, ni) ** 3.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0)))

    def _c0_v_num(self,t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._c0, t, v, ni, None, self._step_size['f_v'])

    def _c0_i(self, t, v, ni):
        # de Villers (2011), F-80
        return (self._m_i(ni) * (8.0 * self._eta(t, v, ni) - 2.0 * self._eta(t, v, ni) ** 2.0) /
                ((1.0 - self._eta(t, v, ni)) ** 4.0) -
                self._m_i(ni) * (20.0 * self._eta(t, v, ni) - 27.0 * self._eta(t, v, ni) ** 2.0 +
                                       12.0 * self._eta(t, v, ni) ** 3.0 - 2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0)) +
                2.0 * (1.0 - self._m(ni)) * self._eta_i(t, v, ni) * (20.0 * self._eta(t, v, ni) -
                                                                          27.0 * self._eta(t, v, ni) ** 2.0 +
                                                                          12.0 * self._eta(t, v, ni) ** 3.0 -
                                                                          2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 3.0)) +
                2.0 * (1.0 - self._m(ni)) * self._eta_i(t, v, ni) * (20.0 * self._eta(t, v, ni) -
                                                                          27.0 * self._eta(t, v, ni) ** 2.0 +
                                                                          12.0 * self._eta(t, v, ni) ** 3.0 -
                                                                          2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 3.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0)) +
                4.0 * self._m(ni) * self._eta_i(t, v, ni) * (8.0 * self._eta(t, v, ni) -
                                                                  2.0 * self._eta(t, v, ni) ** 2.0) /
                ((1.0 - self._eta(t, v, ni)) ** 5.0) +
                self._m(ni) * self._eta_i(t, v, ni) * (8.0 - 4.0 * self._eta(t, v, ni)) /
                ((1.0 - self._eta(t, v, ni)) ** 4.0) +
                (1.0 - self._m(ni)) * self._eta_i(t, v, ni) * (20.0 - 54.0 * self._eta(t, v, ni) +
                                                                    36.0 * self._eta(t, v, ni) ** 2.0 -
                                                                    8.0 * self._eta(t, v, ni) ** 3.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0)))

    def _c1(self, t, v, ni):
        # de Villers (2011), Equation F-67
        return (1.0 + self._m(ni) * (8.0 * self._eta(t, v, ni) - 2.0 * self._eta(t, v, ni) ** 2.0) /
                ((1.0 - self._eta(t, v, ni)) ** 4.0) +
                (1.0 - self._m(ni)) * (20.0 * self._eta(t, v, ni) - 27.0 * self._eta(t, v, ni) ** 2.0 + 12.0 *
                                       self._eta(t, v, ni) ** 3.0 - 2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0))) ** -1.0

    def _c1_v(self, t, v, ni):
        # de Villers (2011), Equation F-100
        return -self._c0_v(t, v, ni) / (self._c0(t, v, ni) ** 2.0)

    def _c1_v_num(self,t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._c1, t, v, ni, None, self._step_size['f_v'])

    def _c1_i(self, t, v, ni):
        # de Villers (2011), Equation F-81
        return -self._c0_i(t, v, ni) / self._c0(t, v, ni) ** 2.0

    def _fdisp(self, t, v, ni):
        # de Villers, Equation F-59
        return -(2.0 * PI * NA * (self._n(ni) ** 2.0) * self._i1(t, v, ni) * self._m2eps1sig3(t, ni) / self._v_a3(v) +
                 PI * NA * (self._n(ni) ** 2.0) * self._c1(t, v, ni) * self._m(ni) * self._i2(t, v, ni) *
                 self._m2eps2sig3(t, ni) / self._v_a3(v))

    def _fdisp1(self, t, v, ni):
        # Created for debugging purposes (allows us to test partial derivatives numerically).
        return -(2.0 * PI * NA * (self._n(ni) ** 2.0) * self._i1(t, v, ni) * self._m2eps1sig3(t, ni) / self._v_a3(v))

    def _fdisp1_v(self, t, v, ni):
        # de Villers (2011), Equation F-95
        #
        # Note that the conversion factor from [m3] to [Å**3] exists here but not in de Villers. This is due to the
        # conversion in this implementation being separated into _v_a3 and application of the chain rule to _v_a3.
        return ((10.0 ** 30.0) * 2.0 * PI * NA * (self._n(ni) ** 2.0) * self._i1(t, v, ni) * self._m2eps1sig3(t, ni) /
                (self._v_a3(v) ** 2.0)) - \
               (2.0 * PI * NA * (self._n(ni) ** 2.0) * self._m2eps1sig3(t, ni) * self._i1_v(t, v, ni) / self._v_a3(v))

    def _fdisp1_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._fdisp1, t, v, ni, None, self._step_size['f_v'])

    def _fdisp1_i(self, t, v, ni):
        # de Villers (2011), Equation F-73
        return -PI * NA * (4.0 * self._n(ni) * self._i1(t, v, ni) * self._m2eps1sig3(t, ni) +
                           2.0 * (self._n(ni) ** 2.0) * self._i1(t, v, ni) * self._m2eps1sig3_i(t, ni) +
                           2.0 * (self._n(ni) ** 2.0) * self._i1_i(t, v, ni) * self._m2eps1sig3(t, ni)) / \
               self._v_a3(v)

    def _fdisp2(self, t, v, ni):
        # Created for debugging purposes (allows us to test partial derivatives numerically).
        return -(PI * NA * (self._n(ni) ** 2.0) * self._c1(t, v, ni) * self._m(ni) * self._i2(t, v, ni) *
                 self._m2eps2sig3(t, ni) / self._v_a3(v))

    def _fdisp2_v(self, t, v, ni):
        # de Villers (2011), Equation F-98
        #
        # Note that the conversion factor from [m3] to [Å**3] exists here but not in de Villers. This is due to the
        # conversion in this implementation being separated into _v_a3 and application of the chain rule to _v_a3.
        return ((10.0 ** 30.0) * PI * NA * (self._n(ni) ** 2.0) * self._m(ni) * self._c1(t, v, ni) *
                 self._i2(t, v, ni) *self._m2eps2sig3(t, ni) / (self._v_a3(v) ** 2.0)) - \
               (PI * NA * (self._n(ni) ** 2.0) * self._m(ni) * self._i2(t, v, ni) * self._m2eps2sig3(t, ni) *
                self._c1_v(t, v, ni) / self._v_a3(v)) - \
               (PI * NA * (self._n(ni) ** 2.0) * self._m(ni) * self._c1(t, v, ni) * self._m2eps2sig3(t, ni) *
                self._i2_v(t, v, ni) / self._v_a3(v))

    def _fdisp2_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._fdisp2, t, v, ni, None, self._step_size['f_v'])

    def _fdisp2_i(self, t, v, ni):
        # de Villers, Equation F-78
        return -PI * NA * ((self._n(ni) ** 2.0) * self._c1(t, v, ni) * self._i2(t, v, ni) * self._m2eps2sig3(t, ni) *
                           self._m_i(ni) +
                           2.0 * self._n(ni) * self._c1(t, v, ni) * self._i2(t, v, ni) * self._m2eps2sig3(t, ni) *
                           self._m(ni) +
                           (self._n(ni) ** 2.0) * self._c1(t, v, ni) * self._i2(t, v, ni) *
                           self._m2eps2sig3_i(t, ni) * self._m(ni) +
                           (self._n(ni) ** 2.0) * self._c1(t, v, ni) * self._i2_i(t, v, ni) *
                           self._m2eps2sig3(t, ni) * self._m(ni) +
                           (self._n(ni) ** 2.0) * self._c1_i(t, v, ni) * self._i2(t, v, ni) *
                           self._m2eps2sig3(t, ni) * self._m(ni)) / \
               self._v_a3(v)

    def _fdisp_v(self, t, v, ni):
        # de Villers (2011), Equation F-94
        return self._fdisp1_v(t, v, ni) + self._fdisp2_v(t, v, ni)

    def _fdisp_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._fdisp, t, v, ni, None, self._step_size['f_v'])

    def _fdisp_i(self, t, v, ni):
        # de Villers (2011), Equation F-72
        return self._fdisp1_i(t, v, ni) + self._fdisp2_i(t, v, ni)

    def _f(self, t, v, ni, xai):
        # de Villers (2011), by definition
        return self._fhs(t, v, ni) + self._fhc(t, v, ni) + self._fdisp(t, v, ni)

    def _f_v(self, t, v, ni, xai):
        # de Villers (2011), by definition.
        return self._fhs_v(t, v, ni) + self._fhc_v(t, v, ni) + self._fdisp_v(t, v, ni)

    def _f_v_num(self, t, v, ni, xai=None):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._f, t, v, ni, xai, self._step_size['f_v'])

    def _f_i(self, t, v, ni, xai):
        # de Villers (2011), by definition.
        return self._fhs_i(t, v, ni) + self._fhc_i(t, v, ni) + self._fdisp_i(t, v, ni)

    def _f_i_num_vect(self, t, v, ni, xai=None):
        # Finite difference derivative implemented for testing.
        return self._func_i_grad_num(self._f, t, v, ni, xai, self._step_size['f_i'])

    def __eq__(self, other):
        if isinstance(other, sPCSAFT):
            phys_inter_eq = self.phys_inter == other.phys_inter
            assoc_inter_eq = self.assoc_inter == other.assoc_inter
            return phys_inter_eq and assoc_inter_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.phys_inter, self.assoc_inter))


# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
methane = Comp('Methane')
methane.formula = "CH4"
methane.family = "Alkanes"
methane.cas_no = "74-82-8"
methane.mw = 16.0425
methane.tc = 190.564
methane.pc = 4599000.0
methane.vc = 0.0000986
methane.acentric = 0.0115478

methane.spc_saft_phys = PCSAFTParms(comp=methane,
                                    pc_saft_spec=GS,
                                    source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                    seg_num=1.0, seg_diam=3.7039, disp_energy=150.03)

# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
ethane = Comp('Ethane')
ethane.formula = "C2H6"
ethane.family = "Alkanes"
ethane.cas_no = "74-84-0"
ethane.mw = 30.069
ethane.tc = 305.32
ethane.pc = 4872000.0
ethane.vc = 0.0001455
ethane.acentric = 0.099493

# Parameters from Ind. Eng. Chem. Res. 2001, 40, 1244-1260.
ethane.spc_saft_phys = PCSAFTParms(comp=ethane,
                                   pc_saft_spec=GS,
                                   source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                   seg_num=1.6069, seg_diam=3.5206, disp_energy=191.42)

# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
propane = Comp('Propane')
propane.formula = "C3H8"
propane.family = "Alkanes"
propane.cas_no = "74-98-6"
propane.mw = 44.09562
propane.tc = 369.83
propane.pc = 4248000.0
propane.vc = 0.0002
propane.acentric = 0.152291

propane.spc_saft_phys = PCSAFTParms(comp=propane,
                                    pc_saft_spec=GS,
                                    source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                    seg_num=2.002, seg_diam=3.6184, disp_energy=208.11)

# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
nitrogen = Comp('Nitrogen')
nitrogen.formula = "N2"
nitrogen.family = "Inorganics"
nitrogen.cas_no = "7727-37-9"
nitrogen.mw = 28.0134
nitrogen.tc = 126.2
nitrogen.pc = 3400000.0
nitrogen.vc = 0.00008921
nitrogen.acentric = 0.0377215

nitrogen.spc_saft_phys = PCSAFTParms(comp=nitrogen,
                                     pc_saft_spec=GS,
                                     source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                     seg_num=1.2053, seg_diam=3.3130, disp_energy=90.96)

# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
water = Comp('Water')
water.formula = "H2O"
water.family = "Inorganics"
water.cas_no = "7732-18-5"
water.mw = 18.01528
water.tc = 647.096
water.pc = 22064000.0
water.vc = 0.0000559472
water.acentric = 0.344861

water.pvap = Corel(a=73.649,
                   b=-7258.2,
                   c=-7.3037,
                   d=4.1653 * 10 ** -6,
                   e=2.0,
                   eq_id=1,
                   source_t_min=273.16,
                   source_t_max=647.1,
                   source_t_unit='K',
                   source_unit='Pa',
                   source= "Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data "
                           "Compilation of Pure Chemical Properties; Design Institute for Physical Properties, AIChE: "
                           "New York, NY, 2016.",
                   notes="DIPPR correlation parameters taken from Perry's Chemical Engineers' Handbook, 9th")

water.hvap = Corel(a=5.66 * 10 ** 7,
                   b=0.612041,
                   c=-0.625697,
                   d=0.398804,
                   e=0.0,
                   f=647.096,
                   eq_id=4,
                   source_t_min=273.16,
                   source_t_max=647.096,
                   source_t_unit='K',
                   source_unit='J/kmol',
                   source= "Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data "
                           "Compilation of Pure Chemical Properties; Design Institute for Physical Properties, AIChE: "
                           "New York, NY, 2016.",
                   notes="DIPPR correlation parameters taken from Perry's Chemical Engineers' Handbook, 9th")

water.cp_ig = Corel(a=0.33363 * 10 ** 5.0,
                    b=0.26790 * 10 ** 5.0,
                    c=2.61050 * 10 ** 3.0,
                    d=0.08896 * 10 ** 5.0,
                    e=1169.0,
                    eq_id=7,
                    source_t_min=100.0,
                    source_t_max=2273.15,
                    source_t_unit='K',
                    source_unit="J/kmol.K",
                    source="Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data "
                           "Compilation of Pure Chemical Properties; Design Institute for Physical Properties, AIChE: "
                           "New York, NY, 2016.",
                    notes="DIPPR correlation parameters taken from Perry's Chemical Engineers' Handbook, 9th")

water.spc_saft_phys = PCSAFTParms(comp=water,
                                  pc_saft_spec=GS,
                                  source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                                  seg_num=2.0, seg_diam=2.3449, disp_energy=171.67)

# Define association sites corresponding to water as a symmetrically associating 4C molecule.
ea1 = AssocSite(comp=water, site='H1', type='ea')
ea2 = AssocSite(comp=water, site='H2', type='ea')
ed1 = AssocSite(comp=water, site='O1', type='ed')
ed2 = AssocSite(comp=water, site='O2', type='ed')

water.assoc_sites = [ea1, ea2, ed1, ed2]

ea1_ed1 = AssocSiteInter(site_a=ea1, site_b=ed1,
                         eos='sPC-SAFT',
                         source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                         assoc_energy=1704.06, assoc_vol=0.3048)
ea1_ed2 = AssocSiteInter(site_a=ea1, site_b=ed2,
                         eos='sPC-SAFT',
                         source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                         assoc_energy=1704.06, assoc_vol=0.3048)
ea2_ed1 = AssocSiteInter(site_a=ea2, site_b=ed1,
                         eos='sPC-SAFT',
                         source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                         assoc_energy=1704.06, assoc_vol=0.3048)
ea2_ed2 = AssocSiteInter(site_a=ea2, site_b=ed2,
                         eos='sPC-SAFT',
                         source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                         assoc_energy=1704.06, assoc_vol=0.3048)

water.spc_saft_assoc = [ea1_ed1, ea1_ed2, ea2_ed1, ea2_ed2]


# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
methanol = Comp('Methanol')
methanol.formula = "CH3OH"
methanol.family = "Alcohols"
methanol.cas_no = "67-56-1"
methanol.mw = 32.04186
methanol.tc = 512.5
methanol.pc = 8084000.0
methanol.vc = 0.000117
methanol.acentric = 0.565831

methanol.spc_saft_phys = PCSAFTParms(comp=methanol,
                                     pc_saft_spec=GS,
                                     source="Ind. Eng. Chem. Res. 2012, 45, 14903–14914.",
                                     seg_num=1.88238, seg_diam=3.0023, disp_energy=181.77)

# Define association sites corresponding to methanol as a 2B molecule.
ea1 = AssocSite(comp=methanol, site='H1', type='ea')
ed1 = AssocSite(comp=methanol, site='O1', type='ed')

methanol.assoc_sites = [ea1, ed1]

ea1_ed1 = AssocSiteInter(site_a=ea1, site_b=ed1,
                         eos='sPC-SAFT',
                         source="Ind. Eng. Chem. Res. 2012, 45, 14903–14914.",
                         assoc_energy=2738.03, assoc_vol=0.054664)


mm = BinaryInterParm(comp_a=methane, comp_b=methane,
                     phys_eos_spec=GS, eos='sPC-SAFT',
                     source='pure component',
                     temp_indep_coef=0.0)

me = BinaryInterParm(comp_a=methane, comp_b=ethane,
                     phys_eos_spec=GS, eos='sPC-SAFT',
                     source='initial guess',
                     temp_indep_coef=0.03)

mp = BinaryInterParm(comp_a=methane, comp_b=propane,
                     phys_eos_spec=GS, eos='sPC-SAFT',
                     source='initial guess',
                     temp_indep_coef=0.03)

ee = BinaryInterParm(comp_a=ethane, comp_b=ethane,
                     phys_eos_spec=GS, eos='sPC-SAFT',
                     source='pure component',
                     temp_indep_coef=0.0)

ep = BinaryInterParm(comp_a=ethane, comp_b=propane,
                     phys_eos_spec=GS, eos='sPC-SAFT',
                     source='initial guess',
                     temp_indep_coef=0.01)

pp = BinaryInterParm(comp_a=propane, comp_b=propane,
                     phys_eos_spec=GS, eos='sPC-SAFT',
                     source='pure component',
                     temp_indep_coef=0.0)

wm = BinaryInterParm(comp_a=water, comp_b=methane,
                     phys_eos_spec=GS, eos='sPC-SAFT',
                     source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                     temp_indep_coef=0.2306,
                     inv_temp_coef=-92.62)

we = BinaryInterParm(comp_a=water, comp_b=ethane,
                     phys_eos_spec=GS, eos='sPC-SAFT',
                     source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                     temp_indep_coef=0.1773,
                     inv_temp_coef=-53.97)

wp = BinaryInterParm(comp_a=water, comp_b=propane,
                     phys_eos_spec=GS, eos='sPC-SAFT',
                     source='initial guess',
                     temp_indep_coef=0.05)

mem = BinaryInterParm(comp_a=methanol, comp_b=methane,
                      phys_eos_spec=GS, eos='sPC-SAFT',
                      source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                      temp_indep_coef=0.01)

mee = BinaryInterParm(comp_a=methanol, comp_b=ethane,
                      phys_eos_spec=GS, eos='sPC-SAFT',
                      source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                      temp_indep_coef=0.02)

mep = BinaryInterParm(comp_a=methanol, comp_b=propane,
                      phys_eos_spec=GS, eos='sPC-SAFT',
                      source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                      temp_indep_coef=0.02)

wme = BinaryInterParm(comp_a=water, comp_b=methanol,
                      phys_eos_spec=GS, eos='sPC-SAFT',
                      source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                      temp_indep_coef=-0.066)


cs = CompSet(comps=[ethane, propane])

spc_saft_phys = PCSAFTPhysInter(comps=cs, eos='sPC-SAFT', pc_saft_spec=GS)
spc_saft_phys.load_pure_comp_parms()
spc_saft_phys.load_binary_parms([ee, ep, pp])

print("---------------")
print("seg_num: {}".format(spc_saft_phys.seg_num))
print("seg_diam: {}".format(spc_saft_phys.seg_diam))
print("disp_energy: {}".format(spc_saft_phys.disp_energy))
print("k_ij: {}".format(spc_saft_phys.k_ij(298.15)))


spc_saft = sPCSAFT(phys_inter=spc_saft_phys, assoc_inter=None)

phase = Phase(comps=cs, eos=spc_saft)
phase.compos.set(xi=[0.2, 0.8])
phase.state.spec = 'PT'

phase.state.set(p=50000000.0, t=400.0)
p = phase.state.p
vm = phase.state.vm
t = phase.state.t
ni = phase.compos.xi
mw = phase.props.mw
i = 0
xai = None

# Checking the values for all functions that make up sPC-SAFT.  The most important check is that the resulting f, f_v,
# and f_i values equal their numeric derivatives. The
print("---------------")
print("p = {}".format(p))
print("t = {}".format(t))
print("vm = {}".format(vm))
print("mw = {}".format(phase.props.mw))
print("rho_m = {}".format(phase.props.rho_m))
print("ni = {}".format(ni))
print("rho = {}".format(phase.eos._rho(vm, ni)))
print("v_a3 = {}".format(phase.eos._v_a3(vm)))
print("m = {}".format(phase.eos._m(ni)))
print("m_i = {}".format(phase.eos._m_i(ni)))
print("di = {}".format(phase.eos._di(t)))
print("d3 = {}".format(phase.eos._d3(t, ni)))
print("d_i = {}".format(phase.eos._d_i(t, ni)))
print("eta = {}".format(phase.eos._eta(t, vm, ni)))
print("eta_v = {}".format(phase.eos._eta_v(t, vm, ni)))
print("eta_i = {}".format(phase.eos._eta_i(t, vm, ni)))
print("ghs = {}".format(phase.eos._ghs(t, vm, ni)))
print("ghs_v = {}".format(phase.eos._ghs_v(t, vm, ni)))
print("ghs_i = {}".format(phase.eos._ghs_i(t, vm, ni)))
print("i1 = {}".format(phase.eos._i1(t, vm, ni)))
print("i1_v = {}".format(phase.eos._i1_v(t, vm, ni)))
print("i1_i = {}".format(phase.eos._i1_i(t, vm, ni)))
print("i2 = {}".format(phase.eos._i2(t, vm, ni)))
print("i2_v = {}".format(phase.eos._i2_v(t, vm, ni)))
print("i2_i = {}".format(phase.eos._i2_i(t, vm, ni)))
print("eps_ij = {}".format(phase.eos._epsij(t)))
print("sig_ij = {}".format(phase.eos._sigij()))
print("ai_i = {}".format(phase.eos._ai_i(2, ni)))
print("bi_i = {}".format(phase.eos._bi_i(2, ni)))
print("m2eps1sig3 = {}".format(phase.eos._m2eps1sig3(t, ni)))
print("m2eps1sig3_i = {}".format(phase.eos._m2eps1sig3_i(t, ni)))
print("m2eps2sig3 = {}".format(phase.eos._m2eps2sig3(t, ni)))
print("m2eps2sig3_i = {}".format(phase.eos._m2eps2sig3_i(t, ni)))
print("c0 = {}".format(phase.eos._c0(t, vm, ni)))
print("c0_v = {}".format(phase.eos._c0_v(t, vm, ni)))
print("c0_i = {}".format(phase.eos._c0_i(t, vm, ni)))
print("c1 (as inverse of c0) = {}".format(1.0 / phase.eos._c0(t, vm, ni)))
print("c1 = {}".format(phase.eos._c1(t, vm, ni)))
print("c1_v = {}".format(phase.eos._c1_v(t, vm, ni)))
print("c1_i = {}".format(phase.eos._c1_i(t, vm, ni)))
print("fhs = {}".format(phase.eos._fhs(t, vm, ni)))
print("fhs_v = {}".format(phase.eos._fhs_v(t, vm, ni)))
print("fhs_i = {}".format(phase.eos._fhs_i(t, vm, ni)))
print("fhc = {}".format(phase.eos._fhc(t, vm, ni)))
print("fhc_v = {}".format(phase.eos._fhc_v(t, vm, ni)))
print("fhc_i = {}".format(phase.eos._fhc_i(t, vm, ni)))
print("fdisp = {}".format(phase.eos._fdisp(t, vm, ni)))
print("fdisp1 = {}".format(phase.eos._fdisp1(t, vm, ni)))
print("fdisp1_v = {}".format(phase.eos._fdisp1_v(t, vm, ni)))
print("fdisp1_i = {}".format(phase.eos._fdisp1_i(t, vm, ni)))
print("fdisp2 = {}".format(phase.eos._fdisp2(t, vm, ni)))
print("fdisp2_v = {}".format(phase.eos._fdisp2_v(t, vm, ni)))
print("fdisp2_i = {}".format(phase.eos._fdisp2_i(t, vm, ni)))
print("fdisp_v = {}".format(phase.eos._fdisp_v(t, vm, ni)))
print("fdisp_i = {}".format(phase.eos._fdisp_i(t, vm, ni)))
print("f = {}".format(phase.eos._f(t, vm, ni, xai)))
print("f_v = {}".format(phase.eos._f_v(t, vm, ni, xai)))
print("f_v_num = {}".format(phase.eos._f_v_num(t, vm, ni, xai)))
print("f_i = {}".format(phase.eos._f_i(t, vm, ni, xai)))
print("f_i_num = {}".format(phase.eos._f_i_num_vect(t, vm, ni, xai)))
print("ln_phi_j = {}".format(phase.eos._ln_phi_j(t, vm ,ni, xai)))

# Examples illustrating how the Corel class functions.
print("---------------")
print("Water vapor pressure correlation defined: {}".format(water.pvap.defined()))
print("Water vapor pressure at triple point: {}".format(water.pvap(water.pvap.t_min)))
print("Water vapor pressure at critical point: {}".format(water.pvap(water.pvap.t_max)))

print("Water ideal gas heat capacity correlation defined: {}".format(water.cp_ig.defined()))
print("Water ideal gas heat capacity at triple point: {}".format(water.cp_ig(water.cp_ig.t_min)))
print("Water ideal gas heat capacity at critical point: {}".format(water.cp_ig(water.cp_ig.t_max)))


print("Water heat of vaporization correlation defined: {}".format(water.hvap.defined()))
print("Water heat of vaporization at triple point: {}".format(water.hvap(water.hvap.t_min)))
print("Water heat of vaporization at critical point: {}".format(water.hvap(water.hvap.t_max)))