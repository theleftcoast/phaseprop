"""Objects representing pure chemical components and pseudo-components.

Attributes
----------
COMP_FAM : list
    Pre-defined component family pick list based on definitions in _[1] and _[2].

Notes
-----


References
----------
[1] Perry's Chemical Engineers' Handbook; Perry, R. H., Southard, M. Z., Eds.; McGraw-Hill Education: New York, 2019.
[2] Tihic, A.; Kontogeorgis, G. M.; von Solms, N.;Michelsen, M. L. Applications of the simplified perturbed-chain SAFT
equation of state using an extended parameter table. Fluid Phase Equilib. 2006, 248, 29-43.
"""

COMP_FAM = ['Alkanes', 'Alkenes', 'Alkynes', 'Cycloalkanes' 'Aromatics', 'Polynuclear Aromatics', 'Aldehydes',
            'Ketones', 'Heterocyclics', 'Elements', 'Alcohols', 'Phenols', 'Ethers', 'Acids', 'Esters', 'Amines',
            'Amides', 'Nitriles', 'Nitro Compounds', 'Isocyanates', 'Mercaptans', 'Sulfides',
            'Halogenated Hydrocarbons', 'Silanes', 'Inorganics', 'Multifunctional']

import numpy as np
from utilities import *


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
            self.hsub = None
            self.ig_hform = None
            self.ig_gform = None
            self.ig_entr = None
            self.hcomb = None

            # Temperature dependent properties.
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
            self.tcond_s = None
            self.tcond_l = None
            self.tcond_ig = None
            self.sigma = None

            # TODO: Ensure CEOS physical terms are built with getter/setter checks for CEOS objects.
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
            raise TypeError("hfus must be a float.")

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
    def ig_hform(self):
        """float : Ideal gas enthalpy of formation at 298.15, unit TBD.

        The compounds are considered to be formed from the elements in their standard states at 298.15K and 1 bar.
        These include  C (graphite) and S (rhombic).
        """
        return self._ig_hform

    @ig_hform.setter
    def ig_hform(self, value):
        if isinstance(value, float) or value is None:
            self._ig_hform = value
        else:
            raise TypeError("ig_hform must be a float.")

    @property
    def ig_gform(self):
        """float : Ideal gas gibbs energy of formation at 298.15, unit TBD."""
        return self._ig_gform

    @ig_gform.setter
    def ig_gform(self, value):
        if isinstance(value, float) or value is None:
            self._ig_gform = value
        else:
            raise TypeError("ig_gform must be a float.")

    @property
    def ig_entr(self):
        """float : Ideal gas gibbs entropy, unit TBD."""
        return self._ig_entr

    @ig_entr.setter
    def ig_entr(self, value):
        if isinstance(value, float) or value is None:
            self._ig_entr = value
        else:
            raise TypeError("ig_entr must be a float.")

    @property
    def hcomb(self):
        """float : Standard net enthalpy of combustion, unit TBD.

        Enthalpy of combustion is the net value for the compound in its standard state at 298.15K and 1 bar.  Products
        of combustion are taken to be CO2 (gas), H2O (gas), F2 (gas), Cl2 (gas), Br2 (gas), I2 (gas), SO2 (gas), N2
        (gas), P4O10 (crystalline), SiO2 (crystobalite), and Al2O3 (crystal, alpha).
        """
        return self._hcomb

    @hcomb.setter
    def hcomb(self, value):
        if isinstance(value, float) or value is None:
            self._hcomb = value
        else:
            raise TypeError("hcomb must be a float.")

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

        beta = -(1/V)*(dV/dP) = (1/rho)*(drho/dP), sign change due to conversion from V to rho with chain rule."""
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

        beta = -(1/V)*(dV/dP) = (1/rho)*(drho/dP), sign change due to conversion from V to rho with chain rule."""
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
    def tcond_s(self):
        """float : Solid thermal conductivity, unit TBD."""

    @tcond_s.setter
    def tcond_s(self, value):
        if isinstance(value, Corel) or value is None:
            self._tcond_s = value
        else:
            raise TypeError("tcond_s must be an instance of Corel.")

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

    def melting_curve(self, t=None):
        """Melting curve for a pure solid.

        Melting pressure can be estimated with the Clausius-Clapeyron equation. A very simple approach assumes that
        changes in the enthalpy of fusion, liquid molar volume, and solid molar volume are negligible over the
        temperature range of interest.  With these assumptions, the equation of the melting curve is as follows:

        dp/dt = delta_h / (t * delta_v)

        p - p0 = (delta_h / delta_v) * ln(t / t0)

        Variation of the enthalpy of fusion with temperature is related to differences in heat capacity between phases.

        ddelta_h/dt = dh_l/dt - dh_s/dt = cp_l - cp_s = detla_cp

        delta_h = integral_from_t0_to_t(delta_cp * dt) + delta_h0

        Temperature dependence of the heat capacity difference is usually weak and can be neglected.

        delta_h = delta_cp * (t - t0) + delta_h0

        Liquid and solid heat capacities can be evaluated at the melting temperature. These values are be derived from
        cp_l(t) and cp_s(t) correlations OR it can be entered as a constant (a useful adjustable parameter for
        correlating melting curves, sublimation curves, and solubility. The equation of the melting curve is as follows:

        p - p0 = (delta_cp / delta_v) * (t - t0) + ((delta_h - t0 * delta_cp) / delta_v) * ln(t / t0)

        Parameters
        ----------
        t : float
            Temperature specification, K.

        Returns
        -------
        float
            Melting pressure, Pa.

        References
        ----------
        [1] Tosun, I. The thermodynamics of phase and reaction equilibria, 1st ed., Elsevier: Amsterdam, 2013.
        [2] Poling, B. E.; Praunitz, J. M.; O'Connell, J. P. The properties of gases and liquids, 5th ed.,
        McGraw-Hill, 2000.
        [3] Pappa, G. D.; Voutsas, E. C.; Magoulas, K.; Tassios, D. P. Estimation of the differential molar heat
        capacities of organic compounds at their melting point. Ind. Eng. Chem. Res. 2005, 44, 3799–3806.
        """
        if isinstance(t, float) and t > 0.0:
            if self._hfus and self._mp and self._den_l and self._den_s:
                p0 = 101325.0
                t0 = self._mp
                tavg = (t + t0) / 2.0
                v_l = 1.0 / self._den_l(tavg)
                v_s = 1.0 / self._den_s(tavg)
                delta_v = v_l - v_s
                if self._cp_l and self._cp_s:
                    delta_cp = self._cp_l(tavg) - self._cp_s(tavg)
                    return p0 + (delta_cp/delta_v) * (t - t0) + ((self._hfus - t0 * delta_cp)/delta_v) * np.log(t/t0)
                else:
                    return p0 + (self._hfus/delta_v) * np.log(t/t0)
            else:
                raise RuntimeError("Enthalpy of fusion at melting point, liquid density, and solid density required.")
        else:
            raise RuntimeError("t must be a positive float.")

    def sublimation_curve(self):
        """Sublimation curve for a pure solid.

        Sublimation pressure can be estimated with the Clausius-Clapeyron equation. A very simple approach assumes that
        changes in the enthalpy of sublimation and solid molar volume are negligible over the temperature range of
        interest.  With these assumptions, the equation of the melting curve is as follows:

        dp/dt = delta_h / (t * delta_v)

        delta_h_sub = delta_h_vap + delta_h_fusion

        delta_v = v_v - v_s ~ v_v = R*t/p_sub

        p - p0 = (delta_h / delta_v) * ln(t / t0)

        Parameters
        ----------
        t : float
            Temperature specification, K.

        Returns
        -------
        float
            Sublimation pressure, Pa.

        References
        ----------
        [1] Tosun, I. The thermodynamics of phase and reaction equilibria, 1st ed., Elsevier: Amsterdam, 2013.
        [2] Poling, B. E.; Praunitz, J. M.; O'Connell, J. P. The properties of gases and liquids, 5th ed.,
        McGraw-Hill, 2000.
        [3] Pappa, G. D.; Voutsas, E. C.; Magoulas, K.; Tassios, D. P. Estimation of the differential molar heat
        capacities of organic compounds at their melting point. Ind. Eng. Chem. Res. 2005, 44, 3799–3806.
        """
        return

    def density(self, t=None, p=None, phase='l', spec='molar'):
        """Pure component liquid and/or solid density estimation.

        This function evaluates liquid or solid densities at pressures deviating from the vapor pressure curve or
        melting curve. The first step is calculating an initial density along the vapor pressure curve or melting curve.
        The next step is to correct for pressure with the isothermal compressibility:

        rho = rho_0 * exp(beta * (p - p_0))

        In this expression, rho_0 and p_0 are the saturation or melting density.  If pressure is not specified, then the
        saturated liquid or melting point solid density is returned. IF

        Parameters
        ----------
        p : float
            Pressure, Pa.
        t : float
            Temperature, K.
        spec : str
            Phase specification (either 'mass' or 'molar').

        Returns
        -------
        float or None
            Density of saturated liquid or melting point solid. None returned if 't' is outside correlation temp limits.
        float or None
            Density of solid phase at temperature 't' (None if no solid density correlation available).
        str
            Unit ('kg/m3' or 'mol/m3').
        """
        if not isinstance(p, (float, None)):
            raise TypeError("p must be None or a positive float.")
        elif not isinstance(t, float) and p > 0.0:
            raise TypeError("t must be a positive float.")
        elif phase not in ['l', 's']:
            raise ValueError("phase must be either 'l' or 's'.")
        elif spec not in ['mass', 'molar']:
            raise ValueError("spec must be either 'molar' or 'mass'.")
        else:
            if self._beta_l and p:
                p_corr_l = np.exp(self._beta_l(t) * ())
            else:
                beta_l = 0.0

            if self._beta_s:
                beta_s = self._beta_s(t)
            else:
                beta_s = 0.0

        if self._den_l and not self._den_s:
            # Liquid density correlation exists, solid density correlation does not.
            if spec == 'molar':
                if self._den_l.unit == 'kg/m3':
                    return self._den_l(t) * 1000.0 / self._mw
                elif self._den_l.unit == 'mol/m3':
                    return self._den_l(t)
                else:
                    raise RuntimeError("liquid density correlation does not have a valid unit.")
            elif spec == 'mass':
                return
            else:
                raise ValueError("spec is not valid.")
        elif not self._den_l and self._den_s:
            # Liquid density correlation does not exist, solid density correlation exists.
            if spec == 'molar':
                return
            elif spec == 'mass':
                return
            else:
                raise ValueError("spec is not valid.")
        elif self._den_l and self._den_s:
            # Liquid density and solid density correlations exist.
            if spec == 'molar':
                return
            elif spec == 'mass':
                return
            else:
                raise ValueError("spec is not valid.")
        else:
            raise RuntimeError("liquid and solid density correlations not loaded for Comp.")

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
    """A pseudo-component (polymer, distillation cut, asphaltene, etc.).

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
