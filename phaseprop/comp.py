"""Objects representing pure chemical components and pseudo-components.

Attributes
----------
FAMILY : list
    Pre-defined component family pick list based on definitions in _[1] and _[2].


References
----------
[1] Perry's Chemical Engineers' Handbook; Perry, R. H., Southard, M. Z., Eds.; McGraw-Hill Education: New York, 2019.
[2] Tihic, A.; Kontogeorgis, G. M.; von Solms, N.;Michelsen, M. L. Applications of the simplified perturbed-chain SAFT
equation of state using an extended parameter table. Fluid Phase Equilib. 2006, 248, 29-43.
"""
import numpy as np
import dataclasses
import utility
import typing
import assoc
import cubic
import spc_saft

FAMILY = ['alkane',
          'alkene',
          'alkyne',
          'cycloalkane' 
          'aromatic',
          'polynuclear aromatic',
          'aldehyde',
          'ketone',
          'heterocyclic',
          'element',
          'alcohol',
          'phenol',
          'ether',
          'acid',
          'ester',
          'amine',
          'amide',
          'nitrile',
          'nitro compound',
          'isocyanate',
          'mercaptan',
          'sulfide',
          'halogenated hydrocarbon',
          'silane',
          'inorganic',
          'multifunctional']


@dataclasses.dataclass
class Comp(object):
    """A pure chemical component.

    Attributes
    ----------
    name : str
        Name of chemical compound.
    cas_no : str, optional
        Chemical Abstracts Service Registry Number.
    formula : str, optional
        Chemical formula.
    family : str, optional
        Chemical family.
    mw : float or utility.Const, optional
        Molecular weight.
    vdwv : float or utility.Const, optional
        Van der Waal's volume.
    vdwa : float or utility.Const, optional
        Van der Waal's volume surface area.
    rgyr : float or utility.Const, optional
        Radius of gyration.
    dipole : float or utility.Const, optional
        Gas phase dipole moment.
    quadrupole : float or utility.Const, optional
        Gas phase quadrupole moment.
    acentric : float or utility.Const, optional
        Acentric factor.
    tc : float or utility.Const, optional
        Critical temperature.
    pc : float or utility.Const, optional
        Critical pressure.
    vc : float or utility.Const, optional
        Critical volume.
    rhoc : float or utility.Const, optional
        Critical density.
    tt : float or utility.Const, optional
        Triple point temperature.
    pt : float or utility.Const, optional
        Triple point pressure.
    bp : float or utility.Const, optional
        Boiling point.
    mp : float or utility.Const, optional
        Melting point.
    hfus : float or utility.Const, optional
        Enthalpy of fusion.
    hsub : float or utility.Const, optional
        Enthalpy of sublimation.
    ig_hform : float or utility.Const, optional
        Ideal gas enthalpy of formation.
    ig_gform : float or utility.Const, optional
        Ideal gas Gibbs energy of formation.
    ig_entr : float or utility.Const, optional
        Ideal gas entropy.
    hcomb : float or utility.Const, optional
        Enthalpy of combustion.
    pvap_l : utility.ReidelPvap, optional
        Satureated liquid vapor pressure.
    hvap_l : utility.PerryHvap, optional
        Enthalpy of vaporization.
    den_l : utility.DaubertDenL or utility.IAPWSDenL, optional
        Saturated liquid density.
    cp_l : utility.PolyCpL or utility.DIPPRCpL, optional
        Saturated liquid heat capacity.
    cp_ig : utility.AlyLeeCpIg or utility.PolyCpIg, optional
        Ideal gas heat capacity.
    visc_l : utility.AndradeViscL, optional
        Saturated liquid viscosity.
    visc_ig : utility.KineticViscIg, optional
        Ideal gas viscosity.
    tcond_l : utility.PolyTcondL, optional
        Saturated liquid thermal conductivity.
    tcond_ig : utility.KineticTcondIg or utility.PolyTcondIg, optional
        Ideal gas thermal conductivity.
    surf_ten : utility.SurfTen, optional
        Surface tension.
    srk_phys : cubic.SRKParms, optional
        SRK equation of state parameters.
    cpa_phys : cubic.CPAParms, optional
        CPA equation of state parameters.
    pr_phys : cubic.PRParms, optional
        CPA equation of state parameters.
    gpr_phys : cubic.GPRParms, optional
        GPR equation of state parameters.
    tpr_phys : cubic.TPRParms, optional
        TPR equation of state parameters.
    spc_saft_phys : spc_saft.sPCSAFTParms, optional
        sPC-SAFT equation of state parameters.
    assoc_sites : list, optional

    cpa_assoc : list, optional

    spc_saft_assoc : list

    """
    # Metadata and constants.
    name: str
    cas_no: typing.Optional[str] = None
    formula: typing.Optional[str] = None
    family: typing.Optional[str] = None
    mw: typing.Optional[typing.Union[float, utility.Const]] = None
    vdwv: typing.Optional[typing.Union[float, utility.Const]] = None
    vdwa: typing.Optional[typing.Union[float, utility.Const]] = None
    rgyr: typing.Optional[typing.Union[float, utility.Const]] = None
    dipole: typing.Optional[typing.Union[float, utility.Const]] = None
    quadrupole: typing.Optional[typing.Union[float, utility.Const]] = None
    acentric: typing.Optional[typing.Union[float, utility.Const]] = None
    tc: typing.Optional[typing.Union[float, utility.Const]] = None
    pc: typing.Optional[typing.Union[float, utility.Const]] = None
    vc: typing.Optional[typing.Union[float, utility.Const]] = None
    rhoc: typing.Optional[typing.Union[float, utility.Const]] = None
    tt: typing.Optional[typing.Union[float, utility.Const]] = None
    pt: typing.Optional[typing.Union[float, utility.Const]] = None
    bp: typing.Optional[typing.Union[float, utility.Const]] = None
    mp: typing.Optional[typing.Union[float, utility.Const]] = None
    hfus: typing.Optional[typing.Union[float, utility.Const]] = None
    hsub: typing.Optional[typing.Union[float, utility.Const]] = None
    ig_hform: typing.Optional[typing.Union[float, utility.Const]] = None
    ig_gform: typing.Optional[typing.Union[float, utility.Const]] = None
    ig_entr: typing.Optional[typing.Union[float, utility.Const]] = None
    hcomb: typing.Optional[typing.Union[float, utility.Const]] = None

    # Temperature dependent properties.
    pvap_l: typing.Optional[utility.RiedelPvap] = None
    hvap_l: typing.Optional[utility.PerryHvap] = None
    # den_s: typing.Optional[float] = None
    den_l: typing.Optional[typing.Union[utility.DaubertDenL, utility.IAPWSDenL]] = None
    # beta_s: typing.Optional[float] = None
    # Note: beta = -(1/V)*(dV/dP) = (1/rho)*(drho/dP), sign change due to conversion from V to rho with chain rule.
    # beta_l: typing.Optional[float] = None
    # cp_s: typing.Optional[float] = None
    cp_l: typing.Optional[typing.Union[utility.PolyCpL, utility.DIPPRCpL]] = None
    cp_ig: typing.Optional[utility.AlyLeeCpIg, utility.PolyCpIg] = None
    visc_l: typing.Optional[utility.AndradeViscL] = None
    visc_ig: typing.Optional[utility.KineticViscIg] = None
    # tcond_s: typing.Optional[float] = None
    tcond_l: typing.Optional[utility.PolyTcondL] = None
    tcond_ig: typing.Optional[utility.KineticTcondIg, utility.PolyTcondIg] = None
    surf_ten: typing.Optional[utility.SurfTen] = None

    # Cubic EOS physical parameter dictionaries.
    srk_phys = typing.Optional[cubic.SRKParms] = None
    cpa_phys = typing.Optional[cubic.CPAParms] = None
    pr_phys = typing.Optional[cubic.PRParms] = None
    gpr_phys = typing.Optional[cubic.GPRParms] = None
    tpr_phys = typing.Optional[cubic.TPRParms] = None

    # SAFT EOS physical parameter dictionaries.
    spc_saft_phys = typing.Optional[spc_saft.sPCSAFTParms] = None

    # SAFT EOS association parameter objects.
    assoc_sites = typing.Optional[list] = None
    cpa_assoc = typing.Optional[list] = None
    spc_saft_assoc = typing.Optional[list] = None

    def __post_init__(self):
        if self.family not in FAMILY:
            raise ValueError("family not valid.")

    def k_wilson(self, p: float, t: float) -> float:
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
        return (self._pc / p) * np.exp(5.37 * (1.0 + self._acentric) * (1.0 - self._tc / t))

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
        return

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
        saturated liquid or melting point solid density is returned.

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
        return

    def _check_assoc_parameters(self):
        # Check to ensure association parameter are lists of AssocSiteInter objects.
        if self.assoc_sites is not None:
            if self.cpa_assoc is not None:
                for asi in self.cpa_assoc:
                    if not isinstance(asi, assoc.AssocSiteInter):
                        raise TypeError("Association parameter lists must contain AssocSiteInter objects.")


class PseudoComp(object):
    """A pseudo-component (polymer, distillation cut, asphaltene, etc.).

    Notes
    -----
    TODO: Improve agreement bewteeen Comp and PseudoComp classes. Currently a skeleton implementation.
    TODO: Implement Riazi correlations to estimate Tc, Pc, Vc, Rhoc from mw, sg, NBP.
    """


@dataclasses.dataclass
class CompSet(object):
    """Set of components or pseudo-components.

    Attributes
    ----------
    comps : list of Comps or list of PseudoComp
        List of components that are a part of the CompSet instance.
    size : int
        The number of Comp or PseudoComp objects in 'comps'.
    can_associate : list of bool
        Indicator of if Comp or PseudoComp objects in 'comps' can associate.
    mw : list of float or None
        Molecular weight for each Comp or PseudoComp objects in 'comps' (returns None if any are missing 'mw').
    """
    comps: typing.Union[list[Comp], list[PseudoComp]]

    @property
    def size(self):
        """int : The number of Comp or PseudoComp objects in 'comps'."""
        return len(self.comps)

    @property
    def can_associate(self):
        """list of bool : Boolean indicating if Comp or PseudoComp objects in 'comps' can associate."""
        result = []
        for comp in self.comps:
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
        for comp in self.comps:
            if comp.mw is None:
                return None
            else:
                result.append(comp.mw)
        return np.array(result)
