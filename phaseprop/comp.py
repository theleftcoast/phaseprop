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
import textwrap as tw
import dataclasses
import typing
import utility


FAMILY = ['alkane',
          'alkene',
          'alkyne',
          'cycloalkane',
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
class sPCSAFTPhys(object):
    """Simplified PC-SAFT equation of state physical parameters for a single component.

    Parameters
    ----------
    seg_num : float
        Segment number.
    seg_diam : float
        Segment diameter.
    disp_energy : float
        Soave alpha function.
    ck_parm : float
        Chen and Kreglewski parameter (usually set to 0.12, except for hydrogen and helium).
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.

    Notes
    -----
    Chen and Kreglewski temperature-dependent integral parameter is 0.12 for nearly all compounds. However, it can be
    set to 0.241 for hydrogen (see eq 2-6 in _[1] and eq 2-10 in _[2]). This is a useful correction for quantum gases
    (hydrogen and helium).

    References
    ----------
    [1] de Villiers, A. J. Evaluation and improvement of the sPC-SAFT equation of state for complex mixtures. PhD
     thesis, Stellenbosch University, 2011.
    [2] Tihic, A.; Group contribution sPC-SAFT equation of state, Ph.D. Thesis, Denmark Technical University, 2008.
    """
    seg_num: float
    seg_diam: float
    disp_energy: float
    ck_parm: float = 0.12
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def __str__(self):
        return "m: {}, sigma: {}, epsilon: {}, ck: {}".format(self.seg_num,
                                                              self.seg_diam,
                                                              self.disp_energy,
                                                              self.ck_parm)


@dataclasses.dataclass
class sPCSAFTPolar(object):
    """Simplified PC-SAFT equation of state polar parameters for a single component.

    Parameters
    ----------
    polar_strength : float
        Polar strength of the compound.
    phantom_dipole : bool, default: false
        Cancels the interaction between phantom dipoles in the absence of permanent dipoles.
    source : str, optional
        Source of the parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the parameters.

    Notes
    -----
    The 'polar parameter' lumps several adjustable parameters into a single variable which simplifies implementation of
    Jog and Chapman's SAFT dipolar term. The 'phantom dipole' represents dipole induced dipole interactions in
    unsaturated hydrocarbons. If an unsaturated hydrocarbon is not in

    References
    ----------
    [1] Marshall, B. D.; Bokis, C. P. A PC-SAFT Model for Hydrocarbons I: Mapping Aromatic π-π Interactions onto a
    Dipolar Free Energy. Fluid Phase Equilibria 2019, 489, 83–89.
    [2] Marshall, B. D. A PC-SAFT Model for Hydrocarbons III: Phantom Dipole Representation of Dipole Induced Dipole
    Interactions in Unsaturated Hydrocarbons. Fluid Phase Equilibria 2019, 493, 153–159.
    """
    polar_strength: float
    phantom_dipole: bool = False
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def __str__(self):
        return "alpha: {}, phantom: {}".format(self.polar_strength,
                                               self.phantom_dipole)


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
    assoc_sites : list, optional
        List of AssocSite objects representing association sites.
    spc_saft_phys : spc_saft.sPCSAFTPhys, optional
        sPC-SAFT equation of state physical parameters.
    spc_saft_phys : spc_saft.sPCSAFTPolar, optional
        sPC-SAFT equation of state polar parameters.
    spc_saft_assoc : list, optional
        sPC-SAFT equation of state association interactions (list of AssocSiteInter objects).
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
    # cp_s: typing.Optional[float] = None
    cp_l: typing.Optional[typing.Union[utility.PolyCpL, utility.DIPPRCpL]] = None
    cp_ig: typing.Optional[typing.Union[utility.AlyLeeCpIg, utility.PolyCpIg]] = None
    visc_l: typing.Optional[utility.AndradeViscL] = None
    visc_ig: typing.Optional[utility.KineticViscIg] = None
    # tcond_s: typing.Optional[float] = None
    tcond_l: typing.Optional[utility.PolyTcondL] = None
    tcond_ig: typing.Optional[typing.Union[utility.KineticTcondIg, utility.PolyTcondIg]] = None
    surf_ten: typing.Optional[utility.SurfTen] = None

    # Association sites.
    assoc_sites: typing.Optional[list] = None

    # sPC-SAFT EOS.
    spc_saft_phys: typing.Optional[sPCSAFTPhys] = None
    spc_saft_polar: typing.Optional[sPCSAFTPolar] = None
    spc_saft_assoc: typing.Optional[list] = None


    def __post_init__(self):
        if self.family not in FAMILY and self.family is not None:
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

    def __str__(self):
        metadata = {'Name': self.name,
                    'CAS Registry Number': self.cas_no,
                    'Formula': self.formula,
                    'Family': self.family}
        constants = {'Molecular Weight': self.mw,
                     'Van der Waal Volume': self.vdwv,
                     'Van der Waal Area': self.vdwa,
                     'Radius of Gyration': self.rgyr,
                     'Dipole Moment': self.dipole,
                     'Quadrupole Moment': self.quadrupole,
                     'Acentric Factor': self.acentric,
                     'Critical Temperature': self.tc,
                     'Critical Pressure': self.pc,
                     'Critical Volume': self.vc,
                     'Critical Density': self.rhoc,
                     'Acentric Factor': self.acentric,
                     'Melting Point': self.mp,
                     'Enthalpy of Fusion': self.hfus,
                     'Ideal Gas Enthalpy of Formation': self.ig_hform,
                     'Ideal Gas Gibbs Energy of Formation': self.ig_gform,
                     'Ideal Gas Entropy': self.ig_entr,
                     'Standard Net Enthalpy of Combustion': self.hcomb}
        correlations = {'Vapor Pressure': (self.pvap_l, 'K', 'Pa'),
                        'Liquid Density': (self.den_l, 'K', 'mol/m3'),
                        'Heat of Vaporization': (self.hvap_l, 'K', 'J/mol'),
                        'Liquid Heat Capacity:': (self.cp_l, 'K', 'J/mol.K'),
                        'Ideal Gas Heat Capacity': (self.cp_ig, 'K', 'J/mol.K'),
                        'Vapor Viscosity': (self.visc_ig, 'K', 'Pa.s'),
                        'Liquid Viscosity': (self.visc_l, 'K', 'Pa.s'),
                        'Vapor Thermal Conductivity': (self.tcond_ig, 'K', 'W/m.K'),
                        'Liquid Thermal Conductivity': (self.tcond_l, 'K', 'W/m.K'),
                        'Surface Tension': (self.surf_ten, 'K', 'N/m')}
        spc_saft = {'sPC-SAFT Physical': self.spc_saft_phys,
                    'sPC-SAFT Polar': self.spc_saft_polar,
                    'sPC-SAFT Assoc': self.spc_saft_assoc}

        output = []
        for key, value in metadata.items():
            if value is not None:
                output.append("{}: {}\n".format(key, value))
        for key, value in constants.items():
            if value is not None:
                if isinstance(value, utility.Const):
                    output.append("{}: {} {}\n".format(key, value, value.unit))
                else:
                    output.append("{}: {}\n".format(key, value))
        for key, value in correlations.items():
            if value[0] is not None:
                output.append("{} Correlation \n".format(key))
                output.append("    Minimum Temperature: {} {}, Value: {} {}\n".format(value[0].t_min,
                                                                                      value[1],
                                                                                      value[0](value[0].t_min),
                                                                                      value[2]))
                output.append("    Maximum Temperature: {} {}, Value: {} {}\n".format(value[0].t_max,
                                                                                      value[1],
                                                                                      value[0](value[0].t_max),
                                                                                      value[2]))
        if self.assoc_sites is not None:
            output.append("Association Sites\n")
            for site in self.assoc_sites:
                output.append("    {}\n".format(str(site)))
        for key, value in spc_saft.items():
            if value is not None:
                output.append("{} \n".format(key))
                output.append("{}\n".format(tw.indent(str(value), "    ")))
        return "".join(output)


@dataclasses.dataclass
class PseudoComp(object):
    """A pseudo-component (polymer, distillation cut, asphaltene, etc.).

    Notes
    -----
    Future implementation of Riazi correlations to estimate Tc, Pc, Vc, Rhoc from mw, sg, NBP.
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
    comps: typing.Union[typing.List[Comp], typing.List[PseudoComp]]

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


@dataclasses.dataclass(eq=True, frozen=True)
class AssocSite(object):
    """Association site."""
    comp: Comp
    site: str
    type: str
    desc: typing.Optional[str] = None

    def __post_init__(self):
        if self.type not in ['ed', 'ea', 'glue', 'pi_stack']:
            raise ValueError("Site type not valid.")

    def __str__(self):
        return "Comp: {}, Site: {}, Type: {}".format(self.comp.name, self.site, self.type)

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


@dataclasses.dataclass
class AssocSiteInter(object):
    """Interaction between two association sites.

    Attributes
    ----------
    site_a : AssocSite
        AssocSite participating in the interaction.
    site_b : AssocSite
        AssocSite participating in the interaction.
    assoc_energy : float
        Association energy between sites.
    assoc_vol : float
        Association volume between sites.
    source : str, optional
        Source for the association interaction parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.
    """
    site_a: AssocSite
    site_b: AssocSite
    assoc_energy: float
    assoc_vol: float
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def __str__(self):
        return "'{}' interacting with '{}'\n" \
               "    Association Energy: {}\n" \
               "    Association Volume: {}".format(self.site_a.site,
                                                   self.site_b.type,
                                                   self.assoc_energy,
                                                   self.assoc_vol)

