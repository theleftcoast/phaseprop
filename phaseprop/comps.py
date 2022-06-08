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
import typing
import utility
import units
import refs
import spc_saft


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
    spc_saft_parms : assoc.sPCSAFTParms
        sPC-SAFT equation of state parameters.
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

    # sPC-SAFT EOS.
    spc_saft_parms: typing.Optional[spc_saft.sPCSAFTParms] = None

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
        if self.tc and self.pc and self.acentric:
            return (self.pc / p) * np.exp(5.37 * (1.0 + self.acentric) * (1.0 - self.tc / t))
        else:
            raise ValueError("tc, pc, and acentric must be defined.")

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
        # if self.assoc_groups:
        #     output.append("Association Groups\n")
        #     for group in self.assoc_groups:
        #         output.append("    {}\n".format(str(group)))
        # if self.spc_saft_phys is not None:
        #     output.append("sPC-SAFT Physical Parameters\n")
        #     output.append("    {}\n".format(str(self.spc_saft_phys)))
        # if self.spc_saft_polar is not None:
        #     output.append("sPC-SAFT Polar Parameters\n")
        #     output.append("    {}\n".format(str(self.spc_saft_polar)))
        # if self.spc_saft_assoc:
        #     output.append("sPC-SAFT Association Parameters\n")
        #     for i, interaction in enumerate(self.spc_saft_assoc):
        #         output.append('    Interaction {}\n'.format(i))
        #         output.append("{}\n".format(tw.indent(str(interaction), "        ")))
        # for key, value in spc_saft.items():
        #     if value is not None:
        #         output.append("{} \n".format(key))
        #         output.append("{}\n".format(tw.indent(str(value), "    ")))
        return "".join(output)


@dataclasses.dataclass
class PseudoComp(object):
    """A distillation cut or asphaltene.

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
    mw : list of float or None
        Molecular weight for each Comp or PseudoComp objects in 'comps' (returns None if any are missing 'mw').
    """
    comps: typing.Union[typing.List[Comp], typing.List[PseudoComp]]

    @property
    def size(self):
        """int : The number of Comp or PseudoComp objects in 'comps'."""
        return len(self.comps)

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


# Added by John Towne on April 1st, 2022, hcomb is suspiciously different than NIST
methane = Comp(name='methane',
                    cas_no='74-82-8',
                    formula='CH4',
                    family='alkane',
                    mw=utility.Const(value=16.0425,
                                     unit='g/mol',
                                     source=refs.dippr),
                    tc=utility.Const(value=190.564,
                                     unit='K',
                                     source=refs.dippr),
                    pc=utility.Const(value=units.to_si(4.599, 'MPa'),
                                     unit=units.to_si_unit('MPa'),
                                     source=refs.dippr),
                    vc=utility.Const(value=units.to_si(0.0986, 'm3/kmol'),
                                     unit=units.to_si_unit('m3/kmol'),
                                     source=refs.dippr),
                    acentric=utility.Const(value=0.0115478,
                                           unit='dimensionless',
                                           source=refs.dippr),
                    ig_hform=utility.Const(value=units.to_si(-7.452 * 10 ** 7.0, 'J/kmol'),
                                           unit=units.to_si_unit('J/kmol'),
                                           source=refs.dippr),
                    ig_gform=utility.Const(value=units.to_si(-5.049 * 10 ** 7.0, 'J/kmol'),
                                           unit=units.to_si_unit('J/kmol'),
                                           source=refs.dippr),
                    ig_entr=utility.Const(value=units.to_si(1.8627 * 10 ** 5.0, 'J/kmol.K'),
                                          unit=units.to_si_unit('J/kmol.K'),
                                          source=refs.dippr),
                    hcomb=utility.Const(value=units.to_si(-0.80262 * 10 ** 9.0, 'J/kmol'),
                                        # NIST shows different values.
                                        unit=units.to_si_unit('J/kmol'),
                                        source=refs.dippr),
                    mp=utility.Const(value=units.to_si(-182.48, 'C'),
                                     unit=units.to_si_unit('C'),
                                     source=refs.perry),
                    hfus=utility.Const(value=units.to_si(14.03 * 16.0425, 'cal/mol'),
                                       unit=units.to_si_unit('cal/mol'),
                                       source=refs.perry),
                    pvap_l=utility.RiedelPvap(a=39.205,
                                              b=-1324.4,
                                              c=-3.4366,
                                              d=0.000031019,
                                              e=2.0,
                                              unit='Pa',
                                              t_unit='K',
                                              t_min=90.69,
                                              t_max=190.56,
                                              source=refs.dippr),
                    den_l=utility.DaubertDenL(a=2.9214,
                                              b=0.28976,
                                              c=190.56,
                                              d=0.28881,
                                              unit='mol/dm3',
                                              t_unit='K',
                                              t_min=90.69,
                                              t_max=190.56,
                                              source=refs.dippr),
                    hvap_l=utility.PerryHvap(a=1.0194 * 10 ** 7.0,
                                             b=0.26087,
                                             c=-0.14694,
                                             d=0.22154,
                                             e=190.564,
                                             unit='J/kmol',
                                             t_unit='K',
                                             t_min=90.690,
                                             t_max=190.564,
                                             source=refs.dippr),
                    cp_l=utility.DIPPRCpL(a=65.708,
                                          b=38883.0,
                                          c=-257.95,
                                          d=614.07,
                                          e=190.564,
                                          unit='J/kmol.K',
                                          t_unit='K',
                                          t_min=90.69,
                                          t_max=190.00,
                                          source=refs.dippr),
                    cp_ig=utility.AlyLeeCpIg(a=0.33298 * 10 ** 5.0,
                                             b=0.79933 * 10 ** 5.0,
                                             c=2.0869 * 10 ** 3.0,
                                             d=0.41602 * 10 ** 5.0,
                                             e=991.96,
                                             unit='J/kmol.K',
                                             t_unit='K',
                                             t_min=50.0,
                                             t_max=1500.0,
                                             source=refs.dippr),
                    visc_ig=utility.KineticViscIg(a=5.2546 * 10 ** -7,
                                                  b=0.59006,
                                                  c=105.67,
                                                  unit='Pa.s',
                                                  t_unit='K',
                                                  t_min=90.69,
                                                  t_max=1000,
                                                  source=refs.dippr),
                    visc_l=utility.AndradeViscL(a=-6.1572,
                                                b=178.15,
                                                c=-0.95239,
                                                d=-9.0606 * 10 ** -24,
                                                e=10.0,
                                                unit='Pa.s',
                                                t_unit='K',
                                                t_min=90.69,
                                                t_max=188.0,
                                                source=refs.dippr),
                    tcond_ig=utility.KineticTcondIg(a=8.3983 * 10 ** -6,
                                                    b=1.4268,
                                                    c=-49.654,
                                                    unit='W/m.K',
                                                    t_unit='K',
                                                    t_min=111.63,
                                                    t_max=600.0,
                                                    source=refs.dippr),
                    tcond_l=utility.PolyTcondL(a=0.41768,
                                               b=-0.0024528,
                                               c=3.5588 * 10 ** -6,
                                               unit='W/m.K',
                                               t_unit='K',
                                               t_min=90.69,
                                               t_max=180.0,
                                               source=refs.dippr),
                    surf_ten=utility.SurfTen(a=37.432,
                                             b=190.56,
                                             c=1.092,
                                             unit='dyne/cm',
                                             t_unit='K',
                                             t_min=90.69,
                                             t_max=190.56,
                                             source=refs.yaws))

ethane = Comp(name='ethane',
                   cas_no='74-84-0',
                   formula='C2H6',
                   family='alkane',
                   mw=utility.Const(value=30.069,
                                    unit='g/mol',
                                    source=refs.dippr),
                   pvap_l=utility.RiedelPvap(a=51.857,
                                             b=-2598.7,
                                             c=-5.1283,
                                             d=0.000014913,
                                             e=2.0,
                                             unit='Pa',
                                             t_unit='K',
                                             t_min=90.35,
                                             t_max=305.32,
                                             source=refs.dippr),
                   den_l=utility.DaubertDenL(a=1.9122,
                                             b=0.27937,
                                             c=305.32,
                                             d=0.29187,
                                             unit='mol/dm3',
                                             t_unit='K',
                                             t_min=90.35,
                                             t_max=305.32,
                                             source=refs.dippr),
                   mp=utility.Const(value=units.to_si(-183.23, 'C'),
                                    unit=units.to_si_unit('C'),
                                    source=refs.perry),
                   hfus=utility.Const(value=units.to_si(22.712 * 30.069, 'cal/mol'),
                                      unit=units.to_si_unit('cal/mol'),
                                      source=refs.perry),
                   hvap_l=utility.PerryHvap(a=2.1091 * 10 ** 7.0,
                                            b=0.60646,
                                            c=-0.55492,
                                            d=0.32799,
                                            e=305.32,
                                            unit='J/kmol',
                                            t_unit='K',
                                            t_min=90.35,
                                            t_max=305.32,
                                            source=refs.dippr),
                   cp_l=utility.DIPPRCpL(a=44.009,
                                         b=89718.0,
                                         c=918.77,
                                         d=-1886.0,
                                         e=305.32,
                                         unit='J/kmol.K',
                                         t_unit='K',
                                         t_min=92.0,
                                         t_max=290.0,
                                         source=refs.dippr),
                   cp_ig=utility.AlyLeeCpIg(a=0.44256 * 10 ** 5,
                                            b=0.84737 * 10 ** 5,
                                            c=0.87224 * 10 ** 3,
                                            d=0.67130 * 10 ** 5,
                                            e=2430.0,
                                            unit='J/kmol.K',
                                            t_unit='K',
                                            t_min=298.15,
                                            t_max=1500.0),
                   ig_hform=utility.Const(value=units.to_si(-8.382 * 10 ** 7, 'J/kmol'),
                                          unit=units.to_si_unit('J/kmol'),
                                          source=refs.dippr),
                   ig_gform=utility.Const(value=units.to_si(-3.192 * 10 ** 7, 'J/kmol'),
                                          unit=units.to_si_unit('J/kmol'),
                                          source=refs.dippr),
                   ig_entr=utility.Const(value=units.to_si(2.2912 * 10 ** 5, 'J/kmol.K'),
                                         unit=units.to_si_unit('J/kmol.K'),
                                         source=refs.dippr),
                   hcomb=utility.Const(value=units.to_si(-1.42864 * 10 ** 9, 'J/kmol'),
                                       unit=units.to_si_unit('J/kmol'),
                                       source=refs.dippr),
                   tc=utility.Const(value=305.32,
                                    unit='K',
                                    source=refs.dippr),
                   pc=utility.Const(value=units.to_si(4.872, 'MPa'),
                                    unit=units.to_si_unit('MPa'),
                                    source=refs.dippr),
                   vc=utility.Const(value=units.to_si(0.1455, 'm3/kmol'),
                                    unit=units.to_si_unit('m3/kmol'),
                                    source=refs.dippr),
                   acentric=utility.Const(value=0.099493,
                                          unit='dimensionless',
                                          source=refs.dippr),
                   visc_ig=utility.KineticViscIg(a=2.5906 * 10 ** -7,
                                                 b=0.67988,
                                                 c=98.902,
                                                 unit='Pa.s',
                                                 t_unit='K',
                                                 t_min=90.35,
                                                 t_max=1000.0,
                                                 source=refs.dippr),
                   visc_l=utility.AndradeViscL(a=-7.0046,
                                               b=276.38,
                                               c=-0.6087,
                                               d=-3.11 * 10 ** -18,
                                               e=7.0,
                                               unit='Pa.s',
                                               t_unit='K',
                                               t_min=90.35,
                                               t_max=300.0,
                                               source=refs.dippr),
                   tcond_ig=utility.KineticTcondIg(a=0.000073869,
                                                   b=1.1689,
                                                   c=500.73,
                                                   unit='W/m.K',
                                                   t_unit='K',
                                                   t_min=184.55,
                                                   t_max=1000.0,
                                                   source=refs.dippr),
                   tcond_l=utility.PolyCpL(a=0.35758,
                                           b=-0.0011458,
                                           c=6.1866 * 10 ** -7,
                                           unit='W/m.K',
                                           t_unit='K',
                                           t_min=90.35,
                                           t_max=300.0,
                                           source=refs.dippr),
                   surf_ten=utility.SurfTen(a=49.63,
                                            b=305.32,
                                            c=1.2065,
                                            unit='dyne/cm',
                                            t_unit='K',
                                            t_min=90.37,
                                            t_max=305.32,
                                            source=refs.yaws))

propane = Comp(name='propane',
                    cas_no='74-98-6',
                    formula='C3H8',
                    family='alkane',
                    mw=utility.Const(value=44.09562,
                                     unit='g/mol',
                                     source=refs.dippr),
                    pvap_l=utility.RiedelPvap(a=59.078,
                                              b=-3492.6,
                                              c=-6.0669,
                                              d=0.000010919,
                                              e=2.0,
                                              unit='Pa',
                                              t_unit='K',
                                              t_min=85.47,
                                              t_max=369.83,
                                              source=refs.dippr),
                    den_l=utility.DaubertDenL(a=1.3757,
                                              b=0.27453,
                                              c=369.83,
                                              d=0.29359,
                                              unit='mol/dm3',
                                              t_unit='K',
                                              t_min=85.47,
                                              t_max=369.83,
                                              source=refs.dippr),
                    mp=utility.Const(value=units.to_si(-187.65, 'C'),
                                     unit=units.to_si_unit('C'),
                                     source=refs.perry),
                    hfus=utility.Const(value=units.to_si(19.1 * 44.09562, 'cal/mol'),
                                       unit=units.to_si_unit('cal/mol'),
                                       source=refs.perry),
                    hvap_l=utility.PerryHvap(a=2.9209 * 10 ** 7,
                                             b=0.78237,
                                             c=-0.77319,
                                             d=0.39246,
                                             e=369.83,
                                             unit='J/kmol',
                                             t_unit='K',
                                             t_min=85.47,
                                             t_max=369.83,
                                             source=refs.dippr),
                    cp_l=utility.DIPPRCpL(a=62.983,
                                          b=113630.0,
                                          c=633.21,
                                          d=-873.46,
                                          e=369.83,
                                          unit='J/kmol.K',
                                          t_unit='K',
                                          t_min=85.47,
                                          t_max=360.0,
                                          source=refs.dippr),
                    cp_ig=utility.AlyLeeCpIg(a=0.59474 * 10 ** 5,
                                             b=1.2661 * 10 ** 5,
                                             c=0.84431 * 10 ** 3,
                                             d=0.86165 * 10 ** 5,
                                             e=2482.8,
                                             unit='J/kmol.K',
                                             t_unit='K',
                                             t_min=298.15,
                                             t_max=1500,
                                             source=refs.dippr),
                    ig_hform=utility.Const(value=units.to_si(-10.468 * 10 ** 7, 'J/kmol'),
                                           unit=units.to_si_unit('J/kmol'),
                                           source=refs.dippr),
                    ig_gform=utility.Const(value=units.to_si(-2.439 * 10 ** 7, 'J/kmol'),
                                           unit=units.to_si_unit('J/kmol'),
                                           source=refs.dippr),
                    ig_entr=utility.Const(value=units.to_si(2.702 * 10 ** 5, 'J/kmol.K'),
                                          unit=units.to_si_unit('J/kmol.K'),
                                          source=refs.dippr),
                    hcomb=utility.Const(value=units.to_si(-2.04311 * 10 ** 9, 'J/kmol'),
                                        unit=units.to_si_unit('J/kmol'),
                                        source=refs.dippr),
                    tc=utility.Const(value=369.83,
                                     unit='K',
                                     source=refs.dippr),
                    pc=utility.Const(value=units.to_si(4.248, 'MPa'),
                                     unit=units.to_si_unit('MPa'),
                                     source=refs.dippr),
                    vc=utility.Const(value=units.to_si(0.2, 'm3/kmol'),
                                     unit=units.to_si_unit('m3/kmol'),
                                     source=refs.dippr),
                    acentric=utility.Const(value=0.152291,
                                           unit='dimensionless',
                                           source=refs.dippr),
                    visc_ig=utility.KineticViscIg(a=4.9054 * 10 ** -8,
                                                  b=0.90125,
                                                  unit='Pa.s',
                                                  t_unit='K',
                                                  t_min=85.47,
                                                  t_max=1000.0,
                                                  source=refs.dippr),
                    visc_l=utility.AndradeViscL(a=-17.156,
                                                b=646.25,
                                                c=1.1101,
                                                d=-7.3439 * 10 ** -11,
                                                e=4.0,
                                                unit='Pa.s',
                                                t_unit='K',
                                                t_min=85.47,
                                                t_max=360.0,
                                                source=refs.dippr),
                    tcond_ig=utility.KineticTcondIg(a=-1.12,
                                                    b=0.10972,
                                                    c=-9834.6,
                                                    d=-7535800.0,
                                                    unit='W/m.K',
                                                    t_unit='K',
                                                    t_min=231.11,
                                                    t_max=1000.0,
                                                    source=refs.dippr),
                    tcond_l=utility.PolyTcondL(a=0.26755,
                                               b=-0.00066457,
                                               c=2.77 * 10 ** -7,
                                               unit='W/m.K',
                                               t_unit='K',
                                               t_min=85.47,
                                               t_max=350.0,
                                               source=refs.dippr),
                    surf_ten=utility.SurfTen(a=49.179,
                                             b=369.83,
                                             c=1.22222,
                                             unit='dyne/cm',
                                             t_unit='K',
                                             t_min=85.53,
                                             t_max=369.83,
                                             source=refs.yaws))

n_butane = Comp(name='butane',
                     cas_no='106-97-8',
                     formula='C4H10',
                     family='alkane',
                     mw=utility.Const(value=58.1222,
                                      unit='g/mol',
                                      source=refs.dippr),
                     pvap_l=utility.RiedelPvap(a=66.343,
                                               b=-4363.2,
                                               c=-7.046,
                                               d=9.4509 * 10 ** -6,
                                               e=2.0,
                                               unit='Pa',
                                               t_unit='K',
                                               t_min=134.86,
                                               t_max=425.12,
                                               source=refs.dippr),
                     den_l=utility.DaubertDenL(a=1.0677,
                                               b=0.27188,
                                               c=425.12,
                                               d=0.28688,
                                               unit='mol/dm3',
                                               t_unit='K',
                                               t_min=134.86,
                                               t_max=425.12,
                                               source=refs.dippr),
                     mp=utility.Const(value=units.to_si(138.33, 'C'),
                                      unit=units.to_si_unit('C'),
                                      source=refs.perry),
                     hfus=utility.Const(value=units.to_si(19.167 * 58.1222, 'cal/mol'),
                                        unit=units.to_si_unit('cal/mol'),
                                        source=refs.perry),
                     hvap_l=utility.PerryHvap(a=3.6238 * 10 ** 7,
                                              b=0.8337,
                                              c=-0.82274,
                                              d=0.39613,
                                              e=425.12,
                                              unit='J/kmol',
                                              t_unit='K',
                                              t_min=134.86,
                                              t_max=425.12,
                                              source=refs.dippr),
                     cp_l=utility.PolyCpL(a=191030.0,
                                          b=-1675.0,
                                          c=12.5,
                                          d=-0.03874,
                                          e=0.000046121,
                                          unit='J/kmol.K',
                                          t_unit='K',
                                          t_min=134.86,
                                          t_max=400.0,
                                          source=refs.dippr),
                     cp_ig=utility.AlyLeeCpIg(a=0.80154 * 10 ** 5,
                                              b=1.6242 * 10 ** 5,
                                              c=0.84149 * 10 ** 3,
                                              d=1.0575 * 10 ** 5,
                                              e=2476.1,
                                              unit='J/kmol.K',
                                              t_unit='K',
                                              t_min=298.15,
                                              t_max=1500,
                                              source=refs.dippr),
                     ig_hform=utility.Const(value=units.to_si(-12.579 * 10 ** 7, 'J/kmol'),
                                            unit=units.to_si_unit('J/kmol'),
                                            source=refs.dippr),
                     ig_gform=utility.Const(value=units.to_si(-1.67 * 10 ** 7, 'J/kmol'),
                                            unit=units.to_si_unit('J/kmol'),
                                            source=refs.dippr),
                     ig_entr=utility.Const(value=units.to_si(3.0991 * 10 ** 5, 'J/kmol.K'),
                                           unit=units.to_si_unit('J/kmol.K'),
                                           source=refs.dippr),
                     hcomb=utility.Const(value=units.to_si(-2.65732 * 10 ** 9, 'J/kmol'),
                                         unit=units.to_si_unit('J/kmol'),
                                         source=refs.dippr),
                     tc=utility.Const(value=425.12,
                                      unit='K',
                                      source=refs.dippr),
                     pc=utility.Const(value=units.to_si(3.796, 'MPa'),
                                      unit=units.to_si_unit('MPa'),
                                      source=refs.dippr),
                     vc=utility.Const(value=units.to_si(0.255, 'm3/kmol'),
                                      unit=units.to_si_unit('m3/kmol'),
                                      source=refs.dippr),
                     acentric=utility.Const(value=0.200164,
                                            unit='dimensionless',
                                            source=refs.dippr))

i_butane = Comp(name='2-methylpropane',
                     cas_no='75-28-5',
                     formula='C4H10',
                     family='alkane',
                     mw=utility.Const(value=58.1222,
                                      unit='g/mol',
                                      source=refs.dippr),
                     pvap_l=utility.RiedelPvap(a=108.43,
                                               b=-5039.9,
                                               c=-15.012,
                                               d=0.022725,
                                               e=1.0,
                                               unit='Pa',
                                               t_unit='K',
                                               t_min=113.54,
                                               t_max=407.8,
                                               source=refs.dippr),
                     den_l=utility.DaubertDenL(a=1.0631,
                                               b=0.27506,
                                               c=407.8,
                                               d=0.2758,
                                               unit='mol/dm3',
                                               t_unit='K',
                                               t_min=113.54,
                                               t_max=407.8,
                                               source=refs.dippr),
                     tc=utility.Const(value=407.8,
                                      unit='K',
                                      source=refs.dippr),
                     pc=utility.Const(value=units.to_si(3.64, 'MPa'),
                                      unit=units.to_si_unit('MPa'),
                                      source=refs.dippr),
                     vc=utility.Const(value=units.to_si(0.259, 'm3/kmol'),
                                      unit=units.to_si_unit('m3/kmol'),
                                      source=refs.dippr),
                     acentric=utility.Const(value=0.183512,
                                            unit='dimensionless',
                                            source=refs.dippr))

n_pentane = Comp(name='pentane',
                      cas_no='109-66-0',
                      formula='C5H12',
                      family='alkane',
                      mw=utility.Const(value=72.14878,
                                       unit='g/mol',
                                       source=refs.dippr),
                      pvap_l=utility.RiedelPvap(a=78.741,
                                                b=-5420.3,
                                                c=-8.8253,
                                                d=9.6171 * 10 ** -6,
                                                e=2.0,
                                                unit='Pa',
                                                t_unit='K',
                                                t_min=143.42,
                                                t_max=469.7,
                                                source=refs.dippr),
                      den_l=utility.DaubertDenL(a=0.84947,
                                                b=0.26726,
                                                c=469.7,
                                                d=0.27789,
                                                unit='mol/dm3',
                                                t_unit='K',
                                                t_min=143.42,
                                                t_max=469.70,
                                                source=refs.dippr),
                      tc=utility.Const(value=469.7,
                                       unit='K',
                                       source=refs.dippr),
                      pc=utility.Const(value=units.to_si(3.37, 'MPa'),
                                       unit=units.to_si_unit('MPa'),
                                       source=refs.dippr),
                      vc=utility.Const(value=units.to_si(0.313, 'm3/kmol'),
                                       unit=units.to_si_unit('m3/kmol'),
                                       source=refs.dippr),
                      acentric=utility.Const(value=0.251506,
                                             unit='dimensionless',
                                             source=refs.dippr))

i_pentane = Comp(name='2-methylbutane',
                      cas_no='78-78-4',
                      formula='C5H12',
                      family='alkane',
                      mw=utility.Const(value=72.14878,
                                       unit='g/mol',
                                       source=refs.dippr),
                      pvap_l=utility.RiedelPvap(a=71.308,
                                                b=-4976.0,
                                                c=-7.7169,
                                                d=8.7271 * 10 ** -6,
                                                e=2.0,
                                                unit='Pa',
                                                t_unit='K',
                                                t_min=113.25,
                                                t_max=460.4,
                                                source=refs.dippr),
                      den_l=utility.DaubertDenL(a=0.91991,
                                                b=0.27815,
                                                c=460.4,
                                                d=0.28667,
                                                unit='mol/dm3',
                                                t_unit='K',
                                                t_min=113.25,
                                                t_max=460.4,
                                                source=refs.dippr),
                      tc=utility.Const(value=460.4,
                                       unit='K',
                                       source=refs.dippr),
                      pc=utility.Const(value=units.to_si(3.38, 'MPa'),
                                       unit=units.to_si_unit('MPa'),
                                       source=refs.dippr),
                      vc=utility.Const(value=units.to_si(0.306, 'm3/kmol'),
                                       unit=units.to_si_unit('m3/kmol'),
                                       source=refs.dippr),
                      acentric=utility.Const(value=0.227875,
                                             unit='dimensionless',
                                             source=refs.dippr))

n_hexane = Comp(name='hexane',
                     cas_no='110-54-3',
                     formula='C6H14',
                     family='alkane',
                     mw=utility.Const(value=86.17536,
                                      unit='g/mol',
                                      source=refs.dippr),
                     pvap_l=utility.RiedelPvap(a=104.65,
                                               b=-6995.5,
                                               c=12.702,
                                               d=0.000012381,
                                               e=2.0,
                                               unit='Pa',
                                               t_unit='K',
                                               t_min=177.83,
                                               t_max=507.6,
                                               source=refs.dippr),
                     den_l=utility.DaubertDenL(a=0.70824,
                                               b=0.26411,
                                               c=507.6,
                                               d=0.27537,
                                               unit='mol/dm3',
                                               t_unit='K',
                                               t_min=177.83,
                                               t_max=507.6,
                                               source=refs.dippr),
                     tc=utility.Const(value=507.6,
                                      unit='K',
                                      source=refs.dippr),
                     pc=utility.Const(value=units.to_si(3.025, 'MPa'),
                                      unit=units.to_si_unit('MPa'),
                                      source=refs.dippr),
                     vc=utility.Const(value=units.to_si(0.371, 'm3/kmol'),
                                      unit=units.to_si_unit('m3/kmol'),
                                      source=refs.dippr),
                     acentric=utility.Const(value=0.301261,
                                            unit='dimensionless',
                                            source=refs.dippr))

c_hexane = Comp(name='cyclohexane',
                     cas_no='110-82-7',
                     formula='C6H12',
                     family='cycloalkane',
                     mw=utility.Const(value=84.15948,
                                      unit='g/mol',
                                      source=refs.dippr),
                     pvap_l=utility.RiedelPvap(a=51.087,
                                               b=-5226.4,
                                               c=-4.2278,
                                               d=9.76 * 10 ** -18,
                                               e=6.0,
                                               unit='Pa',
                                               t_unit='K',
                                               t_min=279.6,
                                               t_max=553.8,
                                               source=refs.dippr),
                     den_l=utility.DaubertDenL(a=0.88998,
                                               b=0.27376,
                                               c=553.8,
                                               d=0.28571,
                                               unit='mol/dm3',
                                               t_unit='K',
                                               t_min=279.69,
                                               t_max=553.8,
                                               source=refs.dippr),
                     tc=utility.Const(value=553.8,
                                      unit='K',
                                      source=refs.dippr),
                     pc=utility.Const(value=units.to_si(4.08, 'MPa'),
                                      unit=units.to_si_unit('MPa'),
                                      source=refs.dippr),
                     vc=utility.Const(value=units.to_si(0.308, 'm3/kmol'),
                                      unit=units.to_si_unit('m3/kmol'),
                                      source=refs.dippr),
                     acentric=utility.Const(value=0.208054,
                                            unit='dimensionless',
                                            source=refs.dippr))

benzene = Comp(name='benzene',
                    cas_no='71-43-2',
                    formula='C6H6',
                    family='aromatic',
                    mw=utility.Const(value=78.11184,
                                     unit='g/mol',
                                     source=refs.dippr),
                    pvap_l=utility.RiedelPvap(a=83.107,
                                              b=-6486.2,
                                              c=-9.2194,
                                              d=6.9844 * 10 ** -6,
                                              e=2.0,
                                              unit='Pa',
                                              t_unit='K',
                                              t_min=278.68,
                                              t_max=562.05,
                                              source=refs.dippr),
                    den_l=utility.DaubertDenL(a=1.0259,
                                              b=0.26666,
                                              c=562.05,
                                              d=0.28394,
                                              unit='mol/dm3',
                                              t_unit='K',
                                              t_min=278.68,
                                              t_max=562.05,
                                              source=refs.dippr),
                    tc=utility.Const(value=562.05,
                                     unit='K',
                                     source=refs.dippr),
                    pc=utility.Const(value=units.to_si(4.895, 'MPa'),
                                     unit=units.to_si_unit('MPa'),
                                     source=refs.dippr),
                    vc=utility.Const(value=units.to_si(0.256, 'm3/kmol'),
                                     unit=units.to_si_unit('m3/kmol'),
                                     source=refs.dippr),
                    acentric=utility.Const(value=0.2103,
                                           unit='dimensionless',
                                           source=refs.dippr))

heptane = Comp(name='heptane',
                    cas_no='142-82-5',
                    formula='C7H16',
                    family='alkane',
                    mw=utility.Const(value=1234.5,
                                     unit='g/mol',
                                     source=refs.dippr),
                    pvap_l=utility.RiedelPvap(a=87.829,
                                              b=-6996.4,
                                              c=-9.8802,
                                              d=7.2099 * 10 ** -6,
                                              e=2.0,
                                              unit='Pa',
                                              t_unit='K',
                                              t_min=182.57,
                                              t_max=540.2,
                                              source=refs.dippr),
                    den_l=utility.DaubertDenL(a=0.61259,
                                              b=0.26211,
                                              c=540.2,
                                              d=0.28141,
                                              unit='mol/dm3',
                                              t_unit='K',
                                              t_min=182.57,
                                              t_max=540.2,
                                              source=refs.dippr),
                    tc=utility.Const(value=540.2,
                                     unit='K',
                                     source=refs.dippr),
                    pc=utility.Const(value=units.to_si(2.74, 'MPa'),
                                     unit=units.to_si_unit('MPa'),
                                     source=refs.dippr),
                    vc=utility.Const(value=units.to_si(0.428, 'm3/kmol'),
                                     unit=units.to_si_unit('m3/kmol'),
                                     source=refs.dippr),
                    acentric=utility.Const(value=0.349469,
                                           unit='dimensionless',
                                           source=refs.dippr))

mc_hexane = Comp(name='methylcyclohexane',
                      cas_no='108-87-2',
                      formula='C7H14',
                      family='alkane',
                      mw=utility.Const(value=98.18606,
                                       unit='g/mol',
                                       source=refs.dippr),
                      pvap_l=utility.RiedelPvap(a=92.684,
                                                b=-7080.8,
                                                c=-10.695,
                                                d=8.1366 * 10 ** -6,
                                                e=2.0,
                                                unit='Pa',
                                                t_unit='K',
                                                t_min=146.58,
                                                t_max=572.1,
                                                source=refs.dippr),
                      den_l=utility.DaubertDenL(a=0.73109,
                                                b=0.26971,
                                                c=572.1,
                                                d=0.29185,
                                                unit='mol/dm3',
                                                t_unit='K',
                                                t_min=146.58,
                                                t_max=572.1,
                                                source=refs.dippr),
                      tc=utility.Const(value=572.1,
                                       unit='K',
                                       source=refs.dippr),
                      pc=utility.Const(value=units.to_si(3.48, 'MPa'),
                                       unit=units.to_si_unit('MPa'),
                                       source=refs.dippr),
                      vc=utility.Const(value=units.to_si(0.369, 'm3/kmol'),
                                       unit=units.to_si_unit('m3/kmol'),
                                       source=refs.dippr),
                      acentric=utility.Const(value=0.236055,
                                             unit='dimensionless',
                                             source=refs.dippr))

toluene = Comp(name='toluene',
                    cas_no='108-88-3',
                    formula='C7H8',
                    family='aromatic',
                    mw=utility.Const(value=92.13842,
                                     unit='g/mol',
                                     source=refs.dippr),
                    pvap_l=utility.RiedelPvap(a=76.945,
                                              b=-6729.8,
                                              c=-8.179,
                                              d=5.3017 * 10 ** -6,
                                              e=2.0,
                                              unit='Pa',
                                              t_unit='K',
                                              t_min=178.18,
                                              t_max=591.75,
                                              source=refs.dippr),
                    den_l=utility.DaubertDenL(a=0.8792,
                                              b=0.27136,
                                              c=591.75,
                                              d=0.29241,
                                              unit='mol/dm3',
                                              t_unit='K',
                                              t_min=178.18,
                                              t_max=591.75,
                                              source=refs.dippr),
                    tc=utility.Const(value=591.75,
                                     unit='K',
                                     source=refs.dippr),
                    pc=utility.Const(value=units.to_si(4.108, 'MPa'),
                                     unit=units.to_si_unit('MPa'),
                                     source=refs.dippr),
                    vc=utility.Const(value=units.to_si(0.316, 'm3/kmol'),
                                     unit=units.to_si_unit('m3/kmol'),
                                     source=refs.dippr),
                    acentric=utility.Const(value=0.264012,
                                           unit='dimensionless',
                                           source=refs.dippr))

octane = Comp(name='octane',
                   cas_no='111-65-9',
                   formula='C8H18',
                   family='alkane',
                   mw=utility.Const(value=114.22852,
                                    unit='g/mol',
                                    source=refs.dippr),
                   pvap_l=utility.RiedelPvap(a=96.084,
                                             b=-7900.2,
                                             c=-11.003,
                                             d=7.1802*10**-6,
                                             e=2,
                                             unit='Pa',
                                             t_unit='K',
                                             t_min=216.38,
                                             t_max=568.7,
                                             source=refs.dippr),
                   den_l=utility.DaubertDenL(a=0.5266,
                                             b=0.25693,
                                             c=568.7,
                                             d=0.28571,
                                             unit='mol/dm3',
                                             t_unit='K',
                                             t_min=216.38,
                                             t_max=568.7,
                                             source=refs.dippr),
                   tc=utility.Const(value=568.7,
                                    unit='K',
                                    source=refs.dippr),
                   pc=utility.Const(value=units.to_si(2.49, 'MPa'),
                                    unit=units.to_si_unit('MPa'),
                                    source=refs.dippr),
                   vc=utility.Const(value=units.to_si(0.486, 'm3/kmol'),
                                    unit=units.to_si_unit('m3/kmol'),
                                    source=refs.dippr),
                   acentric=utility.Const(value=0.399552,
                                          unit='dimensionless',
                                          source=refs.dippr))

nonane = Comp(name='nonane',
                   cas_no='111-84-2',
                   formula='C9H20',
                   family='alkane',
                   mw=utility.Const(value=128.2551,
                                    unit='g/mol',
                                    source=refs.dippr),
                   pvap_l=utility.RiedelPvap(a=109.35,
                                             b=-9030.4,
                                             c=-12.882,
                                             d=7.8544*10**-6,
                                             e=2,
                                             unit='Pa',
                                             t_unit='K',
                                             t_min=219.66,
                                             t_max=594.6,
                                             source=refs.dippr),
                   den_l=utility.DaubertDenL(a=0.46321,
                                             b=0.25444,
                                             c=594.6,
                                             d=0.28571,
                                             unit='mol/dm3',
                                             t_unit='K',
                                             t_min=219.66,
                                             t_max=594.6,
                                             source=refs.dippr),
                   tc=utility.Const(value=594.6,
                                    unit='K',
                                    source=refs.dippr),
                   pc=utility.Const(value=units.to_si(2.29, 'MPa'),
                                    unit=units.to_si_unit('MPa'),
                                    source=refs.dippr),
                   vc=utility.Const(value=units.to_si(0.551, 'm3/kmol'),
                                    unit=units.to_si_unit('m3/kmol'),
                                    source=refs.dippr),
                   acentric=utility.Const(value=0.44346,
                                          unit='dimensionless',
                                          source=refs.dippr))

decane = Comp(name='decane',
                   cas_no='124-18-5',
                   formula='C10H22',
                   family='alkane',
                   mw=utility.Const(value=142.48168,
                                    unit='g/mol',
                                    source=refs.dippr),
                   pvap_l=utility.RiedelPvap(a=112.73,
                                             b=-9749.6,
                                             c=-13.245,
                                             d=7.1266*10**-6,
                                             e=2,
                                             unit='Pa',
                                             t_unit='K',
                                             t_min=243.51,
                                             t_max=617.7,
                                             source=refs.dippr),
                   den_l=utility.DaubertDenL(a=0.41084,
                                             b=0.25175,
                                             c=617.7,
                                             d=0.28571,
                                             unit='mol/dm3',
                                             t_unit='K',
                                             t_min=243.51,
                                             t_max=617.7,
                                             source=refs.dippr),
                   tc=utility.Const(value=617.7,
                                    unit='K',
                                    source=refs.dippr),
                   pc=utility.Const(value=units.to_si(2.11, 'MPa'),
                                    unit=units.to_si_unit('MPa'),
                                    source=refs.dippr),
                   vc=utility.Const(value=units.to_si(0.617, 'm3/kmol'),
                                    unit=units.to_si_unit('m3/kmol'),
                                    source=refs.dippr),
                   acentric=utility.Const(value=0.492328,
                                          unit='dimensionless',
                                          source=refs.dippr))

undecane = Comp(name='undecane',
                     cas_no='1120-21-4',
                     formula='C11H24',
                     family='alkane',
                     mw=utility.Const(value=156.30826,
                                      unit='g/mol',
                                      source=refs.dippr),
                     pvap_l=utility.RiedelPvap(a=131.0,
                                               b=-11143.0,
                                               c=-15.855,
                                               d=8.1871*10**-6,
                                               e=2.0,
                                               unit='Pa',
                                               t_unit='K',
                                               t_min=247.57,
                                               t_max=639.0,
                                               source=refs.dippr),
                     den_l=utility.DaubertDenL(a=0.36703,
                                               b=0.24876,
                                               c=639.0,
                                               d=0.28571,
                                               unit='mol/dm3',
                                               t_unit='K',
                                               t_min=247.57,
                                               t_max=639.0,
                                               source=refs.dippr),
                     tc=utility.Const(value=639.0,
                                      unit='K',
                                      source=refs.dippr),
                     pc=utility.Const(value=units.to_si(1.95, 'MPa'),
                                      unit=units.to_si_unit('MPa'),
                                      source=refs.dippr),
                     vc=utility.Const(value=units.to_si(0.685, 'm3/kmol'),
                                      unit=units.to_si_unit('m3/kmol'),
                                      source=refs.dippr),
                     acentric=utility.Const(value=0.530316,
                                            unit='dimensionless',
                                            source=refs.dippr))

dodecane = Comp(name='dodecane',
                     cas_no='112-40-3',
                     formula='C12H26',
                     family='alkane',
                     mw=utility.Const(value=170.33484,
                                      unit='g/mol',
                                      source=refs.dippr),
                     pvap_l=utility.RiedelPvap(a=137.47,
                                               b=-11976.0,
                                               c=-16.698,
                                               d=8.0906*10**-6,
                                               e=2.0,
                                               unit='Pa',
                                               t_unit='K',
                                               t_min=263.57,
                                               t_max=658.0,
                                               source=refs.dippr),
                     den_l=utility.DaubertDenL(a=0.33267,
                                               b=0.24664,
                                               c=658.0,
                                               d=0.28571,
                                               unit='mol/dm3',
                                               t_unit='K',
                                               t_min=263.57,
                                               t_max=658.0,
                                               source=refs.dippr),
                     tc=utility.Const(value=658.0,
                                      unit='K',
                                      source=refs.dippr),
                     pc=utility.Const(value=units.to_si(1.82, 'MPa'),
                                      unit=units.to_si_unit('MPa'),
                                      source=refs.dippr),
                     vc=utility.Const(value=units.to_si(0.755, 'm3/kmol'),
                                      unit=units.to_si_unit('m3/kmol'),
                                      source=refs.dippr),
                     acentric=utility.Const(value=0.576385,
                                            unit='dimensionless',
                                            source=refs.dippr))

nitrogen = Comp(name='nitrogen',
                     cas_no='7727-37-9',
                     formula='N2',
                     family='inorganic',
                     mw=utility.Const(value=28.0134,
                                      unit='g/mol',
                                      source=refs.dippr),
                     pvap_l=utility.RiedelPvap(a=58.282,
                                               b=-1084.1,
                                               c=-8.3144,
                                               d=0.044127,
                                               e=1.0,
                                               unit='Pa',
                                               t_unit='K',
                                               t_min=63.15,
                                               t_max=126.2,
                                               source=refs.dippr),
                     den_l=utility.DaubertDenL(a=3.2091,
                                               b=0.2861,
                                               c=126.2,
                                               d=0.2966,
                                               unit='mol/dm3',
                                               t_unit='K',
                                               t_min=63.15,
                                               t_max=126.2,
                                               source=refs.dippr),
                     tc=utility.Const(value=126.2,
                                      unit='K',
                                      source=refs.dippr),
                     pc=utility.Const(value=units.to_si(3.4, 'MPa'),
                                      unit=units.to_si_unit('MPa'),
                                      source=refs.dippr),
                     vc=utility.Const(value=units.to_si(0.08921, 'm3/kmol'),
                                      unit=units.to_si_unit('m3/kmol'),
                                      source=refs.dippr),
                     acentric=utility.Const(value=0.0377215,
                                            unit='dimensionless',
                                            source=refs.dippr))

oxygen = Comp(name='oxygen',
                   cas_no='7782-44-7',
                   formula='O2',
                   family='inorganic',
                   mw=utility.Const(value=31.9988,
                                    unit='g/mol',
                                    source=refs.dippr),
                   pvap_l=utility.RiedelPvap(a=51.245,
                                             b=-1200.2,
                                             c=-6.4361,
                                             d=0.028405,
                                             e=1.0,
                                             unit='Pa',
                                             t_unit='K',
                                             t_min=54.36,
                                             t_max=154.58,
                                             source=refs.dippr),
                   den_l=utility.DaubertDenL(a=3.9143,
                                             b=0.28772,
                                             c=154.58,
                                             d=0.2924,
                                             unit='mol/dm3',
                                             t_unit='K',
                                             t_min=54.36,
                                             t_max=154.58,
                                             source=refs.dippr),
                   tc=utility.Const(value=154.58,
                                    unit='K',
                                    source=refs.dippr),
                   pc=utility.Const(value=units.to_si(5.043, 'MPa'),
                                    unit=units.to_si_unit('MPa'),
                                    source=refs.dippr),
                   vc=utility.Const(value=units.to_si(0.0734, 'm3/kmol'),
                                    unit=units.to_si_unit('m3/kmol'),
                                    source=refs.dippr),
                   acentric=utility.Const(value=0.0221798,
                                          unit='dimensionless',
                                          source=refs.dippr))

argon = Comp(name='argon',
                  cas_no='7440-37-1',
                  formula='Ar',
                  family='inorganic',
                  mw=utility.Const(value=39.948,
                                   unit='g/mol',
                                   source=refs.dippr),
                  pvap_l=utility.RiedelPvap(a=42.127,
                                            b=-1093.1,
                                            c=-4.1425,
                                            d=0.000057254,
                                            e=2.0,
                                            unit='Pa',
                                            t_unit='K',
                                            t_min=83.78,
                                            t_max=150.86,
                                            source=refs.dippr),
                  den_l=utility.DaubertDenL(a=3.8469,
                                            b=0.2881,
                                            c=150.86,
                                            d=0.29783,
                                            unit='mol/dm3',
                                            t_unit='K',
                                            t_min=83.78,
                                            t_max=150.86,
                                            source=refs.dippr),
                  tc=utility.Const(value=150.86,
                                   unit='K',
                                   source=refs.dippr),
                  pc=utility.Const(value=units.to_si(4.898, 'MPa'),
                                   unit=units.to_si_unit('MPa'),
                                   source=refs.dippr),
                  vc=utility.Const(value=units.to_si(0.07459, 'm3/kmol'),
                                   unit=units.to_si_unit('m3/kmol'),
                                   source=refs.dippr),
                  acentric=utility.Const(value=0.0,
                                         unit='dimensionless',
                                         source=refs.dippr))

water = Comp(name='water',
                  cas_no='',
                  formula='H2O',
                  family='inorganic',
                  mw=utility.Const(value=18.01528,
                                   unit='g/mol',
                                   source=refs.dippr),
                  pvap_l=utility.RiedelPvap(a=73.649,
                                            b=-7258.2,
                                            c=-7.3037,
                                            d=4.1653*10**-6.0,
                                            e=2.0,
                                            unit='Pa',
                                            t_unit='K',
                                            t_min=273.16,
                                            t_max=647.1,
                                            source=refs.dippr),
                  den_l=utility.IAPWSDenL(),
                  tc=utility.Const(value=647.096,
                                   unit='K',
                                   source=refs.dippr),
                  pc=utility.Const(value=units.to_si(22.064, 'MPa'),
                                   unit=units.to_si_unit('MPa'),
                                   source=refs.dippr),
                  vc=utility.Const(value=units.to_si(0.0559472, 'm3/kmol'),
                                   unit=units.to_si_unit('m3/kmol'),
                                   source=refs.dippr),
                  acentric=utility.Const(value=0.344861,
                                         unit='dimensionless',
                                         source=refs.dippr))

methanol = Comp(name='methanol',
                     cas_no='67-56-1',
                     formula='CH3OH',
                     family='alcohol',
                     mw=utility.Const(value=32.04186,
                                      unit='g/mol',
                                      source=refs.dippr),
                     pvap_l=utility.RiedelPvap(a=82.718,
                                               b=-6904.5,
                                               c=-8.8622,
                                               d=7.4664*10**-6.0,
                                               e=2.0,
                                               unit='Pa',
                                               t_unit='K',
                                               t_min=175.47,
                                               t_max=512.5,
                                               source=refs.dippr),
                     den_l=utility.DaubertDenL(a=2.3267,
                                               b=0.27073,
                                               c=512.5,
                                               d=0.24713,
                                               unit='mol/dm3',
                                               t_unit='K',
                                               t_min=175.47,
                                               t_max=512.5,
                                               source=refs.dippr),
                     tc=utility.Const(value=512.5,
                                      unit='K',
                                      source=refs.dippr),
                     pc=utility.Const(value=units.to_si(8.084, 'MPa'),
                                      unit=units.to_si_unit('MPa'),
                                      source=refs.dippr),
                     vc=utility.Const(value=units.to_si(0.117, 'm3/kmol'),
                                      unit=units.to_si_unit('m3/kmol'),
                                      source=refs.dippr),
                     acentric=utility.Const(value=0.565831,
                                            unit='dimensionless',
                                            source=refs.dippr))

acetone = Comp(name='acetone',
                    cas_no='67-64-1',
                    formula='C3H6O',
                    family='ketone',
                    mw=utility.Const(value=58.07914,
                                     unit='g/mol',
                                     source=refs.dippr),
                    pvap_l=utility.RiedelPvap(a=69.006,
                                              b=-5599.6,
                                              c=-7.0985,
                                              d=6.2237*10**-6.0,
                                              e=2.0,
                                              unit='Pa',
                                              t_unit='K',
                                              t_min=178.45,
                                              t_max=508.2),
                    den_l=utility.DaubertDenL(a=1.2332,
                                              b=0.25886,
                                              c=508.2,
                                              d=0.2913,
                                              unit='mol/dm3',
                                              t_unit='K',
                                              t_min=178.45,
                                              t_max=508.2,
                                              source=refs.dippr),
                    tc=utility.Const(value=508.2,
                                     unit='K',
                                     source=refs.dippr),
                    pc=utility.Const(value=units.to_si(4.701, 'MPa'),
                                     unit=units.to_si_unit('MPa'),
                                     source=refs.dippr),
                    vc=utility.Const(value=units.to_si(0.209, 'm3/kmol'),
                                     unit=units.to_si_unit('m3/kmol'),
                                     source=refs.dippr),
                    acentric=utility.Const(value=0.306527,
                                           unit='dimensionless',
                                           source=refs.dippr))