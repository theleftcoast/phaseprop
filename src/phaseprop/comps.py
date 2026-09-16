"""Objects representing pure chemical components and pseudo-components."""

import numpy as np
import dataclasses
import typing
import src.phaseprop.utility as utility
import src.phaseprop.refs as refs
import src.phaseprop.units as units
import src.phaseprop.const as const


@dataclasses.dataclass(frozen=True, eq=True)
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
    zc : float
        Critical compressibility.
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
        Saturated liquid vapor pressure.
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
    """

    # Metadata and constants.
    name: str
    cas_no: typing.Optional[str] = dataclasses.field(default=None, repr=False)
    formula: typing.Optional[str] = dataclasses.field(default=None, repr=False)
    mw: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    vdwv: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    vdwa: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    rgyr: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    dipole: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    quadrupole: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    acentric: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    tc: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    pc: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    vc: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    rhoc: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    tt: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    pt: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    bp: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    mp: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    hfus: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    hsub: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    ig_hform: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    ig_gform: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    ig_entr: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )
    hcomb: typing.Optional[typing.Union[float, const.Const]] = dataclasses.field(
        default=None, repr=False
    )

    # Temperature dependent properties.
    pvap_l: typing.Optional[utility.RiedelPvap] = dataclasses.field(
        default=None, repr=False
    )
    hvap_l: typing.Optional[utility.PerryHvap] = dataclasses.field(
        default=None, repr=False
    )
    den_l: typing.Optional[typing.Union[utility.DaubertDenL, utility.IAPWSDenL]] = (
        dataclasses.field(default=None, repr=False)
    )
    cp_l: typing.Optional[typing.Union[utility.PolyCpL, utility.DIPPRCpL]] = (
        dataclasses.field(default=None, repr=False)
    )
    cp_ig: typing.Optional[typing.Union[utility.AlyLeeCpIg, utility.PolyCpIg]] = (
        dataclasses.field(default=None, repr=False)
    )
    visc_l: typing.Optional[utility.AndradeViscL] = dataclasses.field(
        default=None, repr=False
    )
    visc_ig: typing.Optional[utility.KineticViscIg] = dataclasses.field(
        default=None, repr=False
    )
    tcond_l: typing.Optional[utility.PolyTcondL] = dataclasses.field(
        default=None, repr=False
    )
    tcond_ig: typing.Optional[
        typing.Union[utility.KineticTcondIg, utility.PolyTcondIg]
    ] = dataclasses.field(default=None, repr=False)
    surf_ten: typing.Optional[utility.SurfTen] = dataclasses.field(
        default=None, repr=False
    )

    @property
    def zc(self) -> float:
        return self.pc * self.vc / (const.R * self.tc)

    def __str__(self):
        metadata = {
            "Name": self.name,
            "CAS Registry Number": self.cas_no,
            "Formula": self.formula,
        }
        constants = {
            "Molecular Weight": self.mw,
            "Van der Waal Volume": self.vdwv,
            "Van der Waal Area": self.vdwa,
            "Radius of Gyration": self.rgyr,
            "Dipole Moment": self.dipole,
            "Quadrupole Moment": self.quadrupole,
            "Critical Temperature": self.tc,
            "Critical Pressure": self.pc,
            "Critical Volume": self.vc,
            "Critical Density": self.rhoc,
            "Acentric Factor": self.acentric,
            "Melting Point": self.mp,
            "Enthalpy of Fusion": self.hfus,
            "Ideal Gas Enthalpy of Formation": self.ig_hform,
            "Ideal Gas Gibbs Energy of Formation": self.ig_gform,
            "Ideal Gas Entropy": self.ig_entr,
            "Standard Net Enthalpy of Combustion": self.hcomb,
        }
        correlations = {
            "Vapor Pressure": (self.pvap_l, "K", "Pa"),
            "Liquid Density": (self.den_l, "K", "mol/m3"),
            "Heat of Vaporization": (self.hvap_l, "K", "J/mol"),
            "Liquid Heat Capacity:": (self.cp_l, "K", "J/mol.K"),
            "Ideal Gas Heat Capacity": (self.cp_ig, "K", "J/mol.K"),
            "Vapor Viscosity": (self.visc_ig, "K", "Pa.s"),
            "Liquid Viscosity": (self.visc_l, "K", "Pa.s"),
            "Vapor Thermal Conductivity": (self.tcond_ig, "K", "W/m.K"),
            "Liquid Thermal Conductivity": (self.tcond_l, "K", "W/m.K"),
            "Surface Tension": (self.surf_ten, "K", "N/m"),
        }

        output = []
        for key, value in metadata.items():
            if value is not None:
                output.append("{}: {}\n".format(key, value))
        for key, value in constants.items():
            if value is not None:
                if isinstance(value, const.Const):
                    output.append("{}: {} {}\n".format(key, value, value.unit))
                else:
                    output.append("{}: {}\n".format(key, value))
        for key, value in correlations.items():
            if value[0] is not None:
                output.append("{} Correlation \n".format(key))
                output.append(
                    "    Minimum Temperature: {} {}, Value: {} {}\n".format(
                        value[0].t_min, value[1], value[0](value[0].t_min), value[2]
                    )
                )
                output.append(
                    "    Maximum Temperature: {} {}, Value: {} {}\n".format(
                        value[0].t_max, value[1], value[0](value[0].t_max), value[2]
                    )
                )
        return "".join(output)


@dataclasses.dataclass(frozen=True)
class CompSet(object):
    """Set of components or pseudo-components.

    Attributes
    ----------
    comps : list or tuple of comps.Comps
        List of components that are a part of the CompSet instance.
    mw : list of float or None
        Molecular weight for each Comp object in 'comps'.
    """

    comps: typing.List[Comp]

    @property
    def names(self):
        """list : Name for each Comp or PseudoComp object in 'comps'."""
        return [comp.name for comp in self.comps]

    @property
    def mw(self):
        """np.ndarray : Molecular weight for each Comp or PseudoComp objects in 'comps'."""
        result = [comp.mw for comp in self.comps]
        return np.array(result)

    @property
    def tc(self):
        """np.ndarray : Critical temperature for each Comp or PseudoComp objects in 'comps'."""
        result = [comp.tc for comp in self.comps]
        return np.array(result)

    @property
    def pc(self):
        """np.ndarray : Critical pressure for each Comp or PseudoComp objects in 'comps'."""
        result = [comp.pc for comp in self.comps]
        return np.array(result)

    @property
    def acentric(self):
        """np.ndarray : Acentric factor for each Comp or PseudoComp objects in 'comps'."""
        result = [comp.acentric for comp in self.comps]
        return np.array(result)

    def __len__(self):
        return len(self.comps)

    def __eq__(self, other):
        if isinstance(other, CompSet) and set(self.comps) == set(other.comps):
            return True
        if isinstance(other, Comp) and set(self.comps) == set([other]):
            return True
        return False

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if isinstance(other, CompSet) and set(self.comps).issubset(set(other.comps)):
            return True
        if isinstance(other, Comp) and set(self.comps).issubset(set([other])):
            return True
        return False

    def __gt__(self, other):
        if isinstance(other, CompSet) and set(other.comps).issubset(set(self.comps)):
            return True
        if isinstance(other, Comp) and set([other]).issubset(set(self.comps)):
            return True
        return False

    def __le__(self, other):
        if self < other or self == other:
            return True
        return False

    def __ge__(self, other):
        if self > other or self == other:
            return True
        return False

    def __str__(self):
        output = []
        for comp in self.comps:
            output.append("{}".format(comp.name))
        return ", ".join(output)
