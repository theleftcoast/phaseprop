"""Pure component parameters."""

import refs
from comp import Comp
from utility import Corel
from spc_saft import GS, sPCSAFTParms

# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
methane = Comp('Methane')
methane.formula = "CH4"
methane.family = "alkane"
methane.cas_no = "74-82-8"
methane.mw = 16.0425
methane.tc = 190.564
methane.pc = 4599000.0
methane.vc = 0.0000986
methane.acentric = 0.0115478

methane.spc_saft_phys = sPCSAFTParms(comp=methane,
                                    spc_saft_spec=GS,
                                    source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                    seg_num=1.0, seg_diam=3.7039, disp_energy=150.03)

# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
ethane = Comp('Ethane')
ethane.formula = "C2H6"
ethane.family = "alkane"
ethane.cas_no = "74-84-0"
ethane.mw = 30.069
ethane.tc = 305.32
ethane.pc = 4872000.0
ethane.vc = 0.0001455
ethane.acentric = 0.099493

# Parameters from Ind. Eng. Chem. Res. 2001, 40, 1244-1260.
ethane.spc_saft_phys = sPCSAFTParms(comp=ethane,
                                   spc_saft_spec=GS,
                                   source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                   seg_num=1.6069, seg_diam=3.5206, disp_energy=191.42)

# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
propane = Comp('Propane')
propane.formula = "C3H8"
propane.family = "alkane"
propane.cas_no = "74-98-6"
propane.mw = 44.09562
propane.tc = 369.83
propane.pc = 4248000.0
propane.vc = 0.0002
propane.acentric = 0.152291

propane.spc_saft_phys = sPCSAFTParms(comp=propane,
                                    spc_saft_spec=GS,
                                    source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                    seg_num=2.002, seg_diam=3.6184, disp_energy=208.11)

# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
nitrogen = Comp('Nitrogen')
nitrogen.formula = "N2"
nitrogen.family = "inorganic"
nitrogen.cas_no = "7727-37-9"
nitrogen.mw = 28.0134
nitrogen.tc = 126.2
nitrogen.pc = 3400000.0
nitrogen.vc = 0.00008921
nitrogen.acentric = 0.0377215

nitrogen.spc_saft_phys = sPCSAFTParms(comp=nitrogen,
                                     spc_saft_spec=GS,
                                     source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                     seg_num=1.2053, seg_diam=3.3130, disp_energy=90.96)

# TODO: Break out constants for compounds into their own module.  Put all components into one Comps module.
# TODO: Break out binary interactions into one big module for all binary interactions.
# Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
water = Comp('Water')
water.formula = "H2O"
water.family = "inorganic"
water.cas_no = "7732-18-5"
# TODO: mw: typing.Optional[float]
# TODO: mw: Optional[float] = None
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
                   source=refs.dippr,
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
                   source=refs.dippr,
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
                    source=refs.dippr,
                    notes="DIPPR correlation parameters taken from Perry's Chemical Engineers' Handbook, 9th")

water.spc_saft_phys = sPCSAFTParms(comp=water,
                                  spc_saft_spec=GS,
                                  source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                                  seg_num=2.0, seg_diam=2.3449, disp_energy=171.67)
