"""Unit conversion.

Attributes
----------
MOLECULAR_WEIGHT : dict
    Keys are units for molecular weight and values are the conversion factor for that unit into Daltons (or 'g/mol').
AMOUNT : dict
    Keys are units for amount of substance and values are the conversion factor for that unit into moles.
MASS : dict
    Keys are units for mass and values are the conversion factor for that unit into kilograms.
LENGTH : dict
    Keys are units for length and values are the conversion factor for that unit into meters).
AREA : dict
    Keys are units for area and the values are the conversion factor for that unit into square meters.
VOLUME : dict
    Keys are units for volume and values are the conversion factor for that unit into cubic meters.
TEMPERATURE : dict
    Keys are units for temperature and values are lambda functions (func(x) -> t) which convert that unit into Kelvin.
FORCE : dict
    Keys are units for force and values are the conversion factor for that unit into Newtons.
PRESSURE : dict
    Keys are units for pressure and values are the conversion factor for that unit into Pascals.
DENSITY : dict
    Keys are units for density and values are the conversion factor for that unit into kilograms per cubic meter.
MOLAR_DENSITY : dict
    Keys are units for molar density and values are the conversion factor for that unit into moles per cubic meter.
MOLAR_VOLUME : dict
    Keys are units for molar volume and values are the conversion factor for that unit into cubic meter per mole.
ENERGY : dict
    Keys are units for energy and values are the conversion factor for that unit into Joules.
HEAT_OF_VAPORIZATION : dict
    Keys are units for heat of vaporization and values are the conversion factor for that unit into Joules per mole
HEAT_CAPACITY : dict
    Keys are units for heat capacity and values are the conversion factor for that unit into Joules per mole-kelvin.
VISCOSITY : dict
    Keys are units for viscosity and values are the conversion factor for that unit into Pascal-second.
THERMAL_CONDUCTIVITY : dict
    Keys are units for thermal conductivity and values are the conversion factor for that unit into Watts per
    meter-Kelvin.
SURFACE_TENSION : dict
    Keys are units for surface tension and values are the conversion factor for that unit into Newtons per meter.
DIMENSIONLESS : dict
    Key and value represents dimensionless quantities.
UNITS : dict
    Combination of all unit conversion dictionaries.  Keys are units and values are the conversion factor for that
    unit into the corresponding SI unit.
SI_UNITS : dict
    Keys are the SI unit for the dictionaries stored as values.


Notes
-----
Conversion factors taken from Perry's Chemical Engineer's Handbook [1]_.

References
----------
[1] Perry's Chemical Engineers' Handbook; Perry, R. H., Southard, M. Z., Eds.; McGraw-Hill Education: New York, 2019.
"""

# TODO:  Implement as frozen dictionaries.
MOLECULAR_WEIGHT = {'g/mol': 1.0,
                    'Da': 1.0}

AMOUNT = {'lbmmol': 453.5924,
          'stdm3': 44.6158,
          'stdft3': 1.1953,
          'kmol': 0.001,
          'mol': 1.0}

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
            'MPa': 1000000.0,
            'Pa': 1.0}

DENSITY = {'lbm/cuft': 16.01846,
           'lbm/ft3': 16.01846,
           'lbm/usgal': 119.8264,
           'lbm/ukgal': 99.77633,
           'kg/m3': 1.0}

MOLAR_DENSITY = {'kmol/m3': 1000.0,
                 'mol/dm3': 1000.0,
                 'mol/m3': 1.0}

MOLAR_VOLUME = {'m3/kmol': 0.001,
                'm3/mol': 1.0}

ENERGY = {'Btu': 1054.4,
          'J': 1.0}

HEAT_OF_VAPORIZATION = {'J/kmol': 0.001,
                        'cal/mol': 4.184,
                        'kcal/mol': 4184.0,
                        'J/mol': 1.0}

HEAT_CAPACITY = {'J/kmol.K': 0.001,
                 'J/mol.K': 1.0}

VISCOSITY = {'Pa.s': 1.0}

THERMAL_CONDUCTIVITY = {'W/m.K': 1.0}

SURFACE_TENSION = {'mN/m': 0.001,
                   'dyne/cm': 0.001,
                   'N/m': 1.0}

DIMENSIONLESS = {'dimensionless': 1.0}

# Temperature is left out of this dictionary because conversion is more than just multiplication by a constant.
UNITS = {**MOLECULAR_WEIGHT,
         **AMOUNT,
         **MASS,
         **LENGTH,
         **AREA,
         **VOLUME,
         **FORCE,
         **PRESSURE,
         **DENSITY,
         **MOLAR_DENSITY,
         **MOLAR_VOLUME,
         **ENERGY,
         **HEAT_OF_VAPORIZATION,
         **HEAT_CAPACITY,
         **VISCOSITY,
         **THERMAL_CONDUCTIVITY,
         **SURFACE_TENSION,
         **DIMENSIONLESS}

SI_UNITS = {'g/mol': MOLECULAR_WEIGHT,
            'mol': AMOUNT,
            'kg': MASS,
            'm': LENGTH,
            'm2': AREA,
            'm3': VOLUME,
            'N': FORCE,
            'Pa': PRESSURE,
            'K': TEMPERATURE,
            'kg/m3': DENSITY,
            'mol/m3': MOLAR_DENSITY,
            'm3/mol': MOLAR_VOLUME,
            'J': ENERGY,
            'J/mol': HEAT_OF_VAPORIZATION,
            'J/mol.K': HEAT_CAPACITY,
            'Pa.s': VISCOSITY,
            'W/m.K': THERMAL_CONDUCTIVITY,
            'N/m': SURFACE_TENSION,
            'dimensionless': DIMENSIONLESS}


def to_si(value: float, unit: str) -> float:
    """Convert input value to corresponding SI value.

    Parameters
    ----------
    value : float
        Input value to be converted to corresponding SI value.
    unit : str
        Unit of input value.

    Returns
    -------
    float
        Value converted to corresponding SI unit.
    """
    if unit in UNITS:
        return value * UNITS[unit]
    elif unit in TEMPERATURE:
        return TEMPERATURE[unit](value)
    else:
        raise ValueError("unit is not defined.")


def to_si_unit(unit: str) -> str:
    """Convert input value to corresponding SI value.

    Parameters
    ----------
    unit : str
        Input unit to be converted to corresponding SI unit

    Returns
    -------
    float
        Value converted to corresponding SI unit.
    """
    for si_unit, unit_dict in SI_UNITS.items():
        if unit in unit_dict:
            return si_unit
    raise ValueError("unit is not defined.")
