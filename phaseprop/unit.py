"""Unit conversion.

Attributes
----------
MASS : dict
    Keys are units for mass and values are the conversion factor for that unit into kilograms (or 'kg').
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
UNITS : dict
    Combination of all unit conversion dictionaries.  Keys are units and values are the conversion factor for that
    unit into the corresponding SI unit.

Notes
-----
Conversion factors taken from Perry's Chemical Engineer's Handbook [1]_.

References
----------
[1] Perry's Chemical Engineers' Handbook; Perry, R. H., Southard, M. Z., Eds.; McGraw-Hill Education: New York, 2019.
"""

# TODO:  Implement as frozen dictionaries.
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

# Temperature is left out of this list because conversion is more than just multiplication by a constant.
UNITS = {**MASS,
         **LENGTH,
         **AREA,
         **VOLUME,
         **FORCE,
         **PRESSURE,
         **DENSITY,
         **MOLAR_DENSITY,
         **ENERGY,
         **AMOUNT,
         **HEAT_OF_VAPORIZATION,
         **HEAT_CAPACITY}


def to_si(value: float, unit: str) -> float:
    """Convert input to corresponding SI unit.

    Parameters
    ----------
    value : float
        Input value to be converted to corresponding SI unit.
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
