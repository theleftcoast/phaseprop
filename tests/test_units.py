from src.phaseprop import units
import pytest

def test_to_si_unit():
    assert units.to_si_unit(unit="psi") == "Pa"
    assert units.to_si_unit(unit="F") == "K"
    with pytest.raises(ValueError, match="Unit is not defined."):
        units.to_si_unit(unit="bar")

def test_to_si():
    assert units.to_si(value=32.0, unit="F") == 273.15
    assert units.to_si(value=212.0, unit="F") == 373.15
    assert units.to_si(value=1.0, unit="atm") == 101325.0
    with pytest.raises(ValueError, match="Unit is not defined."):
        units.to_si(value=1.0, unit="bar")