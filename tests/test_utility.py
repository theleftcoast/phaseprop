from src.phaseprop import utility
import pytest


@pytest.fixture
def const():
    return utility.Const(value=273.15, unit='K', uncertainty=0.1, source='ACS style source', notes='Example notes')

def test_const(const):
    assert const == 273.15
    assert const.unit == 'K'
    assert const.uncertainty == 0.1
    assert const.source == 'ACS style source'
    assert const.notes == 'Example notes'
    with pytest.raises(ValueError, match="Unit is not defined."):
            const.unit = 'bar'