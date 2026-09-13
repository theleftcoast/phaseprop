from src.phaseprop import utility
import pytest


@pytest.fixture
def const():
    return utility.Const(
        value=273.15,
        unit="K",
        uncertainty=0.1,
        source="ACS style source",
        notes="Example notes",
    )


def test_const(const):
    assert const == 273.15
    assert const.unit == "K"
    assert const.uncertainty == 0.1
    assert const.source == "ACS style source"
    assert const.notes == "Example notes"
    with pytest.raises(ValueError, match="Unit is not defined."):
        const.unit = "bar"


@pytest.fixture
def methane_vapor_pressure():
    pvap_l = utility.RiedelPvap(
        a=39.205,
        b=-1324.4,
        c=-3.4366,
        d=0.000031019,
        e=2.0,
        unit="Pa",
        t_unit="K",
        t_min=90.69,
        t_max=190.56,
    )
    return pvap_l


def test_RiedelPvap(methane_vapor_pressure):
    assert methane_vapor_pressure(t=methane_vapor_pressure.t_min) == pytest.approx(
        11687.01
    )
    assert methane_vapor_pressure(t=methane_vapor_pressure.t_max) == pytest.approx(
        4589664.72
    )
