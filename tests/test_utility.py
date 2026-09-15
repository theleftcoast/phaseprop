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


@pytest.fixture
def methane_liquid_density():
    den_l = utility.DaubertDenL(
        a=2.9214,
        b=0.28976,
        c=190.56,
        d=0.28881,
        unit="mol/dm3",
        t_unit="K",
        t_min=90.69,
        t_max=190.56,
    )
    return den_l


def test_DaubertDenL(methane_liquid_density):
    assert methane_liquid_density(t=methane_liquid_density.t_min) == pytest.approx(
        28179.94
    )
    assert methane_liquid_density(t=methane_liquid_density.t_max) == pytest.approx(
        10082.13
    )


@pytest.fixture
def water_liquid_density():
    den_l = utility.IAPWSDenL()
    return den_l


def test_IAPWSDenL(water_liquid_density):
    assert water_liquid_density(t=water_liquid_density.t_min) == pytest.approx(55487.44)
    assert water_liquid_density(t=water_liquid_density.t_max) == pytest.approx(17874.0)
