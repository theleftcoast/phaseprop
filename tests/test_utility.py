from src.phaseprop import utility
import pytest


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
