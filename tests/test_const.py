from src.phaseprop import const
import pytest


@pytest.fixture
def constant():
    return const.Const(
        value=273.15,
        unit="K",
        uncertainty=0.1,
        source="ACS style source",
        notes="Example notes",
    )


def test_const(constant):
    assert constant == 273.15
    assert constant.unit == "K"
    assert constant.uncertainty == 0.1
    assert constant.source == "ACS style source"
    assert constant.notes == "Example notes"
    with pytest.raises(ValueError, match="Unit is not defined."):
        constant.unit = "bar"
