"""Data points and sets for evaluation."""

import numpy as np
import dataclasses
import utility
import const
import typing
import json


@dataclasses.dataclass
class PTX(object):
    """Vapor-liquid equilibrium specified as 'PTX' data point."""
    p: float
    t: float
    x: np.ndarray
    model_p_bub: typing.Optional[float] = None
    model_t_bub: typing.Optional[float] = None
    model_x_flash: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False) # TODO: implement as zeros of len(x)
    model_y_flash: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False) # TODO: implement as zeros of len(x)
    model_y_bub: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False) # TODO: implement as zeros of len(x)
    model_bub_success: typing.Optional[bool] = None
    model_flash_success: typing.Optional[bool] = None
    measurement_type: typing.Optional[str] = None
    exclude: typing.Optional[bool] = False
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class PTY(object):
    """Vapor-liquid equilibrium specified as 'PTY' data point.    """
    p: float
    t: float
    y: np.ndarray
    model_p_dew: typing.Optional[float] = None
    model_t_dew: typing.Optional[float] = None
    model_x_flash: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False) # TODO: implement as zeros of len(x)
    model_y_flash: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False) # TODO: implement as zeros of len(x)
    model_x_dew: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False) # TODO: implement as zeros of len(x)
    model_dew_success: typing.Optional[bool] = None
    model_flash_success: typing.Optional[bool] = None
    measurement_type: typing.Optional[str] = None
    exclude: typing.Optional[bool] = False
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class PTXY(object):
    """Vapor-liquid equilibrium specified as 'PTXY' data point.

    Attributes
    ----------
    p : float
        Pressure (in 'kPa').
    t : float
        Temperature (in 'K').
    x : list
        List of liquid phase mole fractions.
    y : list
        List of vapor phase mole fractions.
    model_p_bub : float
        Model predicted bubble point pressure at 't' and 'x' (in 'kPa')
    model_p_dew : float
        Model predicted dew point pressure at 't' and 'y' (in 'kPa')
    model_t_bub : float
        Model predicted bubble point pressure at 'p' and 'x' (in 'kPa')
    model_t_dew : float
        Model predicted dew point pressure at 'p' and 'y' (in 'kPa')
    model_x_flash : list
        Model predicted liquid phase mole fractions from a flash at 't' and 'p'.
    model_y_flash : list
        Model predicted vapor phase mole fractions from a flash at 't' and 'p'.
    model_x_dew : list
        Model predicted liquid phase mole fractions from a dew point calculation.
    model_y_bubble : list
        Model predicted vapor phase mole fractions from a bubble point calculation.
    measurement_type : str
        Measurement type (either 'isothermal' or 'isobaric')
    obj_fun_type : str
        Objective function type (either 'isothermal' or 'isobaric')
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.
    """
    p: float
    t: float
    x: np.ndarray
    y: np.ndarray
    model_p_bub: typing.Optional[float] = None
    model_p_dew: typing.Optional[float] = None
    model_t_bub: typing.Optional[float] = None
    model_t_dew: typing.Optional[float] = None
    model_x_flash: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False) # TODO: implement as zeros of len(x)
    model_y_flash: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False) # TODO: implement as zeros of len(x)
    model_x_dew: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False) # TODO: implement as zeros of len(x)
    model_y_bub: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False) # TODO: implement as zeros of len(x)
    model_bub_success: typing.Optional[bool] = None
    model_dew_success: typing.Optional[bool] = None
    model_flash_success: typing.Optional[bool] = None
    measurement_type: typing.Optional[str] = None
    exclude: typing.Optional[bool] = False
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class LLE(object):
    """Liquid-liquid equilibrium.

    Attributes
    ----------
    p : float
        Pressure (in 'kPa').
    t : float
        Temperature (in 'K').
    x : list
        List of liquid phase mole fractions.
    model_x1 : list
        Model predicted first liquid phase mole fractions.
    model_x2 : list
        Model predicted second liquid phase mole fractions.
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    None
    """
    p: float
    t: float
    x: np.ndarray
    model_x: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2), init=False)
    model_success: typing.Optional[bool] = None
    exclude: typing.Optional[bool] = False
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class SLE(object):
    """Solid-liquid equilibrium.

    Attributes
    ----------
    p : float
        Pressure (in 'kPa').
    t : float
        Temperature (in 'K').
    x : list
        List of liquid phase mole fractions.
    model_t : list
        Model predicted temperature (in 'K').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    None
    """
    p: float
    t: float
    x: np.ndarray
    model_t: typing.Optional[float] = None
    model_success: typing.Optional[bool] = None
    exclude: typing.Optional[bool] = False
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class HE(object):
    """Excess enthalpy.

    Attributes
    ----------
    p : float
        Pressure (in 'kPa').
    t : float
        Temperature (in 'K').
    x : list
        List of mole fractions.
    he : float
        Excess enthalpy (in 'J/mol').
    model_he : list
        Model predicted excess enthalpy (in 'J/mol').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    None
    """
    p: float
    t: float
    x: np.ndarray
    he: float
    model_he: typing.Optional[float] = None
    model_success: typing.Optional[bool] = None
    exclude: typing.Optional[bool] = False
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class CPE(object):
    """Excess isobaric heat capacity.

    Attributes
    ----------
    p : float
        Pressure (in 'kPa').
    t : float
        Temperature (in 'K').
    x : list
        List of mole fractions.
    cpe : float
        Excess isobaric heat capacity (in '').
    model_cpe : list
        Model predicted excess enthalpy (in '').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    None
    """
    p: float
    t: float
    x: np.ndarray
    cpe: float
    model_cpe: typing.Optional[float] = None
    model_success: typing.Optional[bool] = None
    exclude: typing.Optional[bool] = False
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class RHOLSAT(object):
    """Bubble point liquid density.

    Attributes
    ----------
    t : float
        Bubble point temperature (in 'K').
    x : list
        List of mole fractions.
    rho_l : float
        Bubble point liquid density (in 'kg/m3').
    model_rho_l: list
        Model predicted liquid density (in '').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    None
    """
    t: float
    x: np.ndarray
    rho_l: float
    model_rho_l: typing.Optional[float] = None
    model_success: typing.Optional[bool] = None
    exclude: typing.Optional[bool] = False
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class PSAT(object):
    """Bubble point pressure.

    Attributes
    ----------
    t : float
        Bubble point temperature (in 'K').
    x : list
        List of mole fractions.
    pressure : float
        Bubble point pressure (in '').
    model_rho_l: list
        Model predicted bubble point pressure (in '').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    None
    """
    t: float
    x: np.ndarray
    p: float
    model_p: typing.Optional[float] = None
    model_success: typing.Optional[bool] = None
    exclude: typing.Optional[bool] = False
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None


@dataclasses.dataclass
class DataSet(object):
    """Container for binary data set."""
    comps: list = dataclasses.field(init=False, default_factory=list)
    data: list = dataclasses.field(init=False, default_factory=list)
    refs: dict = dataclasses.field(init=False, default_factory=dict)
    vle_isotherms: dict = dataclasses.field(init=False, default_factory=dict)
    vle_isobars: dict = dataclasses.field(init=False, default_factory=dict)
    lle_isobars: dict = dataclasses.field(init=False, default_factory=dict)
    he_isotherms: dict = dataclasses.field(init=False, default_factory=dict)
    cpe_isotherms: dict = dataclasses.field(init=False, default_factory=dict)

    def append(self, data) -> None:
        if isinstance(data, (PTX, PTY, PTXY, LLE, SLE, HE, CPE, RHOLSAT)):
            self.data.append(data)

    def load_from_json(self, file: str) -> None:
        """Load binary data from JSON file."""
        path = const.ROOT_DIR / 'data' / file
        with open(path) as user_file:
            file_contents = user_file.read()
            json_contents = json.loads(file_contents)

            if json_contents.get('COMPS'):
                for comp in json_contents.get('COMPS'):
                    self.comps.append(comp['name'])

            if json_contents.get('REFS'):
                for ref in json_contents['REFS']:
                    self.refs[ref['ref']] = ref['citation']

            if json_contents.get('VLE'):
                for vle in json_contents.get('VLE'):
                    if vle.get('p') and vle.get('t') and vle.get('x1_l') and vle.get('x1_v'):
                        self.append(PTXY(p=vle['p'],
                                         t=vle['t'],
                                         x=np.array([vle['x1_l'], 1.0 - vle['x1_l']]),
                                         y=np.array([vle['x1_v'], 1.0 - vle['x1_v']]),
                                         source=vle['source']))
                    if vle.get('p') and vle.get('t') and vle.get('x1_l') and vle.get('x1_v') is None:
                        self.append(PTX(p=vle['p'],
                                        t=vle['t'],
                                        x=np.array([vle['x1_l'], 1.0 - vle['x1_l']]),
                                        source=vle['source']))
                    if vle.get('p') and vle.get('t') and vle.get('x1_v') and vle.get('x1_l') is None:
                        self.append(PTY(p=vle['p'],
                                        t=vle['t'],
                                        y=np.array([vle['x1_v'], 1.0 - vle['x1_v']]),
                                        source=vle['source']))

            if json_contents.get('LLE'):
                for lle in json_contents.get("LLE"):
                    self.append(LLE(p=lle['p'],
                                    t=lle['t'],
                                    x=np.array([lle['x1_l'], 1.0 - lle['x1_l']]),
                                    source=lle['source']))

            if json_contents.get('SLE'):
                for sle in json_contents.get('SLE'):
                    self.append(SLE(p=sle['p'],
                                    t=sle['t'],
                                    x=np.array([sle['x1_l'], 1.0 - sle['x1_l']]),
                                    source=sle['source']))

            if json_contents.get('HE'):
                for he in json_contents.get('HE'):
                    self.append(HE(p=he['p'],
                                   t=he['t'],
                                   x=np.array([he['x1'], 1.0 - he['x1']]),
                                   he=he['he'] * 1000, 
                                   source=he['source'])) # JSON data source in kJ/mol, converted to J/mole here before creating object.

            if json_contents.get('CPE'):
                for cpe in json_contents.get('CPE'):
                    self.append(CPE(p=cpe['p'],
                                    t=cpe['t'],
                                    x=np.array([cpe['x1'], 1.0 - cpe['x1']]),
                                    cpe=cpe['cpe'],
                                    source=cpe['source']))

            if json_contents.get('RHOLSAT'):
                for rhol_sat in json_contents.get('RHOLSAT'):
                    self.append(RHOLSAT(t=rhol_sat['t'],
                                        x=np.array([rhol_sat['x1_l'], 1.0 - rhol_sat['x1_l']]),
                                        rho_l=rhol_sat['rhol_sat'],
                                        source=rhol_sat['source']))

        self.annotate()

    def annotate(self) -> None:
        # VLE isotherms
        temperatures = [d.t for d in self.data if isinstance(d, (PTXY, PTX, PTY))]
        for isotherm in utility.cluster(data=temperatures, max_gap=0.05):
            if len(isotherm) >= 3:
                for i in isotherm:
                    for d in self.data:
                        if d.t == i and isinstance(d, (PTXY, PTX, PTY)):
                            d.measurement_type = 'isothermal'

        temperatures = [d.t for d in self.data if isinstance(d, (PTXY, PTX, PTY)) and d.measurement_type == 'isothermal']
        for isotherm in utility.cluster(data=temperatures):
            self.vle_isotherms[round(sum(isotherm) / len(isotherm), 3)] = isotherm

        # VLE isobars
        pressures = [d.p for d in self.data if isinstance(d, (PTXY, PTX, PTY))]
        for isobar in utility.cluster(data=pressures, max_gap=0.05):
            if len(isobar) >= 3:
                for i in isobar:
                    for d in self.data:
                        if d.p == i and isinstance(d, (PTXY, PTX, PTY)):
                            d.measurement_type = 'isobaric'

        pressures = [d.p for d in self.data if isinstance(d, (PTXY, PTX, PTY)) and d.measurement_type == 'isobaric']
        for isobar in utility.cluster(data=pressures):
            self.vle_isobars[round(sum(isobar) / len(isobar), 3)] = isobar

        # LLE isobars
        pressures = [d.p for d in self.data if isinstance(d, LLE)]
        for isobar in utility.cluster(data=pressures):
            if len(isobar) >= 3:
                self.lle_isobars[round(sum(isobar) / len(isobar), 3)] = isobar

        # Excess enthalpy isotherms and max value
        temperatures = [d.t for d in self.data if isinstance(d, HE)]
        for isotherm in utility.cluster(data=temperatures):
            if len(isotherm) >= 5:
                self.he_isotherms[round(sum(isotherm) / len(isotherm), 3)] = isotherm

        # Excess isobaric heat capacity isotherms and max values
        temperatures = [d.t for d in self.data if isinstance(d, CPE)]
        for isotherm in utility.cluster(data=temperatures):
            if len(isotherm) >= 5:
                self.cpe_isotherms[round(sum(isotherm) / len(isotherm), 3)] = isotherm

    def summary(self, verbose: bool = False):
        output = list()
        output.append("PTXY Datapoints: {}\n".format(sum(isinstance(x, PTXY) for x in self.data)))
        output.append("PTX Datapoints: {}\n".format(sum(isinstance(x, PTX) for x in self.data)))
        output.append("PTY Datapoints: {}\n".format(sum(isinstance(x, PTY) for x in self.data)))
        output.append("PTX Update Status:")
        bubble_success = []
        bubble_fail = []
        flash_success = []
        flash_fail = []
        for d in self.data:
            if isinstance(d, PTX):
                if d.model_bub_success:
                    bubble_success.append((d.t, d.p))
                else:
                    bubble_fail.append((d.t, d.p))
                if d.model_flash_success:
                    flash_success.append((d.t, d.p))
                else:
                    flash_fail.append((d.t, d.p))
        output.append('  Bubble point successes: {} points\n'.format(len(bubble_success)))
        output.append('  Bubble point failures: {} points\n'.format(len(bubble_fail)))
        output.append('  Flash successes: {} points\n'.format(len(flash_success)))
        output.append('  Flash failures: {} points\n'.format(len(flash_fail)))
        output.append('VLE Isotherms: \n')
        for t, values in self.vle_isotherms.items():
            output.append('  {} K: {} points\n'.format(t, len(values)))
        output.append('VLE Isobars: \n')
        for t, values in self.vle_isobars.items():
            output.append('  {} bar: {} points\n'.format(t, len(values)))
        output.append("LLE Datapoints: {}\n".format(sum(isinstance(x, LLE) for x in self.data)))
        output.append('LLE Isobars: \n')
        for t, values in self.lle_isobars.items():
            output.append('  {} bar: {} points\n'.format(t, len(values)))
        output.append("SLE Datapoints: {}\n".format(sum(isinstance(x, SLE) for x in self.data)))
        output.append("HE Datapoints: {}\n".format(sum(isinstance(x, HE) for x in self.data)))
        output.append('HE Isotherms: \n')
        for t, values in self.he_isotherms.items():
            output.append('  {} K: {} points\n'.format(t, len(values)))
        output.append("CPE Datapoints: {}\n".format(sum(isinstance(x, CPE) for x in self.data)))
        output.append('CPE Isotherms: \n')
        for t, values in self.cpe_isotherms.items():
            output.append('  {} K: {} points\n'.format(t, len(values)))
        output.append("RHOLSAT Datapoints: {}\n".format(sum(isinstance(x, RHOLSAT) for x in self.data)))
        return "".join(output)

    def __str__(self):
        output = list()
        output.append("PTXY Datapoints: {}\n".format(sum(isinstance(x, PTXY) for x in self.data)))
        output.append("PTX Datapoints: {}\n".format(sum(isinstance(x, PTX) for x in self.data)))
        output.append("PTY Datapoints: {}\n".format(sum(isinstance(x, PTY) for x in self.data)))
        output.append("LLE Datapoints: {}\n".format(sum(isinstance(x, LLE) for x in self.data)))
        output.append("SLE Datapoints: {}\n".format(sum(isinstance(x, SLE) for x in self.data)))
        output.append("HE Datapoints: {}\n".format(sum(isinstance(x, HE) for x in self.data)))
        output.append("CPE Datapoints: {}\n".format(sum(isinstance(x, CPE) for x in self.data)))
        output.append("RHOLSAT Datapoints: {}\n".format(sum(isinstance(x, RHOLSAT) for x in self.data)))
        return "".join(output)
