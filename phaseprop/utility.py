"""Utility objects."""

import dataclasses
import typing
import numpy as np
import units


class Callback(list):
    """Manage callback execution.

    References
    ----------
    [1] http://web.archive.org/web/20060612061259/http://www.suttoncourtenay.org.uk/duncan/accu/pythonpatterns.html
    """
    def __init__(self):
        self._delegates = []

    @property
    def delegates(self):
        """list of callable : List of callback functions."""
        return self._delegates

    def add(self, callback):
        """Add a function to delegates.

        Parameters
        ----------
        callback : callable
            Function to be added to delegates.
        """
        if callable(callback):
            self._delegates.append(callback)

    def remove(self, callback):
        """Remove a function from delegates.

        Parameters
        ----------
        callback : callable
            Function to be removed from delegates.
        """
        if callable(callback):
            for i, d in enumerate(self._delegates):
                if d == callback:
                    del self._delegates[i]

    def fire(self, *args, **kwargs):
        """Execute all callback functions in delegates.

        Parameters
        ----------
        *args : list
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        for d in self._delegates:
            d(*args, **kwargs)


class Const(float):
    """Constant with metadata.

    Notes
    -----
    """
    def __new__(cls,
                value: float,
                unit: typing.Optional[str] = None,
                uncertainty: typing.Optional[float] = None,
                source: typing.Optional[str] = None,
                notes: typing.Optional[str] = None):
        return float.__new__(cls, value)

    def __init__(self,
                 value: float,
                 unit: typing.Optional[str] = None,
                 uncertainty: typing.Optional[float] = None,
                 source: typing.Optional[str] = None,
                 notes: typing.Optional[str] = None):
        super().__init__(value)
        self.unit = unit
        self.uncertainty = uncertainty
        self.source = source
        self.notes = notes

    @property
    def unit(self):
        """str : Source unit for constant."""
        return self._unit

    @unit.setter
    def unit(self, value):
        if value in units.UNITS:
            self._unit = value
            return
        elif value is None:
            self._unit = value
            return
        else:
            raise ValueError("unit is not defined.")


@dataclasses.dataclass
class RiedelPvap(object):
    """Riedel vapor pressure correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    e : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'Pa'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'Pa').
    mae : float, optional
        Mean absolute error (in 'Pa')
    mape : float, optional
        Mean absolute percentage error (in 'Pa').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    The Riedel vapor pressure equation is a common correlation used to fit vapor pressure data.  Experimental data can
    be fit within a few tenths of a percent over the entire fluid range for most fluids.  Riedel's equation is
    recommended by several standard references _[1, 2, 3]. Constants are available in the DIPPR database _[2, 3].

        Pvap = exp(a + b/t + c*ln(t) + d*t**e)

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically after the correlation is evaluated (scaling correlation parameters before
    evaluation is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Poling, B.E.; Praunitz, J.M.; O'Connell, J.P. The properties of gases and liquids, 5th ed.; McGraw-Hill, 2000.
    [2] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [3] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    unit: str = 'Pa'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def __post_init__(self):
        if self.unit not in units.UNITS:
            raise ValueError("unit is not defined.")
        elif self.t_unit not in units.TEMPERATURE:
            raise ValueError("t_unit is not defined.")

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """Riedel vapor pressure correlation."""
        return np.exp(self.a +
                      self.b / t +
                      self.c * np.log(t) +
                      self.d * t ** self.e)

    def __call__(self, t: float) -> float:
        """Evaluate vapor pressure.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Vapor pressure evaluated at 't' (in 'Pa').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class DaubertDenL(object):
    """Daubert and Danner liquid density correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'mol/m3'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'mol/m3').
    mae : float, optional
        Mean absolute error (in 'mol/m3')
    mape : float, optional
        Mean absolute percentage error (in 'mol/m3').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    The Daubert and Danner equation is a common correlation used to fit saturated liquid density data.  It is an
    adaptation of the Rackett equation which can fit experimental data within one to two percent over the entire fluid
    range (for most fluids).  It does not perform as well for highly hydrogen bonded fluids (such as water). Daubert and
    Danner's equation is recommended by several standard references _[1, 2, 3, 4]. Constants are available from the
    DIPPR database _[2, 3] and Yaw's handbook _[4] (though the Yaw's constants need conversion).

        DenL = a / (b ** (1.0 + (1.0 - t/c) ** d)

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Poling, B.E.; Praunitz, J.M.; O'Connell, J.P. The properties of gases and liquids, 5th ed.; McGraw-Hill, 2000.
    [2] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [3] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    [4] Yaws, C. L. Thermophysical properties of chemicals and hydrocarbons, 2nd ed.; Gulf Professional Publishing is an
    imprint of Elsevier: Kidlington, Oxford, 2014.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    unit: str = 'mol/m3'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """Daubert and Danner liquid density correlation."""
        return self.a / self.b ** (1.0 + (1.0 - t / self.c) ** self.d)

    def __call__(self, t: float) -> float:
        """Evaluate liquid density.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Liquid density evaluated at 't' (in 'mol/m3').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class IAPWSDenL(object):
    """International Association for the Properties of Water and Steam (IAPWS) liquid density correlation.

    Parameters
    ----------
    a : float, default: 17.874
        Correlation parameter.
    b : float, default: 35.618
        Correlation parameter.
    c : float, default: 19.655
        Correlation parameter.
    d : float, default: -9.1306
        Correlation parameter.
    e : float, default: -31.367
        Correlation parameter.
    f : float, default: -813.56
        Correlation parameter.
    g : float, default: -17421000
        Correlation parameter.
    h : float, default: 647.096
        Critical temperature.
    unit : str, default: 'mol/m3'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'mol/m3').
    mae : float, optional
        Mean absolute error (in 'mol/m3')
    mape : float, optional
        Mean absolute percentage error (in 'mol/m3').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    The International Association for the Properties of Water and Steam (IAPWS) liquid density equation is a very high
    accuracy saturated liquid density correlation for water. Experimental data is reproduced within a tenth of a percent
    over the entire fluid range.

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Sengers, J. M. H. L; Dooley, B. Revised Supplementary Release on Saturation Properties of Ordinary Water
    Substance; IAPWS, 1992.
    """
    a: float = 17.874
    b: float = 35.618
    c: float = 19.655
    d: float = -9.1306
    e: float = -31.367
    f: float = -813.56
    g: float = -17421000
    h: float = 647.096
    unit: str = 'mol/m3'
    t_unit: str = 'K'
    t_min: float = 273.16
    t_max: float = 647.096
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _tau(self, t: float) -> float:
        return 1.0 - t / self.h

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """IAPWS liquid density correlation."""
        return self.a + \
               self.b * self._tau(t) ** (1.0/3.0) + \
               self.c * self._tau(t) ** (2.0/3.0) + \
               self.d * self._tau(t) ** (5.0/3.0) + \
               self.e * self._tau(t) ** (16.0/3.0) + \
               self.f * self._tau(t) ** (43.0/3.0) + \
               self.g * self._tau(t) ** (110.0/3.0)

    def __call__(self, t: float) -> float:
        """Evaluate liquid density.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Liquid density evaluated at 't' (in 'mol/m3').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class PerryHvap(object):
    """Perry's enthalpy of vaporization correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    e : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'J/mol'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'J/mol').
    mae : float, optional
        Mean absolute error (in 'J/mol')
    mape : float, optional
        Mean absolute percentage error (in 'J/mol').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    The Perry's enthalpy of vaporization equation _[1, 2] is a common correlation used to fit experimental data.
    Experimental data can be fit within a few tenths of a percent over the entire fluid range for most fluids. A
    truncated Perry's equation (c, d, e = 0) is recommended for simple fluids as well as for interpolations when limited
    experimental data is available _[3, 4]. Constants are available from the DIPPR database _[1, 2] for the full
    equation and from Yaw's handbook _[4] for the truncated equation.

        Hvap = a * (1 - tr) ** (b + c*tr + d*tr**2 + e*tr**3)

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [2] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    [3] Poling, B.E.; Praunitz, J.M.; O'Connell, J.P. The properties of gases and liquids, 5th ed.; McGraw-Hill, 2000.
    [4] Yaws, C. L. Thermophysical properties of chemicals and hydrocarbons, 2nd ed.; Gulf Professional Publishing is an
    imprint of Elsevier: Kidlington, Oxford, 2014.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    unit: str = 'J/mol'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _tr(self, t: float) -> float:
        """Reduced temperature"""
        return t / self.e

    def _corel(self, t: float) -> float:
        """DIPPR enthalpy of vaporization."""
        return self.a * (1.0 - self._tr(t)) ** (self.b +
                                                self.c * self._tr(t) +
                                                self.d * self._tr(t)**2.0 +
                                                self.e * self._tr(t)**3.0)

    def __call__(self, t: float) -> float:
        """Evaluate enthalpy of vaporization.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Enthalpy of vaporization evaluated at 't' (in 'J/mol').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class PolyCpL(object):
    """Polynomial liquid heat capacity correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    e : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'J/mol.K'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'J/mol.K').
    mae : float, optional
        Mean absolute error (in 'J/mol.K')
    mape : float, optional
        Mean absolute percentage error (in 'J/mol.K').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    A fourth order polynomial equation is a flexible and common correlation used to fit saturated liquid heat capacity
    data _[1, 2, 3]. Third order polynomials are often adequate for fitting experimental data within a percent over the
    entire fluid range for most fluids.  Constants are available from the DIPPR database _[2, 3].

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Poling, B.E.; Praunitz, J.M.; O'Connell, J.P. The properties of gases and liquids, 5th ed.; McGraw-Hill, 2000.
    [2] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [3] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    unit: str = 'J/mol.K'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """Polynomial heat capacity correlation."""
        return self.a + self.b * t + self.c * t ** 2.0 + self.d * t ** 3.0 + self.e * t ** 4.0

    def __call__(self, t: float) -> float:
        """Evaluate liquid heat capacity.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Liquid heat capacity evaluated at 't' (in 'J/mol.K').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class DIPPRCpL(object):
    """DIPPR liquid heat capacity correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    e : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'J/mol.K'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'J/mol.K').
    mae : float, optional
        Mean absolute error (in 'J/mol.K')
    mape : float, optional
        Mean absolute percentage error (in 'J/mol.K').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    A specialized mixed order polynomial equation is used to fit liquid heat capacity data for select fluids _[1, 2].
    Constants are available in the DIPPR database _[1, 2].

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [2] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    unit: str = 'J/mol.K'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _tau(self, t: float) -> float:
        """Parameter used to evaluate heat capacity correlation."""
        return 1.0 - t / self.e

    def _corel(self, t: float) -> float:
        """DIPPR heat capacity correlation."""
        return (self.a ** 2.0) / self._tau(t) + \
               self.b - \
               2.0 * self.a * self.c * self._tau(t) - \
               (self.a * self.d * self._tau(t) ** 2.0) - \
               (self.c ** 2.0) * (self._tau(t) ** 3.0) / 3.0 - \
               (self.c * self.d * self._tau(t) ** 4.0) / 2.0 - \
               (self.d ** 2.0) * (self._tau(t) ** 5.0) / 5.0

    def __call__(self, t: float) -> float:
        """Evaluate liquid heat capacity.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Liquid heat capacity evaluated at 't' (in 'J/mol.K').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class PolyCpIg(object):
    """Polynomial ideal gas heat capacity, enthalpy, and entropy correlations.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    e : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'J/mol.K'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'J/mol.K').
    mae : float, optional
        Mean absolute error (in 'J/mol.K')
    mape : float, optional
        Mean absolute percentage error (in 'J/mol.K').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    A fourth order polynomial equation is a flexible and common correlation used to fit ideal gas heat capacity,
    enthalpy, and entropy data _[1, 2]. Third order polynomials are often adequate for fitting experimental data
    within a percent over the entire fluid range for most lfuids. Constants are available from many sources _[1, 2].

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Poling, B.E.; Praunitz, J.M.; O'Connell, J.P. The properties of gases and liquids, 5th ed.; McGraw-Hill, 2000.
    [2] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    unit: str = 'J/mol.K'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """Polynomial heat capacity correlation."""
        return self.a + self.b * t + self.c * t ** 2.0 + self.d * t ** 3.0 + self.e * t ** 4.0

    def __call__(self, t: float) -> float:
        """Evaluate ideal gas heat capacity.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Ideal gas heat capacity evaluated at 't' (in 'J/mol.K').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)

    def enthalpy(self, t: float) -> float:
        """Evaluate ideal gas enthalpy.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Ideal gas enthalpy evaluated at 't' (in 'J/mol').
        """
        return self.a * t + \
               (self.b * t ** 2.0) / 2.0 + \
               (self.c * t ** 3.0) / 3.0 + \
               (self.d * t ** 4.0) / 4.0 + \
               (self.e * t ** 5.0) / 5.0

    def entropy(self, t: float) -> float:
        """Evaluate ideal gas entropy.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Ideal gas entropy at 't' (in 'J/mol.K').
        """
        return self.a * np.log(t) + self.b * t + \
               ((t ** 2.0) / 12.0) * (6.0 * self.c + 4.0 * self.d * t + 3.0 * self.e * t ** 2.0)


@dataclasses.dataclass
class AlyLeeCpIg(object):
    """Aly and Lee ideal gas heat capacity, enthalpy, and entropy correlations.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    e : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'J/mol.K'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'J/mol.K').
    mae : float, optional
        Mean absolute error (in 'J/mol.K')
    mape : float, optional
        Mean absolute percentage error (in 'J/mol.K').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    The Aly and Lee _[1] ideal gas heat capacity equation is a simplification of fundamental statistical mechanics
    expressions. It is a self-consistent way to calculate ideal gas heat capacity, enthalpy and entropy with reliable
    performance when extrapolated outside the temperature range originally used for parameter estimation. Experimental
    data can be fit within a few tenths of a percent over the entire fluid range for most fluids.  The Ally and Lee
    equation is recommended by standard references _[2, 3]. Constants are available in the DIPPR database _[2, 3].

        Cp = a + b*((c/t) / sinh(c/t))**2.0 + d*((e/t) / cosh(e/t))**2.0
        H = a*t + b*c / tanh(c/t) - d*e * tanh(e/t)
        S = a*ln(t) + b*((c/t) / tanh(c/t) - ln(sinh(c/t))) - d*((e/t)*tanh(e/t) - ln(cosh(e/t)))

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Aly, F. A.; Lee, L. L. Self-Consistent Equations for Calculating the Ideal Gas Heat Capacity, Enthalpy, and
    Entropy. Fluid Phase Equilib. 1981, 6, 169-179.
    [2] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [3] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    unit: str = 'J/mol.K'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """Aly and Lee heat capacity correlation."""
        return self.a + \
               self.b * ((self.c / t) / np.sinh(self.c / t)) ** 2.0 + \
               self.d * ((self.e / t) / np.cosh(self.e / t)) ** 2.0

    def __call__(self, t: float) -> float:
        """Evaluate ideal gas heat capacity.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Ideal gas heat capacity evaluated at 't' (in 'J/mol.K').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)

    def enthalpy(self, t: float) -> float:
        """Evaluate ideal gas enthalpy.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Ideal gas enthalpy evaluated at 't' (in 'J/mol').
        """
        return self._a * t + \
               self._b * self._c / np.tanh(self._c / t) - \
               self._d * self._e * np.tanh(self._e / t)

    def entropy(self, t: float) -> float:
        """Evaluate ideal gas entropy.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Ideal gas entropy at 't' (in 'J/mol.K').
        """
        return self._a * np.log(t) + \
               self._b * ((self._c/t) / np.tanh(self._c/t) - np.log(np.sinh(self._c/t))) - \
               self._d * ((self._e/t) * np.tanh(self._e/t) - np.log(np.cosh(self._e/t)))


@dataclasses.dataclass
class AndradeViscL(object):
    """Generalized Andrade saturated liquid viscosity correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    e : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'Pa.s'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'Pa.s').
    mae : float, optional
        Mean absolute error (in 'Pa.s')
    mape : float, optional
        Mean absolute percentage error (in 'Pa.s').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    The generalized Andrade equation is a common correlation used to fit saturated liquid viscosity data. This is
    analogous to the Reidel vapor pressure equation. Experimental data can be fit within a percent over the entire fluid
    range for most fluids.  The generalized Andrade equation is recommended by standard references _[1, 2]. Constants
    are available in the DIPPR database _[2, 3].

        ViscL = exp(a + b/t + c*ln(t) + d*t**e)

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Poling, B.E.; Praunitz, J.M.; O'Connell, J.P. The properties of gases and liquids, 5th ed.; McGraw-Hill, 2000.
    [2] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [3] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    unit: str = 'Pa.s'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """Generalized Andrade saturated liquid viscosity correlation."""
        return np.exp(self.a +
                      self.b / t +
                      self.c * np.log(t) +
                      self.d * t ** self.e)

    def __call__(self, t: float) -> float:
        """Evaluate saturated liquid viscosity.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Saturated liquid viscosity evaluated at 't' (in 'Pa.s').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class KineticViscIg(object):
    """Generalized kinetic theory ideal gas viscosity correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'Pa.s'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'Pa.s').
    mae : float, optional
        Mean absolute error (in 'Pa.s')
    mape : float, optional
        Mean absolute percentage error (in 'Pa.s').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    The generalized kinetic theory equation is a common correlation used to fit ideal gas vapor viscosity data. The
    collision integral accounts for intermolecular forces and is represented as a second-order polynomial. The natural
    log of viscosity is nearly linear with the natural log of temperature. Care should be taken in extrapolating as the
    polynomial representation of the collision integral can introduce unintended mathematical poles as the denominator.
    approaches zero. Experimental data can be fit within a percent over the entire fluid range for most fluids.  The
    generalized kinetic theory equation is recommended by standard references _[1, 2]. Constants are available in the
    DIPPR database _[2, 3].

        ViscIg = (a*t**b)/(1.0 + c/t + d /t**2.0)

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Poling, B.E.; Praunitz, J.M.; O'Connell, J.P. The properties of gases and liquids, 5th ed.; McGraw-Hill, 2000.
    [2] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [3] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    unit: str = 'Pa.s'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """Generalized kinetic theory vapor viscosity correlation."""
        return (self.a * t ** self.b) / (1.0 + self.c / t + self.d / t ** 2.0)

    def __call__(self, t: float) -> float:
        """Evaluate ideal gas viscosity.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Saturated liquid viscosity evaluated at 't' (in 'Pa.s').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class PolyTcondL(object):
    """Polynomial saturated liquid thermal conductivity correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    e : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'W/(m.K)'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'W/(m.K)').
    mae : float, optional
        Mean absolute error (in 'W/(m.K)')
    mape : float, optional
        Mean absolute percentage error (in 'W/(m.K)').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    A fourth order polynomial equation is a flexible and common correlation used to fit ideal gas vapor thermal
    conductivity data. Experimental data can be fit within a percent over the entire fluid range for most fluids. The
    generalized kinetic theory equation is recommended by standard references _[1]. Constants are available in the
    DIPPR database _[1, 2].

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [2] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    unit: str = 'W/(m.K)'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """Polynomial saturated liquid thermal conductivity correlation."""
        return self.a + self.b * t + self.c * t ** 2.0 + self.d * t ** 3.0 + self.e * t ** 4.0

    def __call__(self, t: float) -> float:
        """Evaluate saturated liquid thermal conductivity.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Saturated liquid viscosity evaluated at 't' (in 'W/(m.K)').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class KineticTcondIg(object):
    """Generalized kinetic theory ideal gas thermal conductivity correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'W/(m.K)'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'W/(m.K)').
    mae : float, optional
        Mean absolute error (in 'W/(m.K)')
    mape : float, optional
        Mean absolute percentage error (in 'W/(m.K)').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    The generalized kinetic theory equation is a common correlation used to fit ideal gas vapor thermal conductivity
    data. The collision integral accounts for intermolecular forces and is represented as a second-order polynomial.
    Care should be taken in extrapolating as the polynomial representation of the collision integral can introduce
    unintended mathematical poles as the denominator approaches zero. Experimental data can be fit within a percent over
    the entire fluid range for most fluids. The generalized kinetic theory equation is recommended by standard
    references _[1, 2]. Constants are available in the DIPPR database _[2, 3].

        TcondIg = (a*t**b)/(1.0 + c/t + d /t**2.0)

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Poling, B.E.; Praunitz, J.M.; O'Connell, J.P. The properties of gases and liquids, 5th ed.; McGraw-Hill, 2000.
    [2] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [3] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    unit: str = 'W/(m.K)'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """Generalized kinetic theory vapor thermal conductivity correlation."""
        return (self.a * t ** self.b) / (1.0 + self.c / t + self.d / t ** 2.0)

    def __call__(self, t: float) -> float:
        """Evaluate ideal gas thermal conductivity.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Saturated liquid viscosity evaluated at 't' (in 'W/(m.K)').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class PolyTcondIg(object):
    """Polynomial ideal gas thermal conductivity correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    d : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'W/(m.K)'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'W/(m.K)').
    mae : float, optional
        Mean absolute error (in 'W/(m.K)')
    mape : float, optional
        Mean absolute percentage error (in 'W/(m.K)').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    A fourth order polynomial equation is a flexible and common correlation used to fit ideal gas vapor thermal
    conductivity data. Experimental data can be fit within a percent over the entire fluid range for most fluids. The
    generalized kinetic theory equation is recommended by standard references _[1, 2]. Constants are available in the
    DIPPR database _[2, 3].

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Green, D.; Southard, M. Perry's Chemical Engineers' Handbook, 9th ed.; McGraw Hill Education: New York, 2019.
    [2] Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data Compilation of Pure
    Chemical Properties; Design Institute for Physical Properties, AIChE: New York, NY, 2016.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    unit: str = 'W/(m.K)'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """Polynomial vapor thermal conductivity correlation."""
        return self.a + self.b * t + self.c * t ** 2.0 + self.d * t ** 3.0

    def __call__(self, t: float) -> float:
        """Evaluate ideal gas thermal conductivity.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Saturated liquid viscosity evaluated at 't' (in 'W/(m.K)').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)


@dataclasses.dataclass
class SurfTen(object):
    """Saturated liquid surface tension correlation.

    Parameters
    ----------
    a : float, default: 0.0
        Correlation parameter.
    b : float, default: 0.0
        Correlation parameter.
    c : float, default: 0.0
        Correlation parameter.
    unit : str, default: 'W/(m.K)'
        Output unit for correlation when evaluated with parameters a-e.
    t_unit : float, default: 'K'
        Input unit for correlation for evaluation with parameters a-e.
    t_min : float, optional
        Minimum temperature (in 'K').
    t_max : float, optional
        Maximum temperature (in 'K').
    rmse : float, optional
        Root mean squared error (in 'W/(m.K)').
    mae : float, optional
        Mean absolute error (in 'W/(m.K)')
    mape : float, optional
        Mean absolute percentage error (in 'W/(m.K)').
    source : str, optional
        Source for the correlation (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.

    Notes
    -----
    This general surface tension equation is a flexible way to correlate saturated liquid surface tension data.
    Experimental data can be fit within 1-2 percent over the entire fluid range for most fluids.  This general equation
    is recommended by standard references _[1, 2]. Constants are available in the Yaws' database _[1].

        SurfTen = a*(1.0 - t/b)**c

    Property correlations are developed in a variety of different unit systems. In this implementation, conversion to SI
    units is accomplished automatically when the correlation is called (scaling correlation parameters before evaluation
    is never recommended).

    Several fit statistics are available to quantify how well a correlation represents experimental data (details for
    each are provided below). These statistics are unfortunately not consistently provided the literature.

        Root Mean Squared Error (RMSE):  Quadratic measure of average magnitude of error without considering direction.
        Useful for uncertainty propagation analysis (notice the functional form is similar to the standard deviation).

            RMSE = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5

        Mean Absolute Error (MAE):  Measure of average magnitude of error without considering direction.

            MAE = (1/n) * sum_i(abs(yi_meas - yi_model))

        Mean Absolute Percentage Error (MAPE): Measure of average magnitude of error without considering direction.

            MAPE = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))

    References
    ----------
    [1] Poling, B.E.; Praunitz, J.M.; O'Connell, J.P. The properties of gases and liquids, 5th ed.; McGraw-Hill, 2000.
    [2] Yaws, C. L. Thermophysical properties of chemicals and hydrocarbons, 2nd ed.; Gulf Professional Publishing is an
    imprint of Elsevier: Kidlington, Oxford, 2014.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    unit: str = 'N/m'
    t_unit: str = 'K'
    t_min: typing.Optional[float] = None
    t_max: typing.Optional[float] = None
    rmse: typing.Optional[float] = None
    mae: typing.Optional[float] = None
    mape: typing.Optional[float] = None
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def _t_conv(self, t: float) -> float:
        """Convert temperature from Kelvin to t_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[self.t_unit](t)

    def _corel(self, t: float) -> float:
        """General saturated liquid surface tension correlation."""
        return self.a * (1.0 - t / self.b) ** self.c

    def __call__(self, t: float) -> float:
        """Evaluate saturated liquid surface tension.

        Parameters
        ----------
        t : float
            Temperature for evaluation (in 'K').

        Returns
        -------
        float
            Saturated liquid viscosity evaluated at 't' (in 'N/m').
        """
        return units.to_si(self._corel(self._t_conv(t)), unit=self.unit)
