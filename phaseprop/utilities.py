"""Utility objects."""

import numpy as np
from units import *


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

# TODO: Build a base class & subclass w/ specific correlations.  Corel1, Corel2, Corel3, etc.
# TODO: Build w/ @dataclass
class Corel(object):
    """Pure component temperature dependent property."""
    def __init__(self, a=None, b=None, c=None, d=None, e=None, f=None, g=None, eq_id=None,
                 source_t_min=None, source_t_max=None, source_rmse=None, source_mae=None, source_mape=None,
                 source_t_unit=None, source_unit=None, source=None, notes=None):
        """
        Parameters
        ----------
        a : float or None
        b : float or None
        c : float or None
        d : float or None
        e : float or None
        f : float or None
        g : float or None
        t_min : float or None
        t_max : float or None
        eq_id : int
        source : str or None
        rmse : float or None
        mae : float or None
        mape : float or None
        """
        # call A-G hyperparameters in ML world...check on this.
        self.a = a  # a: float
        self.b = b  # b: float
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g

        # Property correlations.
        self._correlation = {1: lambda t: np.exp(self._a +
                                                 self._b / t +
                                                 self._c * np.log(t) +
                                                 self._d * t ** self._e),
                             2: lambda t: self._a / self._b ** (1.0 + (1.0 - t / self._c) ** self._d),
                             3: lambda t: self._a +
                                          self._b * self._tau(t, self._f) ** 0.35 +
                                          self._c * self._tau(t, self._f) ** (2.0 / 3.0) +
                                          self._d * self._tau(t, self._f) +
                                          self._e * self._tau(t, self._f) ** (4.0 / 3.0),
                             4: lambda t: self._a * (1.0 - self._tr(t, self._f)) ** (self._b +
                                                                                     self._c*self._tr(t, self._f) +
                                                                                     self._d*self._tr(t, self._f)**2.0 +
                                                                                     self._e*self._tr(t, self._f)**3.0),
                             5: lambda t: self._a +
                                          self._b * t +
                                          self._c * t ** 2.0 +
                                          self._d * t ** 3.0 +
                                          self._e * t ** 4.0,
                             6: lambda t: (self._a ** 2.0) / self._tau(t, self._e) +
                                           self._b -
                                           2.0 * self._a * self._c * self._tau(t, self._e) -
                                           self._a * self._d * self._tau(t, self._e) ** 2.0 -
                                           (self._c ** 2.0) * (self._tau(t, self._e) ** 3.0) / 3.0 -
                                           (self._c * self._d * self._tau(t, self._e) ** 4.0) / 2.0 -
                                           (self._d ** 2.0) * (self._tau(t, self._e) ** 5.0) / 5.0,
                             7: lambda t: self._a +
                                          self._b * ((self._c / t) / np.sinh(self._c / t)) ** 2.0 +
                                          self._d * ((self._e / t) / np.cosh(self._e / t)) ** 2.0,
                             8: lambda t: (self._a * t ** self._b) / (1.0 + self._c / t + self._d / t ** 2.0)}

        # d(correlation)/dt.
        self._derivative = {1: None,
                            2: None,
                            3: None,
                            4: None,
                            5: lambda t: self._b +
                                         self._c * t +
                                         self._d * t ** 2.0 +
                                         self._e * t ** 3.0,
                            6: None,
                            7: lambda t: 2.0 * ((self._b * self._c ** 2.0) *
                                                (self._c / np.tanh(self._c / t) - t) *
                                                ((1.0 / np.sinh(self._c / t)) ** 2.0) +
                                                (self._d * self._e ** 2.0) *
                                                (self._e * np.tanh(self._e / t) - t) *
                                                ((1.0 / np.cosh(self._e / t)) ** 2.0)) / (t ** 4.0),
                            8: None}

        # Integral(correlation.dt).  Useful for evaluating the Ideal Gas Enthalpy
        self._integral = {1: None,
                          2: None,
                          3: None,
                          4: None,
                          5: None,
                          6: None,
                          7: lambda t: self._a * t +
                                       self._b * self._c / np.tanh(self._c / t) -
                                       self._d * self._e * np.tanh(self._e / t),
                          8: None}

        # Integral(correlation.dt/t). Useful for evaluating the Ideal Gas Entropy.
        self._integral_t = {1: None,
                            2: None,
                            3: None,
                            4: None,
                            5: None,
                            6: None,
                            7: lambda t: self._a * np.log(t) +
                                         self._b * ((self._c/t) / np.tanh(self._c/t) - np.log(np.sinh(self._c/t))) -
                                         self._d * ((self._e/t) * np.tanh(self._e/t) - np.log(np.cosh(self._e/t))),
                            8: None}

        # Required parameters for each correlating equation. Keys are eq_id and value lists specify if parameters a-g
        # are required to evaluate the equation.
        self._req_parms = {1: [True, True, True, True, True, False, False],
                           2: [True, True, True, True, False, False, False],
                           3: [True, True, True, True, True, True, False],
                           4: [True, True, True, True, True, True, False],
                           5: [True, True, True, True, True, False, False],
                           6: [True, True, True, True, True, False, False],
                           7: [True, True, True, True, True, False, False],
                           8: [True, True, True, True, False, False, False]}

        self.eq_id = eq_id
        self.source_t_min = None # Initialize to None before assigning any values.
        self.source_t_max = None # Initialize to None before assigning any values.
        self.source_t_min = source_t_min
        self.source_t_max = source_t_max
        self.source_rmse = source_rmse
        self.source_mae = source_mae
        self.source_mape = source_mape
        self.source_t_unit = source_t_unit
        self.source_unit = source_unit
        self.source = source
        self.notes = notes

    @property
    def a(self):
        """float or None : Correlation constant."""
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            self._a = value
        elif isinstance(value, float):
            self._a = value
        else:
            raise TypeError("a must be a float.")

    @property
    def b(self):
        """float or None : Correlation constant."""
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            self._b = value
        elif isinstance(value, float):
            self._b = value
        else:
            raise TypeError("b must be a float.")

    @property
    def c(self):
        """float or None : Correlation constant."""
        return self._c

    @c.setter
    def c(self, value):
        if value is None:
            self._c = value
        elif isinstance(value, float):
            self._c = value
        else:
            raise TypeError("c must be a float.")

    @property
    def d(self):
        """float or None : Correlation constant."""
        return self._d

    @d.setter
    def d(self, value):
        if value is None:
            self._d = value
        elif isinstance(value, float):
            self._d = value
        else:
            raise TypeError("d must be a float.")

    @property
    def e(self):
        """float or None : Correlation constant."""
        return self._e

    @e.setter
    def e(self, value):
        if value is None:
            self._e = value
        elif isinstance(value, float):
            self._e = value
        else:
            raise TypeError("e must be a float.")

    @property
    def f(self):
        """float or None : Correlation constant."""
        return self._f

    @f.setter
    def f(self, value):
        if value is None:
            self._f = value
        elif isinstance(value, float):
            self._f = value
        else:
            raise TypeError("f must be a float.")

    @property
    def g(self):
        """float or None : Correlation constant."""
        return self._g

    @g.setter
    def g(self, value):
        if value is None:
            self._g = value
        elif isinstance(value, float):
            self._g = value
        else:
            raise TypeError("g must be a float.")

    @property
    def eq_id(self):
        """int : Equation identification number."""
        return self._eq_id

    @eq_id.setter
    def eq_id(self, value):
        if value is None:
            self._eq_id = None
        elif isinstance(value, int):
            if value in self._req_parms:
                self._eq_id = value
            else:
                raise ValueError("eq_id must correspond to a pre-defined equation.")
        else:
            raise TypeError("eq_id must be an int.")

    @property
    def source_t_min(self):
        """float or None : Minimum temperature in source temperature unit."""
        return self._source_t_min

    @source_t_min.setter
    def source_t_min(self, value):
        if value is None:
            self._source_t_min = value
        elif isinstance(value, float):
            if self._source_t_max is None:
                self._source_t_min = value
            elif self._source_t_max > value:
                self._source_t_min = value
            else:
                raise ValueError("source_t_min must be less than t_max.")
        else:
            raise TypeError("source_t_min must be a float.")

    @property
    def t_min(self):
        """float or None : Minimum temperature, K."""
        if self._source_t_min is None:
            return None
        else:
            return TEMPERATURE[self._source_t_unit](self._source_t_min)

    @property
    def source_t_max(self):
        """float or None : Maximum temperature in source temperature unit."""
        return self._source_t_max

    @source_t_max.setter
    def source_t_max(self, value):
        if value is None:
            self._source_t_max = value
        elif isinstance(value, float):
            if self._source_t_min is None:
                self._source_t_max = value
            elif self._source_t_min < value:
                self._source_t_max = value
            else:
                raise ValueError("t_max must be greater than t_min.")
        else:
            raise TypeError("t_max must be a float.")

    @property
    def t_max(self):
        """float or None : Minimum temperature, K."""
        if self._source_t_max is None:
            return None
        else:
            return TEMPERATURE[self._source_t_unit](self._source_t_max)

    @property
    def source_rmse(self):
        """float : Root mean squared error for correlated property in source unit.

        Quadratic measure of average magnitude of error without considering direction.  Useful for uncertainty
        propagation analysis (notice the functional form is similar to the standard deviation).

        mae = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5
        """
        return self._source_rmse

    @source_rmse.setter
    def source_rmse(self, value):
        if value is None:
            self._source_rmse = value
        elif isinstance(value, float):
            self._source_rmse = value
        else:
            raise TypeError("source_rmse must be a float.")

    @property
    def rmse(self):
        """float : Root mean squared error for correlated property in SI unit.

        Quadratic measure of average magnitude of error without considering direction.  Useful for uncertainty
        propagation analysis (notice the functional form is similar to the standard deviation).

        mae = ((1/n) * sum_i((yi_meas - yi_model)**2.0))**0.5
        """
        if self._source_rmse is None or self._source_unit is None:
            return None
        else:
            return conv_to_si(self._source_rmse, self._source_unit)

    @property
    def source_mae(self):
        """float : Mean absolute error for correlated property in source unit.

        Measure of average magnitude of error without considering direction.

        mae = (1/n) * sum_i(abs(yi_meas - yi_model))
        """
        return self._source_mae

    @source_mae.setter
    def source_mae(self, value):
        if value is None:
            self._source_mae = value
        elif isinstance(value, float):
            self._source_mae = value
        else:
            raise TypeError("source_mae must be a float.")

    @property
    def mae(self):
        """float : Mean absolute error for correlated property in SI unit.

        Measure of average magnitude of error without considering direction.

        mae = (1/n) * sum_i(abs(yi_meas - yi_model))
        """
        if self._source_mae is None or self._source_unit is None:
            return None
        else:
            return conv_to_si(self._source_mae, self._source_unit)

    @property
    def source_mape(self):
        """float : Mean absolute percentage error for correlated property in source unit.

        Measure of relative average magnitude of error without considering direction.

        mae = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))
        """
        return self._source_mape

    @source_mape.setter
    def source_mape(self, value):
        if value is None:
            self._source_mape = value
        elif isinstance(value, float):
            self._source_mape = value
        else:
            raise TypeError("source_mae must be a float.")

    @property
    def mape(self):
        """float : Mean absolute percentage error for correlated property in source unit.

        Measure of relative average magnitude of error without considering direction.

        mae = (1/n) * sum_i(abs((yi_meas - yi_model)/yi_meas)))
        """
        return self._source_mape

    @property
    def source_t_unit(self):
        """str : Source temperature unit for correlated property."""
        return self._source_t_unit

    @source_t_unit.setter
    def source_t_unit(self, value):
        if isinstance(value, str):
            if value in TEMPERATURE:
                self._source_t_unit = value
                return
            raise ValueError("source_t_unit is not defined.")
        else:
            raise TypeError("source_t_unit must be a string.")

    @property
    def source_unit(self):
        """str : Source unit for correlated property."""
        return self._source_unit

    @source_unit.setter
    def source_unit(self, value):
        if isinstance(value, str):
            for conv_dict in UNITS:
                if value in conv_dict:
                    self._source_unit = value
                    return
            raise ValueError("source_unit is not defined.")
        else:
            raise TypeError("source_unit must be a string.")

    @property
    def unit(self):
        """str : SI unit for correlated property."""
        for conv_dict in UNITS:
            if self._source_unit in conv_dict:
                return si_unit(conv_dict)
        raise ValueError("source_unit is not defined.")

    @property
    def source(self):
        """str : Source for the correlation (ACS citation format preferred)."""
        return self._source

    @source.setter
    def source(self, value):
        if value is None:
            self._source = value
        elif isinstance(value, str):
            self._source = value
        else:
            raise TypeError("source must be a string.")

    @property
    def notes(self):
        """str : Notes associated with the correlation."""
        return self._notes

    @notes.setter
    def notes(self, value):
        if value is None:
            self._notes = value
        elif isinstance(value, str):
            self._notes = value
        else:
            raise TypeError("notes must be a string.")

    def _t_conv(self, t, source_unit):
        """Convert temperature from Kelvin to source_unit."""
        conversion = {'F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
                      'R': lambda t: t * 1.8,
                      'C': lambda t: t - 273.15,
                      'K': lambda t: t}
        return conversion[source_unit](t)

    def _tr(self, t, tc):
        return t / tc

    def _tau(self, t, tc):
        return 1.0 - self._tr(t, tc)

    def t_in_range(self, t):
        if self.t_min <= t <= self.t_max:
            return True
        elif self.t_min is None and t <= self.t_max:
            return True
        elif self.t_max is None and t >= self.t_min:
            return True
        elif self.t_min is None and self.t_max is None:
            return True
        else:
            return False

    def defined(self):
        """Check to see if instance is fully defined.

        Returns
        -------
        bool
            True if instance is fully defined.
        """
        if self._eq_id is not None:
            a_def = self._a is not None
            b_def = self._b is not None
            c_def = self._c is not None
            d_def = self._d is not None
            e_def = self._e is not None
            f_def = self._f is not None
            g_def = self._g is not None
            parms_def = [a_def, b_def, c_def, d_def, e_def, f_def, g_def]
            req_parms_def = parms_def == self._req_parms[self._eq_id]
            source_unit_def = self._source_unit is not None
            correlation_def = self._correlation[self._eq_id] is not None
            return req_parms_def and source_unit_def and correlation_def
        else:
            return False

    def derivative(self, t=None, relax_range=False):
        """Derivative of the correlation.

        Parameters
        ----------
        t : float
            Temperature, K.
        relax_range : bool
            Allow evaluation outside of temperature range defined by t_min and t_max.

        Returns
        -------
        float
            Derivative evaluated at 't'.
        """
        if not isinstance(t, float):
            raise TypeError("t must be a float.")
        elif not self.defined():
            raise RuntimeError("Corel instance not fully defined.")
        elif self._derivative[self._eq_id] is None:
            raise RuntimeError("Derivative is not defined.")
        elif not self.t_in_range(t=t) and relax_range is False:
            raise ValueError("t must be inside the range defined by t_min and t_max.")
        else:
            return conv_to_si(self._derivative[self._eq_id](self._t_conv(t, self._source_t_unit)), self._source_unit)

    # Do you really need None here.  Is that reasonable for all correlations?  Or do you want to set float defaults?
    # None vs 0 or inf or ....etc.  Set this for each Corel class.  Just hard code this.
    def integral(self, t1=None, t2=None, relax_range=False):
        """Integral of the correlation.

        Parameters
        ----------
        t1 : float
            Lower temperature
        t2 : float
            Upper temperature
        relax_range : bool
            Allow evaluation outside of temperature range defined by t_min and t_max.

        Returns
        -------
        float
            Integral evaluated over the interval 't1' to 't2'.
        """
        if not isinstance(t1, float):
            raise TypeError("t1 must be a float.")
        elif not isinstance(t2, float):
            raise TypeError("t2 must be a float.")
        elif not self.defined():
            raise RuntimeError("Corel instance not fully defined.")
        elif self._integral[self._eq_id] is None:
            raise RuntimeError("Integral is not defined.")
        elif not self.t_in_range(t=t) and relax_range is False:
            raise ValueError("t must be inside the range defined by t_min and t_max.")
        else:
            return conv_to_si(self._integral[self._eq_id](self._t_conv(t2, self._source_t_unit)) -
                              self._integral[self._eq_id](self._t_conv(t1, self._source_t_unit)), self._source_unit)

    # TODO: t: float, etc....is a way to avoid type checking.
    def __call__(self, t=None, relax_range=False):
        """Evaluate the correlation.

        Parameters
        ----------
        t : float
            Temperature, K.
        relax_range : bool
            Allow evaluation outside of temperature range defined by t_min and t_max.

        Returns
        -------
        float
            Correlation evaluated at 't'.
        """
        if not isinstance(t, float):
            raise TypeError("t must be a float.")
        elif not self.defined():
            raise RuntimeError("Corel instance not fully defined.")
        elif not self.t_in_range(t=t) and relax_range is False:
            raise ValueError("t must be inside the range defined by t_min and t_max.")
        else:
            return conv_to_si(self._correlation[self._eq_id](self._t_conv(t, self._source_t_unit)), self._source_unit)

    def __eq__(self, other):
        if isinstance(other, Corel):
            a_eq = self._a == other.a
            b_eq = self._b == other.b
            c_eq = self._c == other.c
            d_eq = self._d == other.d
            e_eq = self._e == other.e
            f_eq = self._f == other.f
            g_eq = self._g == other.g
            t_min_eq = self._source_t_min == other.source_t_min
            t_max_eq = self._source_t_max == other.source_t_max
            eq_id_eq = self._eq_id == other.eq_id
            unit_eq = self._source_unit == other.source_unit
            t_unit_eq = self._source_t_unit == other.sourc_t_unit
            return a_eq and b_eq and c_eq and d_eq and e_eq and f_eq and g_eq and \
                   t_min_eq and t_max_eq and eq_id_eq and unit_eq and t_unit_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self._a, self._b, self._c, self._d, self._e, self._f, self._g,
                     self._t_min, self._t_max, self._eq_id))


class Var(object):
    """Variable with metadata.

    Notes
    -----
    TODO: Implement a float with units, uncertainty, source, and other useful metadata (similar to Corel class).
    """
