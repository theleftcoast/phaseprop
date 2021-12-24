"""Objects representing a generalized equation of state."""

import numpy as np
from const import R
from comp import Comp, PseudoComp


class EOS(object):
    """Base class implementation of an equation of state (EOS).

    Notes
    -----
    All thermodynamic properties are calculable as partial derivatives of the residual Helmholtz function.  This is
    fortunate as many popular equations of state are constructed as a series of residual Helmholtz terms.

    Ar(T, V, ni, xai) = A(T, V, ni, xai) - Aig(T, V, ni, xai)

    Note that this is the Helmholtz energy (Ar) rather than the reduced Helmholtz energy (ar).  Multiply 'ar' by the
    total moles to obtain 'Ar'. It is often easier in practice to take derivatives of the reduced residual Helmholtz
    function (F) which has units of mole.

    F = Ar(T, V, ni, xai)/(R*T)

    The methods in this class estimate thermodynamic properties by estimating reduced residual Helmholtz derivatives
    numerically. These numeric derivatives can be replaced with analytic derivatives by overriding parent class methods
    in any child classes. Viewed in this way, this class is intended to be a base class (or a template) used for
    implementing any EOS. Numeric derivatives and can be used to check user-developed analytic derivatives (useful for
    implementing unit testing).  Multivariate finite difference formulas taken from [1]_.

    All thermodynamic property expressions are available (along with derivations and context) in [2]_.

    References
    ----------
    [1] “Finite difference.” Wikipedia, Wikimedia Foundation, 25 Sept 2021, en.wikipedia.org/wiki/Finite_difference
    [2] Michelsen, M. L.; Mollerup, J. Thermodynamic Models: Fundamental and Computational Aspects, 2nd ed.; Tie-Line
    Publications: Holte, Denmark, 2007.
    """

    _step_size = {'f_v': 10.0 ** -6.0,
                  'f_t': 10.0 ** -6.0,
                  'f_i': 10.0 ** -6.0,
                  'f_ai': 10.0 ** -6.0}

    def _n(self, ni):
        # Dimension is moles.
        return np.sum(ni)

    def _f(self, t, v, ni, xai):
        # This function MUST be defined in any child class that inherits the EOS base class.
        pass

    def _func_t_num(self, func, t, v, ni, xai, dt):
        # Finite difference first partial derivative of func(t, v, ni, xai) with respect to 't'.
        dt = t * dt
        return (func(t + dt, v, ni, xai) - func(t - dt, v, ni, xai)) / (2.0 * dt)

    def _f_t(self, t, v, ni, xai):
        return self._func_t_num(self._f, t, v, ni, xai, self._step_size['f_t'])

    def _func_tt_num(self, func, t, v, ni, xai, dt):
        # Finite difference second partial derivative of func(t, v, ni, xai) with respect to 't'.
        dt = t * dt
        return (func(t + dt, v, ni, xai) - 2.0 * func(t, v, ni, xai) + func(t - dt, v, ni, xai)) / (dt ** 2.0)

    def _f_tt(self, t, v, ni, xai):
        return self._func_tt_num(self._f, t, v, ni, xai, self._step_size['f_t'])

    def _func_v_num(self, func, t, v, ni, xai, dv):
        # Finite difference first partial derivative of func(t, v, ni, xai) with respect to 'v'.
        dv = v * dv
        return (func(t, v + dv, ni, xai) - func(t, v - dv, ni, xai)) / (2.0 * dv)

    def _f_v(self, t, v, ni, xai):
        return self._func_v_num(self._f, t, v, ni, xai, self._step_size['f_v'])

    def _func_vv_num(self, func, t, v, ni, xai, dv):
        # Finite difference second partial derivative of func(t, v, ni, xai) with respect to 'v'.
        dv = v * dv
        return (func(t, v + dv, ni, xai) - 2.0 * func(t, v, ni, xai) + func(t, v - dv, ni, xai)) / (dv ** 2.0)

    def _f_vv(self, t, v, ni, xai):
        return self._func_vv_num(self._f, t, v, ni, xai, self._step_size['f_v'])

    def _func_tv_num(self, func, t, v, ni, xai, dt, dv):
        # Finite difference second partial derivative of func(t, v, ni, xai) with respect to 't' and 'v'.
        dv = v * dv
        dt = t * dt
        return (func(t + dt, v + dv, ni, xai) - func(t + dt, v - dv, ni, xai)
                - func(t - dt, v + dv, ni, xai) + func(t - dt, v - dv, ni, xai)) / (4.0 * dt * dv)

    def _f_tv(self, t, v, ni, xai):
        return self._func_tv_num(self._f, t, v, ni, xai, self._step_size['f_t'], self._step_size['f_v'])

    def _func_i_grad_num(self, func, t, v, ni, xai, di):
        # Finite difference first partial derivative of scalar func(t, v, ni, xai) with respect to 'i'.  This
        # implementation mirrors << https://rh8liuqy.github.io/Finite_Difference.html >>.  'ni' is a vector of mole
        # numbers for each comp in the associated CompSet.  Return object is a np.array representing the Gradient.
        ni = np.array(ni, dtype=float)
        di = np.sum(ni) * di
        n = len(ni)
        result = np.zeros(n)
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            f1 = func(t, v, ni + di * ei, xai)
            f2 = func(t, v, ni - di * ei, xai)
            result[i] = (f1 - f2) / (2.0 * di)
        result = result.reshape(n, 1)
        return result

    def _f_i(self, t, v, ni, xai):
        return self._func_i_grad_num(self._f, t, v, ni, xai, self._step_size['f_i'])

    def _func_ij_hess_num(self, func, t, v, ni, xai, di):
        # Finite difference second partial derivative of scalar func(t, v, ni, xai) with respect to 'i'.  This
        # implementation mirrors << https://rh8liuqy.github.io/Finite_Difference.html >>.  'ni' is a vector of mole
        # numbers for each comp in the associated CompSet.  Return object is a np.array representing the Hessian.
        ni = np.array(ni, dtype=float)
        di = np.sum(ni) * di
        n = len(ni)
        result = np.matrix(np.zeros(n * n))
        result = result.reshape(n, n)
        for i in range(n):
            for j in range(n):
                ei = np.zeros(n)
                ei[i] = 1.0
                ej = np.zeros(n)
                ej[j] = 1.0
                f1 = func(t, v, ni + di * ei + di * ej, xai)
                f2 = func(t, v, ni + di * ei - di * ej, xai)
                f3 = func(t, v, ni - di * ei + di * ej, xai)
                f4 = func(t, v, ni - di * ei - di * ej, xai)
                result[i, j] = (f1 - f2 - f3 + f4) / (4.0 * di * di)
        return result

    def _f_ij(self, t, v, ni, xai):
        return self._func_ij_hess_num(self._f, t, v, ni, xai, self._step_size['f_i'])

    def _f_vi(self, t, v, ni, xai):
        return self._func_i_grad_num(self._f_v, t, v, ni, xai, self._step_size['f_i'])

    def _f_ti(self, t, v, ni,  xai):
        return self._func_i_grad_num(self._f_t, t, v, ni, xai, self._step_size['f_i'])

    def ar(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.6
        # Ar(T, V, n) = R*T*F
        return R * t * self._f(t, v, ni, xai)

    def p(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.7
        # P = -R*T*(dF/dV) + n*R*T/V
        return -R * t * self._f_v(t, v, ni, xai) + self._n(ni) * R * t / v

    def z(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.8
        # Z = P*V/(n*R*T)
        return self.p(t, v, ni, xai) * v / (self._n(ni) * R * t)

    def p_v(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.9
        # dP/dV = -R*T*(d2F/dV2) - n*R*T/V**2
        return -R * t * self._f_vv(t, v, ni, xai) - self._n(ni) * R * t / (v ** 2.0)

    def p_t(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.10
        # dP/dT = -R*T*(d2F/dTdV) + P/T
        return -R * t * self._f_tv(t, v, ni, xai) + self.p(t, v, ni, xai) / t

    def p_i(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.11
        # dP/dni = -R*T*(d2F/dVdni) + R*T/V
        return -R * t * self._f_vi(t, v, ni, xai) + R * t / v

    def v_i(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.12
        # dV/dni = -(dP/dni)/(dP/dV)
        return -self.p_i(t, v, ni, xai) / self.p_v(t, v, ni, xai)

    def ln_phi(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.13
        # ln(phi_i) = (dF/dni) - ln(Z)
        # Note: ln(phi_i) estimated for all comps in the associated CompSet and returned as an np.array.
        return self._f_i(t, v, ni, xai) - np.log(self.z(t, v, ni, xai))

    def ln_phi_t(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.14
        # dln(phi_i)/dT = (d2F/dTdni) + 1.0/T - (dV/dni)*(dP/dT)/(R*T)
        return self._f_ti(t, v, ni, xai) + 1.0 / t - self.v_i(t, v, ni, xai) * self.p_t(t, v, ni, xai) / (R * t)

    def ln_phi_p(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.15
        # dln(phi_i)/dP = (dV/dni)/(R*T) - 1.0/P
        return self.v_i(t, v, ni, xai) / (R * t) - 1.0 / self.p(t, v, ni, xai)

    def _func_i_jac_num(self, func, t, v, ni, xai, di):
        # Finite difference first partial derivative of vector func(t, v, ni) with respect to 'i'.  This implementation
        # mirrors << https://rh8liuqy.github.io/Finite_Difference.html >>.  'ni' is a vector of mole numbers for each
        # comp in the associated CompSet.  Return object is a np.array representing the Jacobian.
        ni = np.array(ni, dtype=float)
        di = np.sum(ni) * di
        n_row = len(func(t, v, ni, xai))
        n_col = len(ni)
        result = np.zeros(n_row * n_col)
        result = result.reshape(n_row, n_col)
        for i in range(n_col):
            ei = np.zeros(n_col)
            ei[i] = 1.0
            f1 = func(t, v, ni + di * ei, xai)
            f2 = func(t, v, ni - di * ei, xai)
            result[:,i] = (f1 - f2) / (2.0 * di)
        return result

    def ln_phi_j(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.16 (numerically implemented here)
        #
        # This implementation estimates _ln_phi_j by numerically estimating the Jacobian of _ln_phi (a vector function).
        # _ln_phi is often analytically implemented in many equations of state.  This means that estimating _ln_phi_j by
        # numerical evaluation of the Jacobian of _ln_phi involves only first-order central difference approximations.
        # Estimating _ln_phi_j using Equation 2.16 involves estimating _f_ij using second-order central difference
        # approximations which is expected to degrade accuracy to some degree.
        return self._func_i_jac_num(self.ln_phi, t, v, ni, xai, self._step_size['f_i'])

    def sr(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.17
        # Sr(T, V, n) = -R*T*(dF/dT) - R*F
        return -R * t * self._f_t(t, v, ni, xai) - R * self._f(t, v, ni, xai)

    def cvr(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.18
        # Cvr(T, V, n) = -R*(T**2.0)*(d2F/dT2) - 2.0*R*T*(dF/dT)
        return -R * (t ** 2.0) * self._f_tt(t, v, ni, xai) - 2.0 * R * t * self._f_t(t, v, ni, xai)

    def cpr(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.19
        # Cpr(T, V, n) = -T*((dP/dT)**2)/(dP/dV) - n*R + Cvr(T, V, n)
        return -t * (self.p_t(t, v, ni, xai) ** 2.0) / self.p_v(t, v, ni, xai) - self._n(ni) * R + \
               self.cvr(t, v, ni, xai)

    def hr(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.20
        # Hr(T, P, n) = Ar(T, V, n) + T*Sr(T, V, n) + P*V - n*R*T
        return self.ar(t, v, ni, xai) + t * self.sr(t, v, ni, xai) + self.p(t, v, ni, xai) * v - self._n(ni) * R * t

    def gr(self, t, v, ni, xai):
        # Michelsen & Mollerup (2007), Equation 2.21
        # Gr(T, P, n) = Ar(T, V, n) + P*V - n*R*T - n*R*T*ln(Z)
        return self.ar(t, v, ni, xai) + self.p(t, v, ni, xai) * v - self._n(ni) * R * t - \
               self._n(ni) * R * t * np.log(self.z(t, v, ni, xai))


class BinaryInterParm(object):
    """Binary interaction parameter between components or pseudo-components.
        k_ij(T) = k_ij + a*T + b/T + c*ln(T)
    """

    def __init__(self, comp_a=None, comp_b=None, source=None,
                 temp_indep_coef=None, lin_temp_coef=None, inv_temp_coef=None, ln_temp_coef=None):
        if comp_a is None or comp_b is None:
            raise ValueError("comp_a and comp_b must be provided to create an instance of BinaryInterParm.")
        else:
            self.comp_a = comp_a
            self.comp_b = comp_b
            self.source = source
            self.temp_indep_coef = temp_indep_coef
            self.lin_temp_coef = lin_temp_coef
            self.inv_temp_coef = inv_temp_coef
            self.ln_temp_coef = ln_temp_coef

    @property
    def comp_a(self):
        return self._comp_a

    @comp_a.setter
    def comp_a(self, value):
        try:
            self._comp_a
        except AttributeError:
            if isinstance(value, (Comp, PseudoComp)):
                self._comp_a = value
            else:
                raise TypeError("comp_a must be an instance of Comp or PseudoComp.")

    @property
    def comp_b(self):
        return self._comp_b

    @comp_b.setter
    def comp_b(self, value):
        try:
            self._comp_b
        except AttributeError:
            if isinstance(value, (Comp, PseudoComp)):
                self._comp_b = value
            else:
                raise TypeError("comp_b must be an instance of Comp or PseudoComp.")

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        if value is None:
            self._source = value
        elif isinstance(value, str):
            self._source = value
        else:
            raise ValueError("source must be a string.")

    @property
    def temp_indep_coef(self):
        return self._temp_indep_coef

    @temp_indep_coef.setter
    def temp_indep_coef(self, value):
        if value is None:
            self._temp_indep_coef = value
        elif isinstance(value, float):
            self._temp_indep_coef = value
        else:
            raise ValueError("temp_indep_coef must be a float.")

    @property
    def lin_temp_coef(self):
        return self._lin_temp_coef

    @lin_temp_coef.setter
    def lin_temp_coef(self, value):
        if value is None:
            self._lin_temp_coef = value
        elif isinstance(value, float):
            self._lin_temp_coef = value
        else:
            raise ValueError("lin_temp_coef must be a float.")

    @property
    def inv_temp_coef(self):
        return self._inv_temp_coef

    @inv_temp_coef.setter
    def inv_temp_coef(self, value):
        if value is None:
            self._inv_temp_coef = value
        elif isinstance(value, float):
            self._inv_temp_coef = value
        else:
            raise ValueError("inv_temp_coef must be a float.")

    @property
    def ln_temp_coef(self):
        return self._ln_temp_coef

    @ln_temp_coef.setter
    def ln_temp_coef(self, value):
        if value is None:
            self._ln_temp_coef = value
        elif isinstance(value, float):
            self._ln_temp_coef = value
        else:
            raise ValueError("ln_temp_coef must be a float.")

    def __eq__(self, other):
        if isinstance(other, BinaryInterParm):
            aa_bb_eq = self.comp_a == other.comp_a and self.comp_b == other.comp_b
            ab_ba_eq = self.comp_a == other.comp_b and self.comp_b == other.comp_a
            return aa_bb_eq or ab_ba_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.comp_a), hash(self.comp_b)))

    def __str__(self):
        return "Comp A: ({}), Comp B: ({}), k_ij: {}".format(self.comp_a,
                                                             self.comp_b,
                                                             self.k_ij)

    # TODO: Include a defined flag for all other classes.
    def defined(self):
        # A binary interaction parameter is considered defined if any of the coefficients are not none.
        tic = self.temp_indep_coef is not None
        ltc = self.lin_temp_coef is not None
        itc = self.inv_temp_coef is not None
        lntc = self.ln_temp_coef is not None
        return tic or ltc or itc or lntc

    def k_ij(self, t):
        if t <= 0.0:
            raise ValueError("The temperature cannot be less than zero.")

        ti = self.temp_indep_coef if (self.temp_indep_coef is not None) else 0.0
        lt = self.lin_temp_coef * t if (self.lin_temp_coef is not None) else 0.0
        it = self.inv_temp_coef / t if (self.inv_temp_coef is not None) else 0.0
        lnt = self.ln_temp_coef * np.log(t) if (self.ln_temp_coef is not None) else 0.0
        return ti + lt + it + lnt
