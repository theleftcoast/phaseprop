"""Objects required to describe a thermodynamic system."""
import copy
import numpy as np
from scipy.optimize import minimize
from utilities import Callback
from comps import CompSet
from spc_saft import sPCSAFT
from cpa import CPA


class State(object):
    """Intensive PVT and association state of a system.

    Methods
    -------
    set(p, vm, t)
        Set specification variables and trigger update for other dependent variables and objects.
    is_consistent(other)
        Check if two State objects have the same state definition.
    align_mixture()
        Sets State definition for all Phases to the State definition of the corresponding Mixture.
    defined()
        Check to see if instance is fully defined.

    Notes
    -----
    TODO: Generalize this class a bit?  A state can conceptually exist independently of a system.
    """

    def __init__(self, system=None, spec=None):
        """
        Parameters
        ----------
        system : Phase or Mixture object
        spec : str
        """
        if system is None:
            raise RuntimeError("Must pass an instance of Phase or Mixture to the constructor.")
        else:
            self.system = system
            self.spec = spec
            self._callback = Callback()
            self._p = None
            self._vm = None
            self._t = None
            self._xai = None

    @property
    def system(self):
        """Phase or Mixture object : The phase or mixture associated with a State object.

        Can only be set whe creating a new State instance."""
        return self._system

    @system.setter
    def system(self, value):
        try:
            self._system
        except AttributeError:
            if isinstance(value, (Phase, Mixture)):
                self._system = value
            else:
                raise TypeError("phase must be an instance of Phase or Mixture.")

    @property
    def spec(self):
        """str : Specification of either the 'PT' or 'VT' variable pair that specifies the system.

        Two state variables must be specified to define the PVT state of a system.  However, not all combinations are
        valid or supported. Only the most common pressure-temperature (PT) and volume-temperature(VT) specifications
        are currently implemented.
        """
        return self._spec

    @spec.setter
    def spec(self, value):
        if value is None:
            self._spec = None
            self._reset()
        elif value in ['PT', 'VT']:
            self._spec = value
            self._reset()
        else:
            raise ValueError("spec must be either PT or VT.")

    @property
    def callback(self):
        """Callback object : Manage functions from other classes that should be called if State changes."""
        return self._callback

    @property
    def p(self):
        """float : Pressure, Pa."""
        return self._p

    @property
    def vm(self):
        """float : Molar volume, m**3/mol."""
        return self._vm

    @property
    def t(self):
        """float : Temperature, K."""
        return self._t

    @property
    def xai(self):
        """list of lists : Fraction of sites not bonded for association sites in mixture.

        xai is a list of lists with the following structure:

            xai = [[xa for each site in Comp 1], [xa for each site in Comp 2], ..., [xa for each site in Comp n]]

        The length of xai is equal to the length of the associated CompSet.  The length of each element corresponds
        to the length of the assoc_sites attribute of each Comp in the associated CompSet.  If there are no association
        sites for a compound, then the corresponding element must have a truth value of False (which is represented
        by None this library for simplicity). If there are association sites for a compound, then the fraction of sites
        not bonded can take on a value of 0.0 <= xa <= 1.0.
        """
        return self._xai

    def set(self, p=None, vm=None, t=None):
        """Set specification variables and trigger update for other dependent variables and objects.

        This method takes 'p' & 't' and calculates 'vm' (for spec='PT') or takes 'vm' & 't' and calculates 'p' (for
        spec='VT'). Calculation of the associated dependent variable is done by the equation of state that is part of
        the associated system.  If the equation of state accounts for intermolecular association, then the resulting
        free site fraction estimate will be loaded into 'xai'. If the equation of state does not account for
        intermolecular association, then 'None' will be loaded into 'xai'.

        This method also manages callbacks to other objects that must be synchronized if State changes.

        Parameters
        ----------
        p : float or None
            Pressure, Pa.
        vm : float or None
            Molar volume, m**3/mol.
        t : float or None
            Temperature, K.
        """
        if self._spec is None:
            raise RuntimeError("spec must be initialized prior to setting p, vm, or t.")

        # Evaluate if each input is correctly defined.
        p_def = p is not None and isinstance(p, float) and p > 0.0
        vm_def = vm is not None and isinstance(vm, float) and vm > 0.0
        t_def = t is not None and isinstance(t, float) and t > 0.0

        # Set state variables.
        if self._spec == 'PT':
            # Check if state variables are correctly defined and proceed with update if so.
            if p_def:
                self._p = p
            if t_def:
                self._t = t

            # Update volume.
            if self._p is not None and self._t is not None and self._vm is not None:
                # Use prior volume as initial guess (time savings in vast majority of cases).
                # TODO: Not implemented yet. Need to define limits on how 'close' the prior state must be to justify
                #  using the prior value. A conservative suggestion is that 'p' and 't' are unchanged and the norm of
                #  the new and prior composition vector is less than some cutoff value (i.e. the composition only
                #  changed a little bit).  The norm of the composition change could be passed to State by the Compos
                #  callback.  This approach caters to the most common case where performance is critical...the 'PT
                #  Flash' calculation.  In this circumstance, 'p' and 't' remain unchanged with relatively small updates
                #  to composition at each iteration step.
                self._pt_vol_update(spec='full')
            elif self._p is not None and self._t is not None:
                # Using both liquid and vapor guesses. Choose volume with minimum gibbs energy.
                self._pt_vol_update(spec='full')
        elif self.spec == 'VT':
            # Check if state variables are correctly defined and proceed with update if so.
            if vm_def:
                self._vm = vm
            if t_def:
                self._t = t
            if self._vm is not None and self._t is not None:
                # TODO:  Implement pressure update.  VT specifiation and pressure update is low-priority and can wait
                #  until later.
                return  # TBD
            else:
                raise ValueError("vm and t must be positive floats.")
        else:
            raise RuntimeError("State specification not valid.")

    def _pt_vol_update(self, spec='full'):
        if spec == 'full' and self.defined():
            # Update volume using default liquid and vapor initial guesses.
            # TODO: Add functionality to pass back xai with vol_solver.
            vm_liq, gr_liq = self._system.eos.vol_solver(p=self._p, t=self._t, ni=self._system.compos.xi,
                                                         root='liquid')
            vm_vap, gr_vap = self._system.eos.vol_solver(p=self._p, t=self._t, ni=self._system.compos.xi,
                                                         root='vapor')
            # Choose volume with minimum residual gibbs energy.
            if gr_liq < gr_vap:
                self._vm = vm_liq
            else:
                self._vm = vm_vap
            return
        if spec == 'prior' and self.defined():
            # Update volume using prior value as initial guess.
            vm, gr = self._system.eos.vol_solver(p=self._p, t=self._.t, ni=self._system.compos.xi,
                                                 root='liquid', prior=self._vm)
            self._vm = vm
            return
        else:
            raise ValueError("spec must be either full or prior.")

    def _reset(self):
        self._p = None
        self._vm = None
        self._t = None
        self._xai = None

    def is_consistent(self, other):
        """Check if two State objects have the same state definition.

        This method does not check if two State objects are completely identical (every attribute is the same).  Rather,
        it checks if the state spec (either 'PT' or 'VT') and the corresponding specification variables ('p' & 't' or
        'vm' & 't') are identical.  This is useful because State objects may have the same definition, but different
        values for dependent variables (due to different equation of state methods) or no values for dependent variables
        (if no equation of state was specified).

        Parameters
        ----------
        other : State object
            The State object to be compared with 'self'.

        Returns
        -------
        bool
            True if 'other' is consistent with 'self'.
        """
        if isinstance(other, State):
            spec_eq = self._spec == other.spec
            if self._spec == 'PT':
                p_eq = self._p == other.p
                t_eq = self._t == other.t
                return spec_eq and p_eq and t_eq
            elif self._spec == 'VT':
                vm_eq = self._vm == other.vm
                t_eq = self._t == other.t
                return spec_eq and vm_eq and t_eq
            else:
                return False
        return False

    def align_mixture(self):
        """Sets State definition for all Phases to the State definition of the corresponding Mixture."""
        if isinstance(self._system, Mixture):
            for p in self._system.phases:
                if not self.is_consistent(p):
                    p.state.spec = self._spec
                    if self._spec == 'PT':
                        p.state.set(p=self._p, t=self._t)
                    elif self._spec == 'VT':
                        p.state.set(vm=self._vm, t=self._t)

    def defined(self):
        """Check to see if instance is fully defined.

        Returns
        -------
        bool
            True if instance is fully defined.
        """
        if self.spec == 'PT':
            p_def = self.p is not None
            t_def = self.t is not None
            return p_def and t_def
        elif self.spec == 'VT':
            vm_def = self.vm is not None
            t_def = self.t is not None
            return vm_def and t_def
        else:
            return False

    def __eq__(self, other):
        if isinstance(other, State):
            spec_eq = self.spec == other.spec
            p_eq = self.p == other.p
            t_eq = self.t == other.t
            vm_eq = self.vm == other.vm
            return spec_eq and p_eq and t_eq and vm_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.spec, self.p, self.vm, self.t))


class Extent(object):
    """Extent of a system (phase or mixture)."""

    def __init__(self, system=None):
        """
        Parameters
        ----------
        system : Phase or Mixture object
        """
        if system is None:
            raise ValueError("Must pass an instance of Phase or Mixture to the constructor.")
        else:
            self.system = system
            self.callback = Callback()
            self._ni = None
            self._n = None
            self._mi = None
            self._m = None
            self._v = None

    @property
    def system(self):
        """Phase or Mixture object : The phase or mixture associated with a State object.

        system can only be set whe creating a new State instance."""
        return self._system

    @system.setter
    def system(self, value):
        try:
            self._system
        except AttributeError:
            if isinstance(value, (Phase, Mixture)):
                self._system = value
            else:
                raise TypeError("system must be an instance of Phase or Mixture.")

    @property
    def ni(self):
        # Moles of each component.
        return self._ni

    @property
    def n(self):
        # Total moles.
        return self._n

    @property
    def mi(self):
        # Mass of each component (in kg).
        return self._mi

    @property
    def m(self):
        # Total mass (in kg).
        return self._m

    @property
    def v(self):
        # Units of volume are m**3.
        return self._v

    def set(self, n=None, m=None, v=None):
        # TODO: Add logic to update child phase extents if this is a mixture.
        # TODO: Add logic to callback parent mixture if this is a child phase.
        n_def = n is not None and isinstance(n, float) and n > 0.0
        m_def = m is not None and isinstance(m, float) and m > 0.0
        v_def = v is not None and isinstance(v, float) and v > 0.0

        if self._system.compos is None:
            compos_xi_def = False
            compos_wi_def = False
        else:
            compos_xi_def = self._system.compos.defined(spec='xi')
            compos_wi_def = self._system.compos.defined(spec='wi')

        if self._system.state is None:
            vm_def = False
        else:
            vm_def = self._state.vm is not None

        if n_def and not compos_xi_def:
            self._reset()
            self._n = n
        elif n_def and compos_xi_def:
            self._reset()
            self._xi_n_spec(n=n)
        elif m_def and not compos_wi_def:
            self._reset()
            self._m = m
        elif m_def and compos_wi_def:
            self._reset()
            self._wi_m_spec(m=m)
        elif v_def and not vm_def:
            self._reset()
            self._v = v
        elif v_def and vm_def:
            self._reset()
            self._vm_v_spec(v)
        else:
            raise RuntimeError("Invalid combination of specification variables.")

    def _xi_n_spec(self, n=None):
        if isinstance(n, float) and n >= 0.0:
            self._n = n
            self._ni = self._system.compos.xi * n
            if self._system.comps.mw is not None:
                # Convert grams into kilograms.
                self._mi = (self._ni * self._system.comps.mw) / 1000.0
                self._m = np.sum(self._mi)
        else:
            raise ValueError("n must be a positive float.")

    def _wi_m_spec(self, m=None):
        if isinstance(m, float) and m >= 0.0:
            self._m = m
            self._mi = self._system.compos.wi * m
            if self._system.comps.mw is not None:
                # Convert grams into kilograms.
                self._ni = 1000.0 * (self._mi / self._system.comps.mw)
                self._n = np.sum(self._ni)
        else:
            raise TypeError("m must be a positive float.")

    def _vm_v_spec(self, v=None):
        if isinstance(v, float) and v >= 0.0:
            raise TypeError("v must be a positive float.")
        else:
            self._n = v / self._system.state.vm
            if self._system.compos.defined(spec='xi'):
                self._ni = self._system.compos.xi * self._n
            if self._system.comps.mw is not None:
                # Convert grams into kilograms.
                self._mi = (self._ni * self._system.comps.mw) / 1000.0
                self._m = np.sum(self._mi)

    def _reset(self):
        self._ni = None
        self._n = None
        self._mi = None
        self._m = None
        self._v = None

    def _compos_callback(self):
        # Execute if Compos was updated.
        return

    def _mixture_extent_callback(self):
        # Execute if Mixture extent was updated.
        return

    def __eq__(self, other):
        if isinstance(other, Extent):
            system_eq = self.system == other.system
            ni_eq = self.ni == other.ni
            n_eq = self.n == other.n
            mi_eq = self.mi == other.mi
            m_eq = self.m == other.m
            v_eq = self.v == other.v
            return system_eq and ni_eq and n_eq and mi_eq and m_eq and v_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.system, self.ni, self.n, self.mi, self.m, self.v))

    def defined(self, spec=None):
        if spec is None or spec == 'full':
            if self._n is not None and self._m is not None and self._v is not None:
                return True
            else:
                return False
        elif spec == 'm':
            if self._m is not None:
                return True
            else:
                return False
        elif spec == 'n':
            if self._n is not None:
                return True
            else:
                return False
        elif spec == 'v':
            if self._v is not None:
                return True
            else:
                return False
        else:
            return False


class Compos(object):
    """Intensive composition of a system (phase or mixture).

    TODO: Add uncertainty for each compos. Could be used similarly to a Var for storing experimental data points.
    TODO: Generalize?  Does not have to be part of a system.  Could leave the initialization check for 'None' out and
     implement some safeguards elsewhere in the class.
    """

    def __init__(self, system=None):
        """
        Parameters
        ----------
        system : Phase or Mixture object
        """
        if system is None:
            raise ValueError("Must pass an instance of Phase or Mixture to the constructor.")
        else:
            self.system = system
            self._callback = Callback()
            self._xi = None
            self._wi = None
            if isinstance(system, Phase):
                self._phase_frac = None

    @property
    def system(self):
        """Phase or Mixture object : The phase or mixture associated with a Compos object.

        Can only be set whe creating a new State instance."""
        return self._system

    @system.setter
    def system(self, value):
        try:
            self._system
        except AttributeError:
            if isinstance(value, (Phase, Mixture)):
                self._system = value
            else:
                raise TypeError("phase must be an instance of Phase or Mixture.")

    @property
    def callback(self):
        """Callback object : Manage functions from other classes that should be called if state changes."""
        return self._callback

    @property
    def xi(self):
        """np.ndarray : Mole fraction of each component in phase or mixture."""
        return self._xi

    @property
    def wi(self):
        """np.ndarray : Mass fraction of each component in phase or mixture."""
        return self._wi

    @property
    def phase_frac(self):
        """float : Mole fraction of phase in a mixture.

        Only defined for Phases (not Mixtures) and is used extensively in phase equilibrium calculations.
        """
        return self._phase_frac

    @phase_frac.setter
    def phase_frac(self, value):
        if isinstance(self._system, Phase):
            if value is None:
                self._phase_frac = None
            elif isinstance(value, float) and 0.0 <= value <= 1.0:
                self._phase_frac = value
            else:
                raise TypeError("phase_frac must be a float between zero and one.")
        else:
            raise RuntimeError("phase_frac is only defined for Phases.")

    def set(self, xi=None, wi=None):
        """Set composition and trigger update for other dependent variables and objects.

        Parameters
        ----------
        xi : list, tuple, or np.ndarray
            Mole fraction for each component or pseudo-component in the associated CompSet.
        wi : list, tuple, or np.ndarray
            Mass fraction for each component or pseudo-component in the associated CompSet.
        """
        xi_def = xi is not None
        wi_def = wi is not None

        if xi_def is True and wi_def is False:
            self._reset()
            self._xi_spec(xi=xi)
        elif wi_def is True and xi_def is False:
            self._reset()
            self._wi_spec(wi=wi)
        else:
            raise ValueError("Must specify either xi or wi.")

    def _xi_spec(self, xi=None):
        if not isinstance(xi, (list, tuple, np.ndarray)):
            raise TypeError("xi must be a list, tuple, or np.ndarray.")
        elif not all(isinstance(i, (float, np.floating)) for i in xi):
            raise TypeError("xi can only contain floats.")
        elif not all(i >= 0.0 for i in xi):
            raise ValueError("xi can only contain positive floats.")
        elif len(xi) != self._system.comps.size:
            raise ValueError("xi must be the same size as the associated CompSet.")
        else:
            self._xi = np.array(xi) / np.sum(np.array(xi))
            if self._system.comps.mw is None:
                self._wi = None
                self._callback.fire()
            else:
                self._wi = (self._xi * self._system.comps.mw) / np.sum(self._xi * self._system.comps.mw)
                self._callback.fire()

    def _wi_spec(self, wi=None):
        if not isinstance(wi, (list, tuple, np.ndarray)):
            raise TypeError("wi must be a list, tuple, or np.ndarray.")
        elif not all(isinstance(i, (float, np.floating)) for i in wi):
            raise TypeError("wi can only contain floats.")
        elif not all(i >= 0.0 for i in wi):
            raise ValueError("wi can only contain positive floats.")
        elif len(wi) != self._system.comps.size:
            raise ValueError("wi must be the same size as the associated CompSet.")
        else:
            self._wi = np.array(wi) / np.sum(np.array(wi))
            if self._system.comps.mw is None:
                self._xi = None
                self._callback.fire()
            else:
                self._xi = (self._wi / self._system.comps.mw) / np.sum(self._wi / self._system.comps.mw)
                self._callback.fire()

    def _reset(self):
        self._xi = None
        self._wi = None

    def defined(self):
        """Check to see if instance is fully defined.

        Returns
        -------
        bool
            True if instance is fully defined.
        """
        if self._xi is not None and self._wi is not None:
            return True
        else:
            return False

    def __eq__(self, other):
        if isinstance(other, Compos):
            system_eq = self.system == other.system
            xi_eq = np.array_equal(self.xi, other.xi)
            wi_eq = np.array_equal(self.wi, other.wi)
            return system_eq and xi_eq and wi_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.system, self.xi, self.wi))


class Props(object):
    """Thermodynamic and transport properties of a system (phase or mixture)."""

    def __init__(self, system=None):
        """
        Parameters
        ----------
        system : Phase or Mixture object
        """
        if system is None:
            raise ValueError("Must pass an instance of Phase or Mixture to the constructor.")
        else:
            self.system = system
            self._a_r = None
            self._a_ig = None
            self._u_r = None
            self._u_ig = None
            self._h_r = None
            self._h_ig = None
            self._g_r = None
            self._g_ig = None
            self._s_r = None
            self._s_ig = None
            self._cv_r = None
            self._cv_ig = None
            self._cp_r = None
            self._cp_ig = None
            self._c = None
            self._z = None
            # TODO: Isothermal compressibility
            # TODO: Isentropic compressibility
            # TODO: Joule-Thompson coefficient
            self._phi = None
            self._ln_phi = None
            self._dln_phi_dt = None
            self._dln_phi_dp = None
            self._dln_phi_dni = None
            self._dp_dt = None
            self._dp_dv = None
            self._dp_dni = None
            self._dv_dni = None
            # TODO: Ideal gas viscosity
            # TODO: Ideal gas thermal conductivity
            # TODO: Viscosity
            # TODO: Thermal conductivity

    @property
    def system(self):
        """Phase or Mixture instance : The phase or mixture associated with a Prop object.

        Can only be set whe creating a new Comp instance.
        TODO:  Define Props object behavior for mixture later on.
        """
        return self._system

    @system.setter
    def system(self, value):
        try:
            self._system
        except AttributeError:
            if isinstance(value, (Phase)):
                self._system = value
            else:
                raise TypeError("system must be an instance of Phase.")

    @property
    def mw(self):
        """float or None : Molecular weight, g/mol.

        TODO: Remove mw from Comps?  It is nice to have MW available in comps.  Remove from Props?
        """
        if self._system.comps.mw is None or self._system.compos.xi is None:
            return None
        else:
            return np.sum(self._system.comps.mw * self._system.compos.xi)

    @property
    def rho_m(self):
        """float or None : Mass density, kg/m3"""
        if self.mw is None or self._system.state.vm is None:
            return None
        else:
            return self.mw / (1000.0 * self._system.state.vm)

    @property
    def a(self):
        """float or None : Helmholtz energy, ??"""
        if isinstance(self._a_r, float) and isinstance(self._a_ig, float):
            return self._a_r + self._a_ig
        else:
            return None

    @property
    def a_r(self):
        """float or None : Residual Helmholtz energy, ??"""
        return self._a_r

    @property
    def a_ig(self):
        """float or None : Ideal gas Helmholtz energy, ??"""
        return self._a_ig

    @property
    def u(self):
        """float or None : Internal energy, ??"""
        if isinstance(self._u_r, float) and isinstance(self._u_ig, float):
            return self._u_r + self._u_ig
        else:
            return None

    @property
    def u_r(self):
        """float or None : Residual internal energy, ??"""
        return self._u_r

    @property
    def u_ig(self):
        """float or None : Ideal gas internal energy, ??"""
        return self._u_ig

    @property
    def h(self):
        """float or None : Enthalpy, ??"""
        if isinstance(self._h_r, float) and isinstance(self._h_ig, float):
            return self._h_r + self._h_ig
        else:
            return None

    @property
    def h_r(self):
        """float or None : Residual enthalpy, ??"""
        return self._h_r

    @property
    def h_ig(self):
        """float or None : Ideal gas enthalpy, ??"""
        return self._h_ig

    @property
    def g(self):
        """float or None : Gibbs energy, ??"""
        if isinstance(self._g_r, float) and isinstance(self._g_ig, float):
            return self._g_r + self._g_ig
        else:
            return None

    @property
    def g_r(self):
        """float or None : Residual gibbs energy, ??"""
        return self._g_r

    @property
    def g_ig(self):
        """float or None : Ideal gas gibbs energy, ??"""
        return self._g_ig

    @property
    def s(self):
        """float or None : Entropy, ??"""
        if isinstance(self._s_r, float) and isinstance(self._s_ig, float):
            return self._s_r + self._s_ig
        else:
            return None

    @property
    def s_r(self):
        """float or None : Residual entropy, ??"""
        return self._s_r

    @property
    def s_ig(self):
        """float or None : Ideal gas entropy, ??"""
        return self._s_ig

    @property
    def cv(self):
        """float or None : Isochoric heat capacity, ??"""
        if isinstance(self._cv_r, float) and isinstance(self._cv_ig, float):
            return self._cv_r + self._cv_ig
        else:
            return None

    @property
    def cv_r(self):
        """float or None : Residual isochoric heat capacity, ??"""
        return self._cv_r

    @property
    def cv_ig(self):
        """float or None : Ideal gas isochoric heat capacity, ??"""
        return self._cv_ig

    @property
    def cp(self):
        """float or None : Isobaric heat capacity, ??"""
        if isinstance(self._cp_r, float) and isinstance(self._cp_ig, float):
            return self._cp_r + self._cp_ig
        else:
            return None

    @property
    def cp_r(self):
        """float or None : Residual isobaric heat capacity, ??"""
        return self._cp_r

    @property
    def cp_ig(self):
        """float or None : Ideal gas isobaric heat capacity, ??"""
        return self._cp_ig

    @property
    def z(self):
        """float or None : Compressibility factor, dimensionless"""
        return self._z

    @property
    def c(self):
        """float or None : Speed of sound, m/s"""
        return self._c

    @property
    def phi(self):
        """np.ndarray or None : Fugacity coefficient for each component in corresponding CompSet."""
        return self._phi

    @property
    def ln_phi(self):
        """np.ndarray or None : Natural log of the fugacity coefficient for each component in corresponding CompSet."""
        return self._ln_phi

    @property
    def dln_phi_dt(self):
        return self._dln_phi_dt

    @property
    def dln_phi_dp(self):
        return self._dln_phi_dp

    @property
    def dln_phi_dni(self):
        return self._dln_phi_dni

    @property
    def dp_dt(self):
        """float or None : Derivative of pressure with respect to temperature, Pa/K."""
        return self._dp_dt

    @property
    def dp_dv(self):
        """float or None : Derivative of pressure with respect to volume, ??."""
        return self._dp_dv

    @property
    def dp_dni(self):
        """float or None : Derivative of pressure with respect to mole numbers, ??."""
        return self._dp_dni

    @property
    def dv_dni(self):
        """float or None : Derivative of volume with respect to mole numbers, ??."""
        return self._dv_dni

    def update_ig_props(self):
        """Calculate ideal gas properties."""
        return

    def reset_ig_props(self):
        """Reset all ideal gas properties to None."""
        self._a_ig = None
        self._u_ig = None
        self._h_ig = None
        self._g_ig = None
        self._s_ig = None
        self._cv_ig = None
        self._cp_ig = None

    def update_resid_props(self):
        """Calculate residual properties."""
        return

    def reset_resid_props(self):
        """Reset residual properties to None."""
        self._a_r = None
        self._u_r = None
        self._h_r = None
        self._g_r = None
        self._s_r = None
        self._cv_r = None
        self._cp_r = None
        self._c = None
        self._z = None

    def update_michelsen_flash(self):
        self._ln_phi = self._system.eos.ln_phi(self._system.state.t,
                                               self._system.state.vm,
                                               self._system.compos.xi,
                                               self._system.state.xai)
        self._phi = np.exp(self._ln_phi)

    def reset_michelsen_flash(self):
        self._phi = None
        self._ln_phi = None

    def update_gibbs_minimization_flash(self):
        return

    def reset_gibbs_minimization_flash(self):
        self._dln_phi_dni = None

    def update_phase_envelope(self):
        return

    def reset_phase_envelope(self):
        self._dp_dt = None
        self._dp_dv = None
        self._dp_dni = None
        self._dv_dni = None
        self._dln_phi_dt = None
        self._dln_phi_dp = None

    def update_all(self):
        self._update_ig_props()
        self._update_resid_props()
        self._update_michelsen_flash()
        self._update_gibbs_minimization_flash()
        self._update_phase_envelope()

    def reset_all(self):
        self._reset_ig_props()
        self._reset_resid_props()
        self._reset_michelsen_flash()
        self._reset_gibbs_minimization_flash()
        self._reset_phase_envelope()


class Phase(object):
    """Pure phase with uniform properties.

    TODO: Consider generalizing. A Phase object doesn't have to have an EOS and can be used to store experimental data.
    """

    def __init__(self, comps=None, eos=None):
        """
        Parameters
        ----------
        comps : CompSet object
        eos : sPCSAFT or CPA object
        """
        if comps is None:
            raise ValueError("Must pass an instance of CompSet to the constructor.")
        else:
            self.comps = comps
            self.eos = eos
            self.state = State(system=self)
            self.compos = Compos(system=self)
            self.extent = Extent(system=self)
            self.props = Props(system=self)

    @property
    def comps(self):
        """CompSet object : Set of components or pseudo-components associated with Phase."""
        return self._comps

    @comps.setter
    def comps(self, value):
        try:
            self._comps
        except AttributeError:
            if isinstance(value, CompSet):
                self._comps = value
            else:
                raise TypeError("comps must be an instance of CompSet.")

    @property
    def eos(self):
        """sPCSAFT or CPA object or None : Equation of state associated with Phase."""
        return self._eos

    @eos.setter
    def eos(self, value):
        if value is None:
            self._eos = None
        elif isinstance(value, (sPCSAFT, CPA)):
            self._eos = value
        else:
            raise TypeError("eos must be an instance of sPC-SAFT or CPA.")

    @property
    def state(self):
        """State object : State of the Phase."""
        return self._state

    @state.setter
    def state(self, value):
        try:
            self._state
        except AttributeError:
            if isinstance(value, State):
                self._state = value
                self._state.callback.add(self._state_callback)
            else:
                raise TypeError("state must be an instance of State.")

    @property
    def compos(self):
        """Compos object : Intensive composition of the Phase."""
        return self._compos

    @compos.setter
    def compos(self, value):
        try:
            self._compos
        except AttributeError:
            if isinstance(value, Compos):
                self._compos = value
                self._compos.callback.add(self._compos_callback)
            else:
                raise TypeError("compos must be an instance of Compos.")

    @property
    def extent(self):
        """Extent object : Extensive composition of the Phase."""
        return self._extent

    @extent.setter
    def extent(self, value):
        try:
            self._extent
        except AttributeError:
            if isinstance(value, Extent):
                self._extent = value
            else:
                raise TypeError("extent must be an instance of Extent.")

    @property
    def props(self):
        """Props object : Thermodynamic and transport properties associated with Phase."""
        return self._props

    @props.setter
    def props(self, value):
        try:
            self._props
        except AttributeError:
            if isinstance(value, Props):
                self._props = value
            else:
                raise TypeError("props must be an instance of Props")

    def _compos_callback(self):
        # Execute if Compos was updated.
        # print("Compos was called back.")
        return

    def _state_callback(self, spec=None):
        # Execute if State was updated.
        # print("State was called back. Spec = {}".format(spec))
        return

    def _vt_pres_update(self, spec='full'):
        if spec == 'full' and self.defined():
            # Update volume using both liquid and vapor guesses.  Choose volume with minimum gibbs energy.
            return
        else:
            raise ValueError("spec must be either full or prior.")

    def defined(self):
        """Check to see if the instance is fully defined (all values available for EOS evaluation)

        Returns
        -------
        bool
            True if instance is fully defined.
        """
        comps_def = self.comps is not None
        compos_def = self.compos is not None and self.compos.defined()
        state_def = self.state is not None and self.state.defined()
        eos_def = self.eos is not None
        return comps_def and compos_def and state_def and eos_def

    def __eq__(self, other):
        if isinstance(other, Phase):
            comps_eq = self.comps == other.comps
            compos_eq = self.compos == other.compos
            state_eq = self.state == other.state
            eos_eq = True
            return comps_eq and compos_eq and state_eq and eos_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(())


class Mixture(object):
    """Mixture of phases.

    This class manages multiphase equilibrium calculations.
    """

    def __init__(self, comps=None, eos=None):
        """
        Parameters
        ----------
        comps : CompSet object
        eos : sPCSAFT or CPA object or None
        """
        if comps is None:
            raise ValueError("Must pass an instance of CompSet to the constructor.")
        else:
            self.comps = comps
            self.eos = eos
            self.state = State(system=self)
            self.compos = Compos(system=self)
            self.extent = Extent(system=self)
            self.props = Props(system=self)
            self.phases = []
            self._purgatory = []
            self._reference_phase = None
            self._trial_phase = Phase(comps=self._comps, eos=self._eos)

    @property
    def comps(self):
        """CompSet object : Set of components or pseudo-components associated with Mixture."""
        return self._comps

    @comps.setter
    def comps(self, value):
        try:
            self._comps
        except AttributeError:
            if isinstance(value, CompSet):
                self._comps = value
            else:
                raise TypeError("comps must be an instance of CompSet.")

    @property
    def eos(self):
        """sPCSAFT or CPA object or None : Equation of state associated with Mixture."""
        return self._eos

    @eos.setter
    def eos(self, value):
        if value is None:
            self._eos = None
        elif isinstance(value, (sPCSAFT, CPA)):
            self._eos = value
        else:
            raise TypeError("eos must be an instance of sPC-SAFT or CPA.")

    @property
    def state(self):
        """State object : State of the Mixture."""
        return self._state

    @state.setter
    def state(self, value):
        try:
            self._state
        except AttributeError:
            if isinstance(value, State):
                self._state = value
            else:
                raise TypeError("state must be an instance of State.")

    @property
    def compos(self):
        """Compos object : Intensive composition of the Mixture."""
        return self._compos

    @compos.setter
    def compos(self, value):
        try:
            self._compos
        except AttributeError:
            if isinstance(value, Compos):
                self._compos = value
            else:
                raise TypeError("compos must be an instance of Compos.")

    @property
    def extent(self):
        """Extent object : Extensive composition of the Mixture."""
        return self._extent

    @extent.setter
    def extent(self, value):
        try:
            self._extent
        except AttributeError:
            if isinstance(value, Extent):
                self._extent = value
            else:
                raise TypeError("extent must be an instance of Extent.")

    @property
    def props(self):
        """Props object : Thermodynamic and transport properties associated with the Mixture."""
        return self._props

    @props.setter
    def props(self, value):
        try:
            self._props
        except AttributeError:
            if isinstance(value, Props):
                self._props = value
            else:
                raise TypeError("props must be an instance of Props")

    @property
    def phases(self):
        """list of Phase objects : Phases in the mixture."""
        return self._phases

    @phases.setter
    def phases(self, value):
        try:
            self._phases
        except AttributeError:
            if isinstance(value, list):
                self._phases = value
            else:
                raise TypeError("phases must be a list.")

    def _remove_phase(self, index):
        """Move a phase with negative phase fraction into self._purgatory."""
        self._purgatory.append(self._phases.pop(index))

    def _copy_trial_to_phases(self):
        self._phases.append(copy.deepcopy(self._trial_phase))

    def _vle_stability_trial(self):
        """Build set of equilibrium ratios for stability testing based on Wilson's (Kw) & Raoult's Law correlation (Kr).

        The equilibrium ratio is defined as Ki = yi / xi.

        Returns
        -------
        list of np.ndarray
            Equilibrium ratios for stability testing, Kstab = [Kw, 1/Kw, Kw**0.333, 1/Kw**0.333].

        References
        ----------
        [1] Li, Z.; Firoozabadi, A. General Strategy for Stability Testing and Phase-Split Calculation in Two and Three
        Phases. SPE J. 2012, 17, 1096–1107.
        [2] Michelsen, M. L.; Mollerup, J. Thermodynamic Models: Fundamental and Computational Aspects, 2nd ed.;
        Tie-Line Publications: Holte, Denmark, 2007.
        """
        if self._state.defined():
            kw = []
            for c in self._comps.comps:
                kw.append(c.k_wilson(p=self._state.p, t=self._state.t))
            kw = np.array(kw)
            kw_inv = 1.0 / kw
            kw_cbrt = np.cbrt(kw)
            kw_cbrt_inv = np.cbrt(kw_inv)
            return kw, kw_inv, kw_cbrt, kw_cbrt_inv
        else:
            raise RuntimeError("Cannot evaluate because State object is not fully defined.")
        return

    def _lle_stability_trial(self):
        return

    def _di(self):
        """di vector for Michelsen's modified tangent plane distance.

        For a mixture containing 'i' components, the reference phase is used to calculate di as follows:

            di = ln(zi) + ln(phi(z))

        Returns
        -------
        np.ndarray
            di vector with shanpe (i, 1)
        """
        return self._phases[self._reference_phase].compos.xi + self._phases[self._reference_phase].props.ln_phi

    def _tm(self, wi=None):
        """Michelsen's modified tangent plane distance.

        For a mixture containing 'i' components, the modified tangent plane distance is defined as:

            tm(w) = 1 + sum_over_components(wi * (ln(wi) + ln(phi(wi)) - di - 1)

        Parameters
        ----------
        wi : np.ndarray
            Trial phase composition represented as a vector with shape (i, 1).

        Returns
        -------
        float
            Modified tangent plane distance.
        """
        # Update trial phase properties with new composition.
        self._trial_phase.compos.set(xi=wi)
        # TODO: Good time to make sure updating the composition triggers fugacity coefficient props evaluations.
        return 1.0 + np.sum(wi * (np.log(wi) + self._trial_phase.props.ln_phi - self._di() - 1.0))

    def _tm_i(self, wi):
        """Derivative of Michelsen's modified tangent plane distance with respect to mole numbers.

        For a mixture containing 'i' components, the derivative is defined as:

            d(tm(w))/d(wi) = ln(wi) + ln(phi(wi)) - di

        Parameters
        ----------
        wi : np.ndarray
            Trial phase composition represented as a vector with shape (i, 1).

        Returns
        -------
        np.ndarray
            Derivative of modified tangent plane distance with respect to mole numbers.
        """
        # Update trial phase properties with new composition.
        self._trial_phase.compos.set(xi=wi)
        # TODO: Good time to make sure updating the composition triggers fugacity coefficient props evaluations.
        return np.log(wi) + self._trial_phase.props.ln_phi - self._di()

    def _stability(self, wi):
        """Determine the stationary point of tm.

        Parameters
        ----------
        wi : np.ndarray
            Trial phase composition represented as a vector with shape (i, 1).

        Returns
        -------
        np.ndarray
            Phase composition that minimizes Michelsen's modified tangent plane distance.
        float
            Value of Michelsen's modified tangent plane distance at minimum.
        """
        minimum = minimize(fun=self._tm, x0=wi, method='BFGS', jac=self._tm_i)
        tm = self._tm(wi=minimum)
        return minimum, tm

    def _ei(self):
        """Ei vector for Michelsen's alternative flash.

        For a mixture containing 'i' components and 'k' phases, the E-vector is defined as follows:

            e[i] = sum_over_phases(beta[k] / phi[i, k])

        Parameters
        ----------
        beta : np.ndarray
            Phase fraction vector with shape (k, 1).
        phi : np.ndarray
            Fugacity coefficient matrix with shape (i, k)

        Returns
        -------
        np.ndarray
            E-vector with shape (i, 1).
        """
        beta = []
        for p in self._phases:
            p
        return

    def _q(self, z=None, beta=None, e=None):
        """Objective function for Michelsen's alternative flash.

        For a mixture containing 'i' components and 'j' phases, the Q-function is defined as follows:

            Q = sum_over_phases(beta[j]) - sum_over_components(z[i] * ln(e[i]))

        Parameters
        ----------
        z : np.ndarray
            Mixture composition vector with shape (i, 1).
        beta : np.ndarray
            Phase fraction vector with shape (j, 1).
        e : np.ndarray
            E-vector with shape (i, 1)

        Returns
        -------
        float
            Michelsen's Q-function.
        """
        return

    def _g(self, z=None, phi=None, e=None):
        """Gradient for Michelsen's alternative flash.

        For a mixture containing 'i' components and 'j' phases, the gradient is defined as follows:

            g[j] = 1.0 - sum_over_components(z[i] / (e[i] * phi[i, j]))

        Parameters
        ----------
        z : np.ndarray
            Mixture composition vector with shape (i, 1).
        phi : np.ndarray
            Fugacity coefficient matrix with shape (i, j).
        e : np.ndarray
            E-vector with shape (i, 1)

        Returns
        -------
        np.ndarray
            Gradient vector with shape (j, 1).
        """
        return

    def _pt_flash(self):
        """Evaluate the multiphase isothermal flash for a fluid mixture.

        This muiltiphase isothermal flash algorithm is based on Michelsen & Mollerup's approch as detailed in [1]_. The
        first step to solving the multiphase flash is to define a set of chemical components (CompSet object),
        composition (Compos object), state (State object), and equation of state (CPA or sPCSAFT object).

        The multiphase isothermal flash (at specified P and T) is solved using Michelsen's Q-function procedure [1, 2]_.
        In this approach, the


        The reference phase for stability and equilibrium calculations is taken to be the phase present in the largest
        amount.

        Nonlinear equation solver --> Broyden's method (only f required)
        Minimization --> BFGS (only f and f' required)

        References
        ----------
        [1] Michelsen, M. L.; Mollerup, J. Thermodynamic Models: Fundamental and Computational Aspects, 2nd ed.;
        Tie-Line Publications: Holte, Denmark, 2007.
        [2] Michelsen, M. L. Calculation of Multiphase Equilibrium. Computers Chem. Engn, 1994, 18, 545-550.
        [3] Li, Z.; Firoozabadi, A. General Strategy for Stability Testing and Phase-Split Calculation in Two and Three
        Phases. SPE J. 2012, 17, 1096–1107.
        """
        return

    def _vt_flash(self):
        """Driver for VT flash routine (to be developed later)."""
        return

    def defined(self):
        """Check to see if the instance is fully defined (all values available for multiphase flash evaluation)

        Returns
        -------
        bool
            True if instance is fully defined.
        """
        comps_def = self.comps is not None
        compos_def = self.compos is not None and self.compos.defined()
        state_def = self.state is not None and self.state.defined()
        eos_def = self.eos is not None
        return comps_def and compos_def and state_def and eos_def
