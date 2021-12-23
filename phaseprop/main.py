"""Multiphase equilibrium and thermophysical property estimation.

Attributes
----------
GS : instance of PCSAFTSpec
    PC-SAFT universal constants (Gross and Sadowski, [2, 3]_).
PR : instance of CubicSpec
    Peng-Robinson equation of state (Peng and Robinson, [4]_).
GPR : instance of CubicSpec
    Peng-Robinson equation of state with the Gasem alpha function (Gasem et al., [5]_).
TPR : instance of CubicSpec
    Peng-Robinson equation of state with the Twu alpha function (Twu et al., [6]_).
SRK : instance of CubicSpec
    Soave-Redlich-Kwong equation of state (Soave, [7]_).
CPA : instance of CubicSpec
    Cubic-plus-association equation of state (Kontogeorgis, [8, 9]_).
REFERENCES : dict
    Dictionary of references in ACS style.

Notes
-----


References
----------
[2] Gross, J; Sadowski, G. Perturbed-Chain SAFT:  An Equation of State Based on a Perturbation Theory for Chain
Molecules. Ind. Eng. Chem. Res. 2001, 40, 1244–1260.
[3] Gross, J.; Sadowski, G. Application of the Perturbed-Chain SAFT Equation of State to Associating Systems. Ind. Eng.
Chem. Res. 2002, 41, 5510-5515.
[4] Peng, D. Y.; Robinson, D. B. A New Two-Constant Equation of State. Ind. Eng. Chem. Fundamen. 1976, 15, 59–64.
[5] Gasem, K. A. M.; Gao, W.; Pan, Z.; Robinson R. L. A modified temperature dependence for the Peng-Robinson equation
of state. Fluid Phase Equilib. 2001, 181, 113-125.
[6] Twu, C. H.; Bluck, D.; Cunningham, J. R.; Coon, J. E. A cubic equation of state with a new alpha function and a new
mixing rule. Fluid Phase Equilib. 1991, 69, 33-50.
[7] Soave, G. Equilibrium constants from a modified Redlich-Kwong equation of state. Fluid Phase Equilb. 1972, 27,
1197-1203.
[8] Kontogeorgis, G. M.; Voutsas, E. C.; Yakoumis, I. V.; Tassios, D. P. An Equation of State for Associating Fluids.
Ind. Eng. Chem. Res. 1996, 35, 4310-4318.
[9] Kontogeorgis, G. M.; Yakoumis, I. V.; Meijer, H.; Hendriks, E.; Moorwood, T. Multicomponent phase equilibrium
calculations for water–methanol–alkane mixtures. Fluid Phase Equilib. 1999, 158-160, 201-209.
"""

import numpy as np
from scipy.optimize import minimize
import copy
from assoc import *

# TODO: Check that value errors and type errors are raised consistently.
# TODO: Consider eliminating pre-defined EOS in favor of identification using spec objects instead.
# Pre-defined equation of state (EOS) pick lists.
CUBIC_EOS = ['PR', 'GPR', 'TPR', 'SRK', 'CPA']
PC_SAFT_EOS = ['PC-SAFT', 'sPC-SAFT']
PHYS_EOS = CUBIC_EOS + PC_SAFT_EOS
ASSOC_EOS = ['CPA', 'PC-SAFT', 'sPC-SAFT']





class PCSAFTSpec(object):
    """Constants that define a specific version of the PC-SAFT equation of state."""

    def __init__(self, a=None, b=None):
        """
        Parameters
        ----------
        a : np.ndarray
            2D array with shape (7, 3) containing data of 'float' type.
        b : np.ndarray
            2D array with shape (7, 3) containing data of 'float' type.
        """
        if a is None or b is None:
            raise ValueError("Inputs are not sufficient to define an instance of sPCSAFTSpec.")
        else:
            self.a = a
            self.b = b

    @property
    def a(self):
        """Universal model constants for the first-order perturbation dispersion term."""
        return self._a

    @a.setter
    def a(self, value):
        try:
            self._a
        except AttributeError:
            if isinstance(value, np.ndarray) and value.shape == (7, 3):
                self._a = value
            else:
                raise ValueError("a must be an np.ndarray with shape (7, 3).")

    @property
    def b(self):
        """Universal model constants for the second-order perturbation dispersion term."""
        return self._b

    @b.setter
    def b(self, value):
        try:
            self._b
        except AttributeError:
            if isinstance(value, np.ndarray) and value.shape == (7, 3):
                self._b = value
            else:
                raise ValueError("a must be an np.ndarray with shape (7, 3).")

    def __eq__(self, other):
        if isinstance(other, PCSAFTSpec):
            a_eq = np.array_equal(self.a, other.a)
            b_eq = np.array_equal(self.b, other.b)
            return a_eq and b_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.a, self.b))

    def __str__(self):
        return "a: {} \nb: {}".format(self.a, self.b)


# PC-SAFT universal constants (Gross and Sadowski).
gs_a0 = [0.9105631445, -0.3084016918, -0.0906148351]
gs_a1 = [0.6361281449, 0.1860531159, 0.4527842806]
gs_a2 = [2.6861347891, -2.5030047259, 0.5962700728]
gs_a3 = [-26.547362491, 21.419793629, -1.7241829131]
gs_a4 = [97.759208784, -65.25588533, -4.1302112531]
gs_a5 = [-159.59154087, 83.318680481, 13.77663187]
gs_a6 = [91.297774084, -33.74692293, -8.6728470368]
gs_a = np.array([gs_a0, gs_a1, gs_a2, gs_a3, gs_a4, gs_a5, gs_a6])

gs_b0 = [0.7240946941, -0.5755498075, 0.0976883116]
gs_b1 = [2.2382791861, 0.6995095521, -0.2557574982]
gs_b2 = [-4.0025849485, 3.892567339, -9.155856153]
gs_b3 = [-21.003576815, -17.215471648, 20.642075974]
gs_b4 = [26.855641363, 192.67226447, -38.804430052]
gs_b5 = [206.55133841, -161.82646165, 93.626774077]
gs_b6 = [-355.60235612, -165.20769346, -29.666905585]
gs_b = np.array([gs_b0, gs_b1, gs_b2, gs_b3, gs_b4, gs_b5, gs_b6])

GS = PCSAFTSpec(a=gs_a, b=gs_b)

# TODO:  Move this into its own module and make each dictionary entry a variable in the module.
# TODO:  Stick all references into a module while you are at it.
REFERENCES = {'DIPPR': "Rowley, R. L.; Wilding, W. V.; Oscarson, J. L.; Knotts, T. A.; Giles, N. F. DIPPR Data "
                       "Compilation of Pure Chemical Properties; Design Institute for Physical Properties, AIChE: New "
                       "York, NY, 2016.",
              'YAWS': "Yaws, C. L. Thermophysical properties of chemicals and hydrocarbons, 2nd ed., Gulf Professional "
                      "Publishing: Waltham, MA, 2014."}


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


class BinaryInterParm(object):
    """Binary interaction parameter between components or pseudo-components.
        k_ij(T) = k_ij + a*T + b/T + c*ln(T)
    """

    def __init__(self, comp_a=None, comp_b=None, phys_eos_spec=None, eos=None, source=None,
                 temp_indep_coef=None, lin_temp_coef=None, inv_temp_coef=None, ln_temp_coef=None):
        if comp_a is None or comp_b is None or eos is None:
            raise ValueError("comp_a, comp_b, phys_eos_spec, and eos must be provided to create an instance of "
                             "BinaryInterParm.")
        else:
            self.comp_a = comp_a
            self.comp_b = comp_b
            self.phys_eos_spec = phys_eos_spec
            self.eos = eos
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
    def phys_eos_spec(self):
        return self._phys_eos_spec

    @phys_eos_spec.setter
    def phys_eos_spec(self, value):
        try:
            self._phys_eos_spec
        except AttributeError:
            if isinstance(value, (PCSAFTSpec, CubicSpec)):
                self._phys_eos_spec = value
            else:
                raise TypeError("phys_eos_spec must be an instance of PCSAFTSpec or CubicSpec.")

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, value):
        try:
            self._eos
        except AttributeError:
            if value in PHYS_EOS:
                self._eos = value
            else:
                raise ValueError("eos not valid.")

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
            pes_eq = self.phys_eos_spec == other.phys_eos_spec
            eos_eq = self.eos == other.eos
            return (aa_bb_eq or ab_ba_eq) and pes_eq and eos_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.comp_a), hash(self.comp_b), self.eos))

    def __str__(self):
        # TODO: Add phys_eos_spec somehow.
        return "Comp A: ({}), Comp B: ({}), EOS: {}, k_ij: {}".format(self.comp_a,
                                                                      self.comp_b,
                                                                      self.eos,
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


class CubicParms(object):
    """Cubic equation of state parameters"""


class PCSAFTParms(object):
    def __init__(self, comp=None, pc_saft_spec=None, source=None,
                 seg_num=None, seg_diam=None, disp_energy=None, ck_const=0.12):
        if comp is None or pc_saft_spec is None:
            raise ValueError("comp and pc_saft_spec must be provided to create an instance of PCSAFTParms.")
        else:
            self.comp = comp
            self.pc_saft_spec = pc_saft_spec
            self.source = source
            self.seg_num = seg_num
            self.seg_diam = seg_diam
            self.disp_energy = disp_energy
            self.ck_const = ck_const

    @property
    def comp(self):
        return self._comp

    @comp.setter
    def comp(self, value):
        try:
            self._comp
        except AttributeError:
            if isinstance(value, (Comp, PseudoComp)):
                self._comp = value
            else:
                raise TypeError("comp must be an instance of Comp or PseudoComp.")

    @property
    def pc_saft_spec(self):
        return self._pc_saft_spec

    @pc_saft_spec.setter
    def pc_saft_spec(self, value):
        try:
            self._pc_saft_spec
        except AttributeError:
            if isinstance(value, PCSAFTSpec):
                self._pc_saft_spec = value
            else:
                raise TypeError("pc_saft_spec must be an instance of PCSAFTSpec.")

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
    def seg_num(self):
        return self._seg_num

    @seg_num.setter
    def seg_num(self, value):
        if value is None:
            self._seg_num = value
        elif isinstance(value, float):
            self._seg_num = value
        else:
            raise ValueError("seg_num must be a float.")

    @property
    def seg_diam(self):
        return self._seg_diam

    @seg_diam.setter
    def seg_diam(self, value):
        if value is None:
            self._seg_diam = value
        elif isinstance(value, float):
            self._seg_diam = value
        else:
            raise ValueError("seg_diam must be a float.")

    @property
    def disp_energy(self):
        return self._disp_energy

    @disp_energy.setter
    def disp_energy(self, value):
        if value is None:
            self._disp_energy = value
        elif isinstance(value, float):
            self._disp_energy = value
        else:
            raise ValueError("disp_energy must be a float.")

    @property
    def ck_const(self):
        # Chen and Kreglewski temperature-dependent integral parameter is 0.12 for nearly all compounds. However, it can
        # be set to 0.241 for hydrogen (see eq 2-6 in de Villers' PhD thesis and eq 2-10 in Tihic's PhD thesis). This
        # is a useful correction for the quantum gases (hydrogen and helium).
        return self._ck_const

    @ck_const.setter
    def ck_const(self, value):
        if value is None:
            self._ck_const = value
        elif isinstance(value, float):
            self._ck_const = value
        else:
            raise ValueError("ck_const must be a float.")

    def __eq__(self, other):
        if isinstance(other, PCSAFTParms):
            comp_eq = self.comp == other.comp
            spec_eq = self.pc_saft_spec == other.pc_saft_spec
            return comp_eq and spec_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.comp), hash(self.pc_saft_spec)))

    def defined(self):
        sn = self.seg_num is not None
        sd = self.seg_diam is not None
        de = self.disp_energy is not None
        ck = self.ck_const is not None
        return sn and sd and de and ck

    def init_parms(self, comp_family='alkane'):
        # Initialize pure component parameters as a starting point for optimization routines.
        #
        # PC-SAFT parameter correlations taken from the following sources:
        #   (1) Tihic, A.; Kontogeorgis, G.M.; von Solms, N.; Michelsen, M.L. Applications of the simplified perturbed-
        #   chain SAFT equation of state using an extended parameter table. Fluid Phase Equilib. 2006, 248, 29-43.
        #
        # PC-SAFT parameters for n-hexane taken from the following source:
        #   (2) Gross, J.; Sadowski, G. Perturbed-Chain SAFT: An Equation of State Based on a Perturbation Theory for
        #   Chain Molecules.  Ind. Eng. Chem. Res. 2001, 40, 1244–1260.
        #
        # TODO: Define parameter estimation correlations based on component family.
        if self._comp.mw is not None:
            self._seg_num = 0.0249 * self._comp.mw + 0.9711
            self._seg_diam = ((1.6947 * self._comp.mw + 23.27) / self._seg_num) ** 0.33333
            self._disp_energy = (6.5446 * self._comp.mw + 177.92) / self._seg_num
        else:
            self._seg_num = 3.0576
            self._seg_diam = 3.7983
            self._disp_energy = 236.77


class PCSAFTPhysInter(object):
    """Physical interactions between components."""

    def __init__(self, comps=None, eos=None, pc_saft_spec=None, adj_pure_comp_spec=None, adj_binary_spec=None):
        if comps is None or eos is None or pc_saft_spec is None:
            raise ValueError("comps, eos, and pc_saft_spec must be provided to create an instance of PCSAFTPhysInter.")
        self.comps = comps
        self.eos = eos
        self.pc_saft_spec = pc_saft_spec
        self.pure_comp_parms = self._create_pure_comp_parms()
        self.binary_parms = self._create_binary_parms()
        self._adj_seg_num_members = []
        self._adj_seg_diam_members = []
        self._adj_disp_energy_members = []
        self._adj_temp_indep_coef_members = []
        self._adj_lin_temp_coef_members = []
        self._adj_inv_temp_coef_members = []
        self._adj_ln_temp_coef_members = []
        self.set_adj_pure_comp_parms(adj_pure_comp_spec=adj_pure_comp_spec)
        self.set_adj_binary_parms(adj_binary_spec=adj_binary_spec)

    @property
    def comps(self):
        return self._comps

    @comps.setter
    def comps(self, value):
        try:
            self._comps
        except AttributeError:
            if not isinstance(value, CompSet):
                raise TypeError("Component must be an instance of CompSet.")
            self._comps = value

    @property
    def eos(self):
        # TODO: consider droping the EOS spec...what is it used for anyway?
        return self._eos

    @eos.setter
    def eos(self, value):
        try:
            self._eos
        except AttributeError:
            if value in PC_SAFT_EOS:
                self._eos = value
            else:
                raise ValueError("eos not valid.")

    @property
    def pc_saft_spec(self):
        return self._pc_saft_spec

    @pc_saft_spec.setter
    def pc_saft_spec(self, value):
        try:
            self._pc_saft_spec
        except AttributeError:
            if isinstance(value, PCSAFTSpec):
                self._pc_saft_spec = value
            else:
                raise TypeError("pc_saft_spec not an instance of PCSAFTSpec.")

    @property
    def pure_comp_parms(self):
        return self._pure_comp_parms

    @pure_comp_parms.setter
    def pure_comp_parms(self, value):
        try:
            self._pure_comp_parms
        except AttributeError:
            self._pure_comp_parms = value

    @property
    def binary_parms(self):
        return self._binary_parms

    @binary_parms.setter
    def binary_parms(self, value):
        try:
            self._binary_parms
        except AttributeError:
            self._binary_parms = value

    @property
    def adj_seg_num_members(self):
        return self._adj_seg_num_members

    @property
    def adj_seg_num(self):
        values = []
        for member in self._adj_seg_num_members:
            values.append(member.seg_num)
        return values

    @adj_seg_num.setter
    def adj_seg_num(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_seg_num_members):
            for i, member in enumerate(self._adj_seg_num_members):
                if isinstance(value[i], float):
                    member.seg_num = value[i]
                else:
                    raise ValueError("adj_seg_num values must be floats.")
        else:
            raise TypeError("adj_seg_num must be a list or tuple with {} elements."
                            .format(len(self._adj_seg_num_members)))

    @property
    def adj_seg_diam_members(self):
        return self._adj_seg_diam_members

    @property
    def adj_seg_diam(self):
        values = []
        for member in self._adj_seg_diam_members:
            values.append(member.seg_diam)
        return values

    @adj_seg_diam.setter
    def adj_seg_diam(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_seg_diam_members):
            for i, member in enumerate(self._adj_seg_diam_members):
                if isinstance(value[i], float):
                    member.seg_diam = value[i]
                else:
                    raise ValueError("adj_seg_diam values must be floats.")
        else:
            raise TypeError("adj_seg_diam must be a list or tuple with {} elements."
                            .format(len(self._adj_seg_diam_members)))

    @property
    def adj_disp_energy_members(self):
        return self._adj_disp_energy_members

    @property
    def adj_disp_energy(self):
        values = []
        for member in self._adj_disp_energy_members:
            values.append(member.disp_energy)
        return values

    @adj_disp_energy.setter
    def adj_disp_energy(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_disp_energy_members):
            for i, member in enumerate(self._adj_disp_energy_members):
                if isinstance(value[i], float):
                    member.disp_energy = value[i]
                else:
                    raise ValueError("adj_disp_energy values must be floats.")
        else:
            raise TypeError("adj_disp_energy must be a list or tuple with {} elements."
                            .format(len(self._adj_disp_energy_members)))

    @property
    def adj_temp_indep_coef_members(self):
        return self._adj_temp_indep_coef_members

    @property
    def adj_temp_indep_coef(self):
        values = []
        for member in self._adj_temp_indep_coef_members:
            values.append(member.temp_indep_coef)
        return values

    @adj_temp_indep_coef.setter
    def adj_temp_indep_coef(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_temp_indep_coef_members):
            for i, member in enumerate(self._adj_temp_indep_coef_members):
                if isinstance(value[i], float):
                    member.temp_indep_coef = value[i]
                else:
                    raise ValueError("adj_temp_indep_coef values must be floats.")
        else:
            raise TypeError("adj_temp_indep_coef must be a list or tuple with {} elements."
                            .format(len(self._adj_temp_indep_coef_members)))

    @property
    def adj_lin_temp_coef_members(self):
        return self._adj_lin_temp_coef_members

    @property
    def adj_lin_temp_coef(self):
        values = []
        for member in self._adj_lin_temp_coef_members:
            values.append(member.lin_temp_coef)
        return values

    @adj_lin_temp_coef.setter
    def adj_lin_temp_coef(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_lin_temp_coef_members):
            for i, member in enumerate(self._adj_lin_temp_coef_members):
                if isinstance(value[i], float):
                    member.lin_temp_coef = value[i]
                else:
                    raise ValueError("adj_lin_temp_coef values must be floats.")
        else:
            raise TypeError("adj_lin_temp_coef must be a list or tuple with {} elements."
                            .format(len(self._adj_lin_temp_coef_members)))

    @property
    def adj_inv_temp_coef_members(self):
        return self._adj_inv_temp_coef_members

    @property
    def adj_inv_temp_coef(self):
        values = []
        for member in self._adj_inv_temp_coef_members:
            values.append(member.inv_temp_coef)
        return values

    @adj_inv_temp_coef.setter
    def adj_inv_temp_coef(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_inv_temp_coef_members):
            for i, member in enumerate(self._adj_inv_temp_coef_members):
                if isinstance(value[i], float):
                    member.inv_temp_coef = value[i]
                else:
                    raise ValueError("adj_inv_temp_coef values must be floats.")
        else:
            raise TypeError("adj_inv_temp_coef must be a list or tuple with {} elements."
                            .format(len(self._adj_inv_temp_coef_members)))

    @property
    def adj_ln_temp_coef_members(self):
        return self._adj_ln_temp_coef_members

    @property
    def adj_ln_temp_coef(self):
        values = []
        for member in self._adj_ln_temp_coef_members:
            values.append(member.ln_temp_coef)
        return values

    @adj_ln_temp_coef.setter
    def adj_ln_temp_coef(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_ln_temp_coef_members):
            for i, member in enumerate(self._adj_ln_temp_coef_members):
                if isinstance(value[i], float):
                    member.ln_temp_coef = value[i]
                else:
                    raise ValueError("adj_ln_temp_coef values must be floats.")
        else:
            raise TypeError("adj_ln_temp_coef must be a list or tuple with {} elements."
                            .format(len(self._adj_ln_temp_coef_members)))

    @property
    # TODO: Stylistic point.  Make sure a _ prepends any hidden class attribute abbreviations.
    def seg_num(self):
        sn = []
        for _pcp in self._pure_comp_parms:
            sn.append(_pcp.seg_num)
        return np.array(sn)

    @property
    def seg_diam(self):
        sd = []
        for _pcp in self._pure_comp_parms:
            sd.append(_pcp.seg_diam)
        return np.array(sd)

    @property
    def disp_energy(self):
        de = []
        for _pcp in self._pure_comp_parms:
            de.append(_pcp.disp_energy)
        return np.array(de)

    def set_adj_pure_comp_parms(self, adj_pure_comp_spec):
        # Define which parameters are adjustable in each PCSAFTParms in pure_comp_parms. Reset all 'member' lists to
        # empty lists as a default every time this method is called.
        self._adj_seg_num_members = []
        self._adj_seg_diam_members = []
        self._adj_disp_energy_members = []
        if adj_pure_comp_spec is None or adj_pure_comp_spec == 'none':
            pass
        elif adj_pure_comp_spec == 'all':
            self._adj_seg_num_members = self._pure_comp_parms[:]
            self._adj_seg_diam_members = self._pure_comp_parms[:]
            self._adj_disp_energy_members = self._pure_comp_parms[:]
        elif adj_pure_comp_spec == 'seg_num':
            self._adj_seg_num_members = self._pure_comp_parms[:]
        elif adj_pure_comp_spec == 'seg_diam':
            self._adj_seg_diam_members = self._pure_comp_parms[:]
        elif adj_pure_comp_spec == 'disp_energy':
            self._adj_disp_energy_members = self._pure_comp_parms[:]
        else:
            raise ValueError("adj_pure_comp_spec not valid.")

    def set_adj_binary_parms(self, adj_binary_spec):
        # Create lists that define which parameters are adjustable in each BinaryInterParm (bip) in binary_parms. Reset
        # all 'member' lists to empty lists as a default every time this method is called.
        self._adj_temp_indep_coef_members = []
        self._adj_lin_temp_coef_members = []
        self._adj_inv_temp_coef_members = []
        self._adj_ln_temp_coef_members = []
        if adj_binary_spec is None or adj_binary_spec == 'none':
            pass
        elif adj_binary_spec == 'all':
            for bip in self._binary_parms:
                # If an interaction parameter has not been initialized, then assume it is a temperature independent
                # interaction parameter and initialize the temp_indep_coef to 0.0.
                if not bip.defined():
                    bip.temp_indep_coef = 0.0
                # Only allow parameters to be adjusted for mixed-component binaries.
                if bip.comp_a != bip.comp_b:
                    # Coefficients can be adjustable if they have been initialized (changed from the 'None' default).
                    if bip.temp_indep_coef is not None:
                        self._adj_temp_indep_coef_members.append(bip)
                    if bip.lin_temp_coef is not None:
                        self._adj_lin_temp_coef_members.append(bip)
                    if bip.inv_temp_coef is not None:
                        self._adj_inv_temp_coef_members.append(bip)
                    if bip.ln_temp_coef is not None:
                        self._adj_ln_temp_coef_members.append(bip)
        else:
            raise ValueError("adj_binary_spec not valid.")

    def _create_pure_comp_parms(self):
        result = []
        for comp in self._comps.comps:
            new = PCSAFTParms(comp=comp, pc_saft_spec=self._pc_saft_spec)
            if new not in result:
                result.append(new)
        return result

    def load_pure_comp_parms(self):
        # Load equation of state parameters for all Comp objects. The absence of a PCSAFTParms object in a Comp object
        # triggers the init_parms procedure in PCSAFTParms to generate a first-pass estimate as a basis for further
        # parameter optimization.
        for pcp in self._pure_comp_parms:
            if self._eos == 'sPC-SAFT':
                if isinstance(pcp.comp.spc_saft_phys, PCSAFTParms) and pcp.comp.spc_saft_phys.defined():
                    pcp.seg_num = pcp.comp.spc_saft_phys.seg_num
                    pcp.seg_diam = pcp.comp.spc_saft_phys.seg_diam
                    pcp.disp_energy = pcp.comp.spc_saft_phys.disp_energy
                elif pcp.comp.spc_saft_phys is None:
                    pcp.init_parms()
                else:
                    raise TypeError("spc_saft_phys attribute for Comp object must be a PCSAFTParms object.")
            elif self._eos == 'PC-SAFT':
                if isinstance(pcp.comp.pc_saft_phys, PCSAFTParms) and pcp.comp.pc_saft_phys.defined():
                    pcp.seg_num = pcp.comp.pc_saft_phys.seg_num
                    pcp.seg_diam = pcp.comp.pc_saft_phys.seg_diam
                    pcp.disp_energy = pcp.comp.pc_saft_phys.disp_energy
                elif pcp.comp.pc_saft_phys is None:
                    pcp.init_parms()
                else:
                    raise TypeError("pc_saft_phys attribute for Comp object must be a PCSAFTParms object.")
            else:
                raise ValueError("eos not valid.")

    def _create_binary_parms(self):
        # Create list of all possible component-to-component interactions as BinaryInterParm objects.
        # TODO: Rethink the definition of a bip or assoc term.  Is it associatied with a physical or assciation term.
        #  Or is it assocated with an equation of state which is a collection of terms?
        result = []
        for comp_i in self._comps.comps:
            for comp_j in self._comps.comps:
                if comp_i == comp_j:
                    # TODO: Add keywords for all function arguments for clarity.
                    new = BinaryInterParm(comp_i, comp_j,
                                          self._pc_saft_spec, self._eos,
                                          "pure component", temp_indep_coef=0.0)
                    if new not in result:
                        result.append(new)
                else:
                    new = BinaryInterParm(comp_i, comp_j,
                                          self._pc_saft_spec, self._eos)
                    if new not in result:
                        result.append(new)
        return result

    def load_binary_parms(self, input):
        # Load coefficients and source from BinaryInterParm (bip) objects.
        if isinstance(input, BinaryInterParm):
            for _bp in self._binary_parms:
                if input == _bp:
                    _bp.temp_indep_coef = input.temp_indep_coef
                    _bp.lin_temp_coef = input.lin_temp_coef
                    _bp.inv_temp_coef = input.inv_temp_coef
                    _bp.ln_temp_coef = input.ln_temp_coef
                    _bp.source = input.source
        elif isinstance(input, PCSAFTPhysInter):
            for bp in input.binary_parms:
                for _bp in self._binary_parms:
                    if bp == _bp:
                        _bp.temp_indep_coef = bp.temp_indep_coef
                        _bp.lin_temp_coef = bp.lin_temp_coef
                        _bp.inv_temp_coef = bp.inv_temp_coef
                        _bp.ln_temp_coef = bp.ln_temp_coef
                        _bp.source = bp.source
        elif isinstance(input, list):
            for item in input:
                if isinstance(item, BinaryInterParm):
                    for _bp in self._binary_parms:
                        if item == _bp:
                            _bp.temp_indep_coef = item.temp_indep_coef
                            _bp.lin_temp_coef = item.lin_temp_coef
                            _bp.inv_temp_coef = item.inv_temp_coef
                            _bp.ln_temp_coef = item.ln_temp_coef
                            _bp.source = item.source
                else:
                    raise TypeError("Must pass a list of BinaryInterParm objects.")
        else:
            raise TypeError("Input not BinaryInterParm object, PhysInter object, or list of BinaryInterParm objects.")

    def k_ij(self, t=None):
        # Build array of interaction parameters for all comp combinations.
        n = len(self._comps.comps)
        k_ij = np.zeros((n, n))
        if t is not None and isinstance(t, float):
            for bp in self._binary_parms:
                i = self._comps.comps.index(bp.comp_a)
                j = self._comps.comps.index(bp.comp_b)
                k_ij[i, j] = bp.k_ij(t=t)
                k_ij[j, i] = bp.k_ij(t=t)
        else:
            raise ValueError("temperature must be a float.")
        return k_ij

    def __eq__(self, other):
        if isinstance(other, PCSAFTPhysInter):
            comps_eq = self.comps == other.comps
            spec_eq = self.pc_saft_spec == other.pc_saft_spec
            seg_num_eq = np.array_equal(self.seg_num, other.seg_num)
            seg_diam_eq = np.array_equal(self.seg_diam, other.seg_diam)
            disp_energy_eq = np.array_equal(self.disp_energy, other.disp_energy)
            kij_a_eq = np.array_equal(self.k_ij(298.15), other.k_ij(298.15))
            kij_b_eq = np.array_equal(self.k_ij(398.15), other.k_ij(398.15))
            return comps_eq and spec_eq and seg_num_eq and seg_diam_eq and disp_energy_eq and kij_a_eq and kij_b_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.comps, self.pc_saft_spec, self.seg_num, self.seg_diam, self.disp_energy, self.k_ij(298.15)))


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

    def _f_ti(self, t, v, ni, xai):
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
            result[:, i] = (f1 - f2) / (2.0 * di)
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


class CPA(EOS):
    """Generalized implementation of the Cubic Plus Association equation of state.

    Attributes:
        r: Ideal gas constant
        a: Energy parameter
        b: Co-volume parameter
        v: Molar volume, 1/rho
        rho: Molar density, 1/v
        eta: Reduced fluid density, b/4*V

    Methods:
        p: Returns pressure when volume and temperature are specified
        phi: Returns fugacity coefficients
    """

    def __init__(self, cubic_spec=None):
        # Define a cubic equation of state.
        if isinstance(cubic_spec, CubicSpec):
            self.delta_1 = cubic_spec.delta_1
            self.delta_2
            self.alpha = self._soave
        else:
            raise ValueError("Must pass a CubicSpec instance to the constructor.")

        # Lists of component objects, ci, and their concentrations, ni.
        self.ci = []
        self.ni = []

    def p(self, t, v, a, b):
        return self.r * t / (v - b) - a / ((v + self.d1 * b) * (v + self.d2 * b))

    def z(self, t, v, a, b):
        return 1.0 / (1.0 - b / v) - (a / (self.r * t * b)) * (b / v) / (
                    (1.0 + self.d1 * b / v) * (1.0 + self.d2 * b / v))

    def F(self, n, t, V, B, D):
        return -n * g(V, B) - D(t) * f(V, B) / t

    def g(self, V, B):
        return np.log(V - B) - np.log(V)

    def f(self, V, B):
        return np.log((V + self.d1 * B) / (V + self.d2 * B)) / (self.r * B * (self.d1 - self.d2))


class sPCSAFT(EOS):
    """sPC-SAFT equation of state for a phase."""

    def __init__(self, phys_inter=None, assoc_inter=None):
        if phys_inter is None:
            raise ValueError("Must pass a PCSAFTPhysInter object to the constructor.")
        else:
            self.phys_inter = phys_inter
            self.assoc_inter = assoc_inter

    @property
    def phys_inter(self):
        return self._phys_inter

    @phys_inter.setter
    def phys_inter(self, value):
        try:
            self._phys_inter
        except AttributeError:
            if isinstance(value, PCSAFTPhysInter):
                self._phys_inter = value
            else:
                raise TypeError("Must pass a PCSAFTPhysInter object to the constructor.")

    @property
    def assoc_inter(self):
        return self._assoc_inter

    @assoc_inter.setter
    def assoc_inter(self, value):
        try:
            self._assoc_inter
        except AttributeError:
            if isinstance(value, AssocInter) or value is None:
                self._assoc_inter = value
            else:
                raise TypeError("Must pass a AssocInter object to the constructor.")

    def vol_solver(self, p=None, t=None, ni=None, xai=None, root=None, prior=None, eps=10 ** -2, max_iter=100):
        """Solve for volume when temperature, mole numbers, and pressure are specified."""
        # Check inputs for consistency.
        # TODO: Use type hints to check for type.  isinstance takes a bunch of time.  Still use > 0 checks, but do it
        # TODO: in one line.
        # TODO: Don't initialize as None.  Just type check for float.  Does the same thing.
        if not isinstance(p, float) and p > 0.0:
            raise TypeError("p must be a positive float.")
        elif not isinstance(t, float) and t > 0.0:
            raise TypeError("t must be a positive float.")
        elif not isinstance(ni, (list, tuple, np.ndarray)):
            raise TypeError("ni must be a list, tuple, or np.ndarray.")
        # TODO: type check this also .... typing.List[float].  The general concept here is generics.
        # List[float] equivalent to all(isinstance(i, (float, np.floating)) for i in ni)
        # T = TypeVar()
        # ArrayLike[T] = Union[List[T}, Tuple[T]......]
        # floaty = float | np.floating
        elif not all(isinstance(i, (float, np.floating)) for i in ni):
            raise TypeError("ni can only contain floats.")
        elif not all(i >= 0.0 for i in ni):
            raise ValueError("ni can only contain positive floats.")
        elif not isinstance(eps, float) and eps > 0.0:
            raise TypeError("eps must be a positive float.")
        elif not isinstance(max_iter, int) and max_iter > 0:
            raise TypeError("max_iter must be a positive integer.")

        # Newton's Method iterations solving for pressure by adjusting volume. Used Newton's method structure presented
        # in <<https://secure.math.ubc.ca/~pwalls/math-python/roots-optimization/newton/>>.  Convergence is safeguarded
        # by including modifications suggested in Michelsen, M. L. Robust and Efficient Solution Procedures for
        # Association Models. Ind. Eng. Chem. Res., 45, 2006.

        # Equations to solve.
        def func(t, v, ni, xai):
            return self.p(t, v, ni, xai) - p

        def func_v(t, v, ni, xai):
            # TODO: Consider re-defining with numeric drivative of P.
            # TODO: Need to define function call in terms of xai.
            return self.p_v(t, v, ni, xai) * self._v_from_eta_eta(t, self._eta(t, v, ni), ni)

        # Define reduced density limits based on physics of the problem.  Very close to Hexagonal Close Packed
        eta_min = 0.0
        eta_max = 0.74047
        # TODO: Consider capitalizing.  Shift F6 is a good renaming tool....use it.
        eta_abs_min = 0.0
        eta_abs_max = 0.74047

        # Initial reduced density guess based on user specification.
        if root == 'liquid':
            eta = 0.5
        elif root == 'vapor':
            eta = 0.01
        elif root == 'prior':
            if isinstance(prior, float) and prior > 0.0:
                eta = self._eta(t, prior, ni)
            else:
                raise ValueError("prior must be a positive float.")
        else:
            raise ValueError("root specification must be either liquid, vapor, or prior.")

        for n in range(0, max_iter):
            # Convert reduced density (iteration variable) into volume (practical variable for EOS evaluation).
            print("eta: {}".format(eta))
            v = self._v_from_eta(t, eta, ni)
            # Evaluate f and check for convergence.
            _func = func(t, v, ni, xai)
            if abs(_func) < eps:
                return v, self.gr(t, v, ni, xai)
            # Update iteration variable limits.
            if _func > 0.0:
                eta_max = eta
            else:
                eta_min = eta
            # Evaluate the derivative of f and check for zero denominator. If zero denominator encountered, then
            # estimate new reduced density using a limit bisection and proceed to the next iteration.
            _func_v = func_v(t, v, ni, xai)
            # TODO:  Don't compare to 0...compare to an epsilon tolerance that prevents numerical explosion.
            # TODO:  Reformulate this as a try/except.  Try this thing....except divide by zero error. Ask forgivenes...
            if _func_v == 0.0:
                eta = (eta_min + eta_max) / 2.0
                continue
            # Estimate new reduced density using Newtons Method.
            eta = eta - _func / _func_v
            # Replace reduced density with a limit bisection if Newton's correction fell outside the limits.
            if not eta_min < eta < eta_max:
                eta = (eta_min + eta_max) / 2.0
            # Raise error if reduced density exceeds eta_abs_max.
            # TODO: Make  explicit. Why is density check outside of the max limit special. Why check. Tell the user.
            if eta <= eta_abs_min or eta >= eta_abs_max:
                raise RuntimeError("Exceeded density limits.")
        raise RuntimeError("Exceeded maximum iterations. No solution found.")

    def _v_from_eta(self, t, eta, ni):
        # Dimension is m**3
        #
        # Note that eta cannot exceed ~0.74048 (closest packing of segments).  The argument 'v' has units of [m3] which
        # is converted to units of [Å**3] by multiplying by  10.0 ** 30.0.
        return PI * NA * self._d3(t, ni) * np.sum(ni * self.phys_inter.seg_num) / (6.0 * eta * 10.0 ** 30.0)

    def _v_from_eta_eta(self, t, eta, ni):
        # Dimension is m**3
        #
        # Note that eta cannot exceed ~0.74048 (closest packing of segments).  The argument 'v' has units of [m3] which
        # is converted to units of [Å**3] by multiplying by  10.0 ** 30.0.
        return - PI * NA * self._d3(t, ni) * np.sum(ni * self.phys_inter.seg_num) / (6.0 * (eta ** 2.0) * 10.0 ** 30.0)

    # This implementation of sPC-SAFT follows Appendix F in de Villers, A. J. Evaluation and improvement of the sPC-SAFT
    # equation of state for complex mixtures. PhD Thesis. Stellenbosch University. 2011.

    def _m(self, ni):
        # Dimensionless.
        # de Villers (2011), Equation F-2
        return np.sum(ni * self.phys_inter.seg_num) / self._n(ni)

    def _m_i(self, ni):
        # de Villers (2011), Equation F-8
        return self.phys_inter.seg_num / self._n(ni) - np.sum(ni * self.phys_inter.seg_num) / (self._n(ni) ** 2.0)

    def _di(self, t):
        # Dimension is Å.
        # de Villers (2011), Equation F-6
        # TODO: Make the 0.12 an adjustable parameter, maybe ci. Must be complete before implementing temp derivatives.
        # TODO: Integrate ck_const into PCSAFTPHYSINTER class.
        return self.phys_inter.seg_diam * (1.0 - 0.12 * np.exp(-3.0 * self.phys_inter.disp_energy / t))

    def _d(self, t, ni):
        # Dimension is Å
        # de Villers (2011), Equation F-5
        return self._d3(t, ni) ** (1.0 / 3.0)

    def _d3(self, t, ni):
        # Dimension is Å**3
        # de Villers (2011), Equation F-5
        return np.sum(ni * self.phys_inter.seg_num * self._di(t) ** 3.0) / np.sum(ni * self.phys_inter.seg_num)

    def _d_i(self, t, ni):
        # de Villers (2011), Equation F-10
        return (self.phys_inter.seg_num * self._di(t) ** 3.0 / np.sum(ni * self.phys_inter.seg_num) -
                self.phys_inter.seg_num * np.sum(ni * self.phys_inter.seg_num * self._di(t) ** 3.0) /
                np.sum(ni * self.phys_inter.seg_num) ** 2.0) / (3.0 * self._d(t, ni) ** 2.0)

    def _rho(self, v, ni):
        # Dimension is 1/Å**3.
        # de Villers (2011), Equation F-57
        #
        # Number density of molecules (and not molar density).  Number density is formally defined as the number of
        # molecules per unit volume.  This means that the units for ρ = N / V are really [count / volume].  The "count"
        # is formally treated as dimensionless.  This can be understood by comparing alternative formulations of the
        # ideal gas law and evaluating units.  This representation of rho in "statistical mechanics" units can be tricky
        # for chemists and engineers.
        #
        #    PV = NkT -> P = kT(N/V) -> P = kTρ -> [Pa] = [J/K][K][?] -> [kg/m.s2] = [kg.m2/s2.K][K][?] -> [] = [m3][?]
        return NA * self._n(ni) / self._v_a3(v)

    def _v_a3(self, v):
        # Dimension is Å**3.
        # Function not a part of original de Villers (2011) implementation and was added for convenience.
        #
        # The argument 'v' has units of [m**3] which is converted to units of [Å**3] by multiplying by  10.0 ** 30.0.
        return v * (10.0 ** 30.0)

    def _eta(self, t, v, ni):
        # Dimensionless.
        # de Villers (2011), Equation F-4
        return PI * NA * self._d3(t, ni) * np.sum(ni * self.phys_inter.seg_num) / (6.0 * self._v_a3(v))

    def _eta_v(self, t, v, ni):
        # de Villers (2011), Equation F-16
        #
        # Note that the conversion factor from [m3] to [Å**3] exists here but not in de Villers. This is due to the
        # conversion in this implementation being separated into _v_a3 and application of the chain rule to _v_a3.
        return -(10.0 ** 30.0) * self._eta(t, v, ni) / self._v_a3(v)

    def _eta_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._eta, t, v, ni, None, self._step_size['f_i'])

    def _eta_i(self, t, v, ni):
        # de Villers (2011), Equation F-9
        return (NA * PI / (6.0 * self._v_a3(v))) * (self._d3(t, ni) * self.phys_inter.seg_num +
                                                    3.0 * (self._d(t, ni) ** 2.0) * self._d_i(t, ni) *
                                                    np.sum(ni * self.phys_inter.seg_num))

    def _fhs(self, t, v, ni):
        # Dimensionless.
        # de Villers (2011), Equation F-3
        return self._m(ni) * self._n(ni) * ((4.0 * self._eta(t, v, ni) - 3.0 * self._eta(t, v, ni) ** 2.0) /
                                            ((1.0 - self._eta(t, v, ni)) ** 2.0))

    def _fhs_v(self, t, v, ni):
        # Dimensionless.
        # de Villers (2011), Equation F-15
        return 2.0 * self._m(ni) * self._n(ni) * self._eta_v(t, v, ni) * ((4.0 * self._eta(t, v, ni) -
                                                                           3.0 * self._eta(t, v, ni) ** 2.0) /
                                                                          ((1.0 - self._eta(t, v, ni)) ** 3.0)) + \
               self._m(ni) * self._n(ni) * ((4.0 * self._eta_v(t, v, ni) -
                                             6.0 * self._eta(t, v, ni) * self._eta_v(t, v, ni)) /
                                            ((1.0 - self._eta(t, v, ni)) ** 2.0))

    def _fhs_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._fhs, t, v, ni, None, self._step_size['f_v'])

    def _fhs_i(self, t, v, ni):
        # de Villers (2011), Equation F-7
        return self._m_i(ni) * self._n(ni) * ((4.0 * self._eta(t, v, ni) - 3.0 * self._eta(t, v, ni) ** 2.0) /
                                              ((1.0 - self._eta(t, v, ni)) ** 2.0)) + \
               self._m(ni) * ((4.0 * self._eta(t, v, ni) - 3.0 * self._eta(t, v, ni) ** 2.0) /
                              ((1.0 - self._eta(t, v, ni)) ** 2.0)) + \
               2.0 * self._m(ni) * self._n(ni) * self._eta_i(t, v, ni) * ((4.0 * self._eta(t, v, ni) -
                                                                           3.0 * self._eta(t, v, ni) ** 2.0) /
                                                                          ((1.0 - self._eta(t, v, ni)) ** 3.0)) + \
               self._m(ni) * self._n(ni) * ((4.0 * self._eta_i(t, v, ni) -
                                             6.0 * self._eta(t, v, ni) * self._eta_i(t, v, ni)) /
                                            ((1.0 - self._eta(t, v, ni)) ** 2.0))

    def _ghs(self, t, v, ni):
        # Dimensionless.
        # de Villers (2011), Equation F-37
        return (2.0 - self._eta(t, v, ni)) / (2.0 * (1.0 - self._eta(t, v, ni)) ** 3.0)

    def _ghs_v(self, t, v, ni):
        # de Villers (2011), Equation F-43
        return self._eta_v(t, v, ni) * (5.0 - 2.0 * self._eta(t, v, ni)) / (2.0 * (1.0 - self._eta(t, v, ni)) ** 4.0)

    def _ghs_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._ghs, t, v, ni, None, self._step_size['f_v'])

    def _ghs_i(self, t, v, ni):
        # de Villers (2011), Equation F-39
        return self._eta_i(t, v, ni) * (-1.0 / (2.0 * ((1.0 - self._eta(t, v, ni)) ** 3.0)) +
                                        3.0 * (2.0 - self._eta(t, v, ni)) /
                                        (2.0 * ((1.0 - self._eta(t, v, ni)) ** 4.0)))

    def _fhc(self, t, v, ni):
        # de Villers (2011), Equation F-36
        return np.log(self._ghs(t, v, ni)) * np.sum(ni * (1.0 - self.phys_inter.seg_num))

    def _fhc_v(self, t, v, ni):
        # de Villers (2011), Equation F-42
        return self._ghs_v(t, v, ni) * np.sum(ni * (1.0 - self.phys_inter.seg_num)) / self._ghs(t, v, ni)

    def _fhc_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._fhc, t, v, ni, None, self._step_size['f_v'])

    def _fhc_i(self, t, v, ni):
        # de Villers (2011), Equation F-38
        return (1.0 - self.phys_inter.seg_num) * np.log(self._ghs(t, v, ni)) + \
               np.sum(ni * (1.0 - self.phys_inter.seg_num)) * self._ghs_i(t, v, ni) / self._ghs(t, v, ni)

    def _ai(self, i, ni):
        # de Villers (2011), Equation F-69
        return self.phys_inter.pc_saft_spec.a[i, 0] + \
               self.phys_inter.pc_saft_spec.a[i, 1] * (self._m(ni) - 1.0) / self._m(ni) + \
               self.phys_inter.pc_saft_spec.a[i, 2] * (self._m(ni) - 1.0) * (self._m(ni) - 2.0) / (self._m(ni) ** 2.0)

    def _ai_i(self, i, ni):
        # de Villers (2011), F-76
        return self._m_i(ni) * (-4.0 * self.phys_inter.pc_saft_spec.a[i, 2] +
                                self._m(ni) * (self.phys_inter.pc_saft_spec.a[i, 1] +
                                               3.0 * self.phys_inter.pc_saft_spec.a[i, 2])) / self._m(ni) ** 3.0

    def _bi(self, i, ni):
        # de Villers (2011), Equation F-70
        return self.phys_inter.pc_saft_spec.b[i, 0] + \
               self.phys_inter.pc_saft_spec.b[i, 1] * (self._m(ni) - 1.0) / self._m(ni) + \
               self.phys_inter.pc_saft_spec.b[i, 2] * (self._m(ni) - 1.0) * (self._m(ni) - 2.0) / (self._m(ni) ** 2.0)

    def _bi_i(self, i, ni):
        # de Villers (2011), F-83
        return self._m_i(ni) * (-4.0 * self.phys_inter.pc_saft_spec.b[i, 2] +
                                self._m(ni) * (self.phys_inter.pc_saft_spec.b[i, 1] +
                                               3.0 * self.phys_inter.pc_saft_spec.b[i, 2])) / self._m(ni) ** 3.0

    def _i1(self, t, v, ni):
        # de Villers (2011), Equation F-64
        return np.sum(np.array([self._ai(i, ni) * self._eta(t, v, ni) ** i for i in range(7)]))

    def _i1_v(self, t, v, ni):
        # de Villers (2011), Equation F-96
        return self._eta_v(t, v, ni) * np.sum(np.array([i * self._ai(i, ni) * self._eta(t, v, ni) ** (i - 1.0)
                                                        for i in range(7)]))

    def _i1_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._i1, t, v, ni, None, self._step_size['f_v'])

    def _i1_i(self, t, v, ni):
        # de Villers (2011), F-75
        # TODO: Not fully tested yet.  Need to make sure axis=0 is really right.
        return np.sum(np.array([i * self._ai(i, ni) * self._eta_i(t, v, ni) * self._eta(t, v, ni) ** (i - 1) +
                                self._ai_i(i, ni) * self._eta(t, v, ni) ** i
                                for i in range(7)]), axis=0)

    def _i2(self, t, v, ni):
        # de Villers (2011), Equation F-65
        return np.sum(np.array([self._bi(i, ni) * self._eta(t, v, ni) ** i for i in range(7)]))

    def _i2_v(self, t, v, ni):
        # de Villers (2011), Equation F-101
        return self._eta_v(t, v, ni) * np.sum(np.array([i * self._bi(i, ni) * self._eta(t, v, ni) ** (i - 1.0)
                                                        for i in range(7)]))

    def _i2_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._i2, t, v, ni, None, self._step_size['f_v'])

    def _i2_i(self, t, v, ni):
        # de Villers (2011, F-87
        # TODO: Not fully tested yet.  Need to make sure axis=0 is really right.
        return np.sum(np.array([i * self._bi(i, ni) * self._eta_i(t, v, ni) * self._eta(t, v, ni) ** (i - 1) +
                                self._bi_i(i, ni) * self._eta(t, v, ni) ** i
                                for i in range(7)]), axis=0)

    def _epsij(self, t):
        # de Villers (2011), by definition
        return np.sqrt(np.outer(self.phys_inter.disp_energy, self.phys_inter.disp_energy)) * \
               (1.0 - self.phys_inter.k_ij(t))

    def _sigij(self):
        # de Villers (2011), by definition
        return 0.5 * np.add.outer(self.phys_inter.seg_diam, np.transpose(self.phys_inter.seg_diam))

    def _m2eps1sig3(self, t, ni):
        # de Villers (2011), Equation F-61
        return np.sum(np.outer(ni, ni) *
                      np.outer(self.phys_inter.seg_num, self.phys_inter.seg_num) *
                      np.power(self._epsij(t) / t, 1.0) *
                      np.power(self._sigij(), 3.0)) / (self._n(ni) ** 2.0)

    def _m2eps1sig3_i(self, t, ni):
        # de Villers (2011), Equation F-74
        # TODO: Not fully tested yet.  THis one is tricky.  Summation of array multiplication.  axis=1 makes this work.
        return 2.0 * self.phys_inter.seg_num * np.sum(ni *
                                                      self.phys_inter.seg_num *
                                                      np.power(self._epsij(t) / t, 1.0) *
                                                      np.power(self._sigij(), 3.0), axis=1) / (self._n(ni) ** 2.0) - \
               2.0 * np.sum(np.outer(ni, ni) *
                            np.outer(self.phys_inter.seg_num, self.phys_inter.seg_num) *
                            np.power(self._epsij(t) / t, 1.0) *
                            np.power(self._sigij(), 3.0)) / (self._n(ni) ** 3.0)

    def _m2eps2sig3(self, t, ni):
        # de Villers (2011), Equation F-63
        return np.sum(np.outer(ni, ni) *
                      np.outer(self.phys_inter.seg_num, self.phys_inter.seg_num) *
                      np.power(self._epsij(t) / t, 2.0) *
                      np.power(self._sigij(), 3.0)) / (self._n(ni) ** 2.0)

    def _m2eps2sig3_i(self, t, ni):
        # de Villers (2011), Equation F-79
        # TODO: Not fully tested yet.  THis one is tricky.  Summation of array multiplication.  axis=1 makes this work.
        return 2.0 * self.phys_inter.seg_num * np.sum(ni *
                                                      self.phys_inter.seg_num *
                                                      np.power(self._epsij(t) / t, 2.0) *
                                                      np.power(self._sigij(), 3.0), axis=1) / (self._n(ni) ** 2.0) - \
               2.0 * np.sum(np.outer(ni, ni) *
                            np.outer(self.phys_inter.seg_num, self.phys_inter.seg_num) *
                            np.power(self._epsij(t) / t, 2.0) *
                            np.power(self._sigij(), 3.0)) / (self._n(ni) ** 3.0)

    def _c0(self, t, v, ni):
        # de Villers (2011), Equation F-66
        return (1.0 + self._m(ni) * (8.0 * self._eta(t, v, ni) - 2.0 * self._eta(t, v, ni) ** 2.0) /
                ((1.0 - self._eta(t, v, ni)) ** 4.0) +
                (1.0 - self._m(ni)) * (20.0 * self._eta(t, v, ni) - 27.0 * self._eta(t, v, ni) ** 2.0 + 12.0 *
                                       self._eta(t, v, ni) ** 3.0 - 2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0)))

    def _c0_v(self, t, v, ni):
        # de Villers (2011), Equation F-99
        return (2.0 * (1.0 - self._m(ni)) * self._eta_v(t, v, ni) *
                (20.0 * self._eta(t, v, ni) - 27.0 * self._eta(t, v, ni) ** 2.0 + 12.0 * self._eta(t, v, ni) ** 3.0 -
                 2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 3.0))) + \
               (2.0 * (1.0 - self._m(ni)) * self._eta_v(t, v, ni) *
                (20.0 * self._eta(t, v, ni) - 27.0 * self._eta(t, v, ni) ** 2.0 + 12.0 * self._eta(t, v, ni) ** 3.0 -
                 2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 3.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0))) + \
               (4.0 * self._m(ni) * self._eta_v(t, v, ni) * (8.0 * self._eta(t, v, ni) -
                                                             2.0 * self._eta(t, v, ni) ** 2.0) /
                ((1.0 - self._eta(t, v, ni)) ** 5.0)) + \
               (self._m(ni) * self._eta_v(t, v, ni) * (8.0 - 4.0 * self._eta(t, v, ni)) /
                ((1.0 - self._eta(t, v, ni)) ** 4.0)) + \
               ((1.0 - self._m(ni)) * self._eta_v(t, v, ni) * (20.0 - 54.0 * self._eta(t, v, ni) +
                                                               36.0 * self._eta(t, v, ni) ** 2.0 -
                                                               8.0 * self._eta(t, v, ni) ** 3.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0)))

    def _c0_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._c0, t, v, ni, None, self._step_size['f_v'])

    def _c0_i(self, t, v, ni):
        # de Villers (2011), F-80
        return (self._m_i(ni) * (8.0 * self._eta(t, v, ni) - 2.0 * self._eta(t, v, ni) ** 2.0) /
                ((1.0 - self._eta(t, v, ni)) ** 4.0) -
                self._m_i(ni) * (20.0 * self._eta(t, v, ni) - 27.0 * self._eta(t, v, ni) ** 2.0 +
                                 12.0 * self._eta(t, v, ni) ** 3.0 - 2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0)) +
                2.0 * (1.0 - self._m(ni)) * self._eta_i(t, v, ni) * (20.0 * self._eta(t, v, ni) -
                                                                     27.0 * self._eta(t, v, ni) ** 2.0 +
                                                                     12.0 * self._eta(t, v, ni) ** 3.0 -
                                                                     2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 3.0)) +
                2.0 * (1.0 - self._m(ni)) * self._eta_i(t, v, ni) * (20.0 * self._eta(t, v, ni) -
                                                                     27.0 * self._eta(t, v, ni) ** 2.0 +
                                                                     12.0 * self._eta(t, v, ni) ** 3.0 -
                                                                     2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 3.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0)) +
                4.0 * self._m(ni) * self._eta_i(t, v, ni) * (8.0 * self._eta(t, v, ni) -
                                                             2.0 * self._eta(t, v, ni) ** 2.0) /
                ((1.0 - self._eta(t, v, ni)) ** 5.0) +
                self._m(ni) * self._eta_i(t, v, ni) * (8.0 - 4.0 * self._eta(t, v, ni)) /
                ((1.0 - self._eta(t, v, ni)) ** 4.0) +
                (1.0 - self._m(ni)) * self._eta_i(t, v, ni) * (20.0 - 54.0 * self._eta(t, v, ni) +
                                                               36.0 * self._eta(t, v, ni) ** 2.0 -
                                                               8.0 * self._eta(t, v, ni) ** 3.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0)))

    def _c1(self, t, v, ni):
        # de Villers (2011), Equation F-67
        return (1.0 + self._m(ni) * (8.0 * self._eta(t, v, ni) - 2.0 * self._eta(t, v, ni) ** 2.0) /
                ((1.0 - self._eta(t, v, ni)) ** 4.0) +
                (1.0 - self._m(ni)) * (20.0 * self._eta(t, v, ni) - 27.0 * self._eta(t, v, ni) ** 2.0 + 12.0 *
                                       self._eta(t, v, ni) ** 3.0 - 2.0 * self._eta(t, v, ni) ** 4.0) /
                (((1.0 - self._eta(t, v, ni)) ** 2.0) * ((2.0 - self._eta(t, v, ni)) ** 2.0))) ** -1.0

    def _c1_v(self, t, v, ni):
        # de Villers (2011), Equation F-100
        return -self._c0_v(t, v, ni) / (self._c0(t, v, ni) ** 2.0)

    def _c1_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._c1, t, v, ni, None, self._step_size['f_v'])

    def _c1_i(self, t, v, ni):
        # de Villers (2011), Equation F-81
        return -self._c0_i(t, v, ni) / self._c0(t, v, ni) ** 2.0

    def _fdisp(self, t, v, ni):
        # de Villers, Equation F-59
        return -(2.0 * PI * NA * (self._n(ni) ** 2.0) * self._i1(t, v, ni) * self._m2eps1sig3(t, ni) / self._v_a3(v) +
                 PI * NA * (self._n(ni) ** 2.0) * self._c1(t, v, ni) * self._m(ni) * self._i2(t, v, ni) *
                 self._m2eps2sig3(t, ni) / self._v_a3(v))

    def _fdisp1(self, t, v, ni):
        # Created for debugging purposes (allows us to test partial derivatives numerically).
        return -(2.0 * PI * NA * (self._n(ni) ** 2.0) * self._i1(t, v, ni) * self._m2eps1sig3(t, ni) / self._v_a3(v))

    def _fdisp1_v(self, t, v, ni):
        # de Villers (2011), Equation F-95
        #
        # Note that the conversion factor from [m3] to [Å**3] exists here but not in de Villers. This is due to the
        # conversion in this implementation being separated into _v_a3 and application of the chain rule to _v_a3.
        return ((10.0 ** 30.0) * 2.0 * PI * NA * (self._n(ni) ** 2.0) * self._i1(t, v, ni) * self._m2eps1sig3(t, ni) /
                (self._v_a3(v) ** 2.0)) - \
               (2.0 * PI * NA * (self._n(ni) ** 2.0) * self._m2eps1sig3(t, ni) * self._i1_v(t, v, ni) / self._v_a3(v))

    def _fdisp1_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._fdisp1, t, v, ni, None, self._step_size['f_v'])

    def _fdisp1_i(self, t, v, ni):
        # de Villers (2011), Equation F-73
        return -PI * NA * (4.0 * self._n(ni) * self._i1(t, v, ni) * self._m2eps1sig3(t, ni) +
                           2.0 * (self._n(ni) ** 2.0) * self._i1(t, v, ni) * self._m2eps1sig3_i(t, ni) +
                           2.0 * (self._n(ni) ** 2.0) * self._i1_i(t, v, ni) * self._m2eps1sig3(t, ni)) / \
               self._v_a3(v)

    def _fdisp2(self, t, v, ni):
        # Created for debugging purposes (allows us to test partial derivatives numerically).
        return -(PI * NA * (self._n(ni) ** 2.0) * self._c1(t, v, ni) * self._m(ni) * self._i2(t, v, ni) *
                 self._m2eps2sig3(t, ni) / self._v_a3(v))

    def _fdisp2_v(self, t, v, ni):
        # de Villers (2011), Equation F-98
        #
        # Note that the conversion factor from [m3] to [Å**3] exists here but not in de Villers. This is due to the
        # conversion in this implementation being separated into _v_a3 and application of the chain rule to _v_a3.
        return ((10.0 ** 30.0) * PI * NA * (self._n(ni) ** 2.0) * self._m(ni) * self._c1(t, v, ni) *
                self._i2(t, v, ni) * self._m2eps2sig3(t, ni) / (self._v_a3(v) ** 2.0)) - \
               (PI * NA * (self._n(ni) ** 2.0) * self._m(ni) * self._i2(t, v, ni) * self._m2eps2sig3(t, ni) *
                self._c1_v(t, v, ni) / self._v_a3(v)) - \
               (PI * NA * (self._n(ni) ** 2.0) * self._m(ni) * self._c1(t, v, ni) * self._m2eps2sig3(t, ni) *
                self._i2_v(t, v, ni) / self._v_a3(v))

    def _fdisp2_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._fdisp2, t, v, ni, None, self._step_size['f_v'])

    def _fdisp2_i(self, t, v, ni):
        # de Villers, Equation F-78
        return -PI * NA * ((self._n(ni) ** 2.0) * self._c1(t, v, ni) * self._i2(t, v, ni) * self._m2eps2sig3(t, ni) *
                           self._m_i(ni) +
                           2.0 * self._n(ni) * self._c1(t, v, ni) * self._i2(t, v, ni) * self._m2eps2sig3(t, ni) *
                           self._m(ni) +
                           (self._n(ni) ** 2.0) * self._c1(t, v, ni) * self._i2(t, v, ni) *
                           self._m2eps2sig3_i(t, ni) * self._m(ni) +
                           (self._n(ni) ** 2.0) * self._c1(t, v, ni) * self._i2_i(t, v, ni) *
                           self._m2eps2sig3(t, ni) * self._m(ni) +
                           (self._n(ni) ** 2.0) * self._c1_i(t, v, ni) * self._i2(t, v, ni) *
                           self._m2eps2sig3(t, ni) * self._m(ni)) / \
               self._v_a3(v)

    def _fdisp_v(self, t, v, ni):
        # de Villers (2011), Equation F-94
        return self._fdisp1_v(t, v, ni) + self._fdisp2_v(t, v, ni)

    def _fdisp_v_num(self, t, v, ni):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._fdisp, t, v, ni, None, self._step_size['f_v'])

    def _fdisp_i(self, t, v, ni):
        # de Villers (2011), Equation F-72
        return self._fdisp1_i(t, v, ni) + self._fdisp2_i(t, v, ni)

    def _f(self, t, v, ni, xai):
        # de Villers (2011), by definition
        return self._fhs(t, v, ni) + self._fhc(t, v, ni) + self._fdisp(t, v, ni)

    def _f_v(self, t, v, ni, xai):
        # de Villers (2011), by definition.
        return self._fhs_v(t, v, ni) + self._fhc_v(t, v, ni) + self._fdisp_v(t, v, ni)

    def _f_v_num(self, t, v, ni, xai=None):
        # Finite difference derivative implemented for testing.
        return self._func_v_num(self._f, t, v, ni, xai, self._step_size['f_v'])

    def _f_i(self, t, v, ni, xai):
        # de Villers (2011), by definition.
        return self._fhs_i(t, v, ni) + self._fhc_i(t, v, ni) + self._fdisp_i(t, v, ni)

    def _f_i_num_vect(self, t, v, ni, xai=None):
        # Finite difference derivative implemented for testing.
        return self._func_i_grad_num(self._f, t, v, ni, xai, self._step_size['f_i'])

    def __eq__(self, other):
        if isinstance(other, sPCSAFT):
            phys_inter_eq = self.phys_inter == other.phys_inter
            assoc_inter_eq = self.assoc_inter == other.assoc_inter
            return phys_inter_eq and assoc_inter_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.phys_inter, self.assoc_inter))


if __name__ == "__main__":
    # Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
    methane = Comp('Methane')
    methane.formula = "CH4"
    methane.family = "Alkanes"
    methane.cas_no = "74-82-8"
    methane.mw = 16.0425
    methane.tc = 190.564
    methane.pc = 4599000.0
    methane.vc = 0.0000986
    methane.acentric = 0.0115478

    methane.spc_saft_phys = PCSAFTParms(comp=methane,
                                        pc_saft_spec=GS,
                                        source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                        seg_num=1.0, seg_diam=3.7039, disp_energy=150.03)

    # Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
    ethane = Comp('Ethane')
    ethane.formula = "C2H6"
    ethane.family = "Alkanes"
    ethane.cas_no = "74-84-0"
    ethane.mw = 30.069
    ethane.tc = 305.32
    ethane.pc = 4872000.0
    ethane.vc = 0.0001455
    ethane.acentric = 0.099493

    # Parameters from Ind. Eng. Chem. Res. 2001, 40, 1244-1260.
    ethane.spc_saft_phys = PCSAFTParms(comp=ethane,
                                       pc_saft_spec=GS,
                                       source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                       seg_num=1.6069, seg_diam=3.5206, disp_energy=191.42)

    # Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
    propane = Comp('Propane')
    propane.formula = "C3H8"
    propane.family = "Alkanes"
    propane.cas_no = "74-98-6"
    propane.mw = 44.09562
    propane.tc = 369.83
    propane.pc = 4248000.0
    propane.vc = 0.0002
    propane.acentric = 0.152291

    propane.spc_saft_phys = PCSAFTParms(comp=propane,
                                        pc_saft_spec=GS,
                                        source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                        seg_num=2.002, seg_diam=3.6184, disp_energy=208.11)

    # Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
    nitrogen = Comp('Nitrogen')
    nitrogen.formula = "N2"
    nitrogen.family = "Inorganics"
    nitrogen.cas_no = "7727-37-9"
    nitrogen.mw = 28.0134
    nitrogen.tc = 126.2
    nitrogen.pc = 3400000.0
    nitrogen.vc = 0.00008921
    nitrogen.acentric = 0.0377215

    nitrogen.spc_saft_phys = PCSAFTParms(comp=nitrogen,
                                         pc_saft_spec=GS,
                                         source="Ind. Eng. Chem. Res. 2001, 40, 1244-1260.",
                                         seg_num=1.2053, seg_diam=3.3130, disp_energy=90.96)

    # TODO: Break out constants for compounds into their own module.  Put all components into one Comps module.
    # TODO: Break out binary interactions into one big module for all binary interactions.
    # Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
    water = Comp('Water')
    water.formula = "H2O"
    water.family = "Inorganics"
    water.cas_no = "7732-18-5"
    # TODO: mw: typing.Optional[float]
    # TODO: mw: Optional[float] = None
    water.mw = 18.01528
    water.tc = 647.096
    water.pc = 22064000.0
    water.vc = 0.0000559472
    water.acentric = 0.344861

    water.pvap = Corel(a=73.649,
                       b=-7258.2,
                       c=-7.3037,
                       d=4.1653 * 10 ** -6,
                       e=2.0,
                       eq_id=1,
                       source_t_min=273.16,
                       source_t_max=647.1,
                       source_t_unit='K',
                       source_unit='Pa',
                       source=REFERENCES['DIPPR'],
                       notes="DIPPR correlation parameters taken from Perry's Chemical Engineers' Handbook, 9th")

    water.hvap = Corel(a=5.66 * 10 ** 7,
                       b=0.612041,
                       c=-0.625697,
                       d=0.398804,
                       e=0.0,
                       f=647.096,
                       eq_id=4,
                       source_t_min=273.16,
                       source_t_max=647.096,
                       source_t_unit='K',
                       source_unit='J/kmol',
                       source=REFERENCES['DIPPR'],
                       notes="DIPPR correlation parameters taken from Perry's Chemical Engineers' Handbook, 9th")

    water.cp_ig = Corel(a=0.33363 * 10 ** 5.0,
                        b=0.26790 * 10 ** 5.0,
                        c=2.61050 * 10 ** 3.0,
                        d=0.08896 * 10 ** 5.0,
                        e=1169.0,
                        eq_id=7,
                        source_t_min=100.0,
                        source_t_max=2273.15,
                        source_t_unit='K',
                        source_unit="J/kmol.K",
                        source=REFERENCES['DIPPR'],
                        notes="DIPPR correlation parameters taken from Perry's Chemical Engineers' Handbook, 9th")

    water.spc_saft_phys = PCSAFTParms(comp=water,
                                      pc_saft_spec=GS,
                                      source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                                      seg_num=2.0, seg_diam=2.3449, disp_energy=171.67)

    # Define association sites corresponding to water as a symmetrically associating 4C molecule.
    ea1 = AssocSite(comp=water, site='H1', type='ea')
    ea2 = AssocSite(comp=water, site='H2', type='ea')
    ed1 = AssocSite(comp=water, site='O1', type='ed')
    ed2 = AssocSite(comp=water, site='O2', type='ed')

    water.assoc_sites = [ea1, ea2, ed1, ed2]

    ea1_ed1 = AssocSiteInter(site_a=ea1, site_b=ed1,
                             eos='sPC-SAFT',
                             source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                             assoc_energy=1704.06, assoc_vol=0.3048)
    ea1_ed2 = AssocSiteInter(site_a=ea1, site_b=ed2,
                             eos='sPC-SAFT',
                             source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                             assoc_energy=1704.06, assoc_vol=0.3048)
    ea2_ed1 = AssocSiteInter(site_a=ea2, site_b=ed1,
                             eos='sPC-SAFT',
                             source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                             assoc_energy=1704.06, assoc_vol=0.3048)
    ea2_ed2 = AssocSiteInter(site_a=ea2, site_b=ed2,
                             eos='sPC-SAFT',
                             source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
                             assoc_energy=1704.06, assoc_vol=0.3048)

    water.spc_saft_assoc = [ea1_ed1, ea1_ed2, ea2_ed1, ea2_ed2]

    # Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
    methanol = Comp('Methanol')
    methanol.formula = "CH3OH"
    methanol.family = "Alcohols"
    methanol.cas_no = "67-56-1"
    methanol.mw = 32.04186
    methanol.tc = 512.5
    methanol.pc = 8084000.0
    methanol.vc = 0.000117
    methanol.acentric = 0.565831

    methanol.spc_saft_phys = PCSAFTParms(comp=methanol,
                                         pc_saft_spec=GS,
                                         source="Ind. Eng. Chem. Res. 2012, 45, 14903–14914.",
                                         seg_num=1.88238, seg_diam=3.0023, disp_energy=181.77)

    # Define association sites corresponding to methanol as a 2B molecule.
    ea1 = AssocSite(comp=methanol, site='H1', type='ea')
    ed1 = AssocSite(comp=methanol, site='O1', type='ed')

    methanol.assoc_sites = [ea1, ed1]

    ea1_ed1 = AssocSiteInter(site_a=ea1, site_b=ed1,
                             eos='sPC-SAFT',
                             source="Ind. Eng. Chem. Res. 2012, 45, 14903–14914.",
                             assoc_energy=2738.03, assoc_vol=0.054664)

    mm = BinaryInterParm(comp_a=methane, comp_b=methane,
                         phys_eos_spec=GS, eos='sPC-SAFT',
                         source='pure component',
                         temp_indep_coef=0.0)

    me = BinaryInterParm(comp_a=methane, comp_b=ethane,
                         phys_eos_spec=GS, eos='sPC-SAFT',
                         source='initial guess',
                         temp_indep_coef=0.03)

    mp = BinaryInterParm(comp_a=methane, comp_b=propane,
                         phys_eos_spec=GS, eos='sPC-SAFT',
                         source='initial guess',
                         temp_indep_coef=0.03)

    ee = BinaryInterParm(comp_a=ethane, comp_b=ethane,
                         phys_eos_spec=GS, eos='sPC-SAFT',
                         source='pure component',
                         temp_indep_coef=0.0)

    ep = BinaryInterParm(comp_a=ethane, comp_b=propane,
                         phys_eos_spec=GS, eos='sPC-SAFT',
                         source='initial guess',
                         temp_indep_coef=0.01)

    pp = BinaryInterParm(comp_a=propane, comp_b=propane,
                         phys_eos_spec=GS, eos='sPC-SAFT',
                         source='pure component',
                         temp_indep_coef=0.0)

    wm = BinaryInterParm(comp_a=water, comp_b=methane,
                         phys_eos_spec=GS, eos='sPC-SAFT',
                         source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                         temp_indep_coef=0.2306,
                         inv_temp_coef=-92.62)

    we = BinaryInterParm(comp_a=water, comp_b=ethane,
                         phys_eos_spec=GS, eos='sPC-SAFT',
                         source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                         temp_indep_coef=0.1773,
                         inv_temp_coef=-53.97)

    wp = BinaryInterParm(comp_a=water, comp_b=propane,
                         phys_eos_spec=GS, eos='sPC-SAFT',
                         source='initial guess',
                         temp_indep_coef=0.05)

    mem = BinaryInterParm(comp_a=methanol, comp_b=methane,
                          phys_eos_spec=GS, eos='sPC-SAFT',
                          source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                          temp_indep_coef=0.01)

    mee = BinaryInterParm(comp_a=methanol, comp_b=ethane,
                          phys_eos_spec=GS, eos='sPC-SAFT',
                          source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                          temp_indep_coef=0.02)

    mep = BinaryInterParm(comp_a=methanol, comp_b=propane,
                          phys_eos_spec=GS, eos='sPC-SAFT',
                          source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                          temp_indep_coef=0.02)

    wme = BinaryInterParm(comp_a=water, comp_b=methanol,
                          phys_eos_spec=GS, eos='sPC-SAFT',
                          source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
                          temp_indep_coef=-0.066)

    cs = CompSet(comps=[ethane, propane])

    spc_saft_phys = PCSAFTPhysInter(comps=cs, eos='sPC-SAFT', pc_saft_spec=GS)
    spc_saft_phys.load_pure_comp_parms()
    spc_saft_phys.load_binary_parms([ee, ep, pp])

    print("---------------")
    print("seg_num: {}".format(spc_saft_phys.seg_num))
    print("seg_diam: {}".format(spc_saft_phys.seg_diam))
    print("disp_energy: {}".format(spc_saft_phys.disp_energy))
    print("k_ij: {}".format(spc_saft_phys.k_ij(298.15)))

    spc_saft = sPCSAFT(phys_inter=spc_saft_phys, assoc_inter=None)

    phase = Phase(comps=cs, eos=spc_saft)
    phase.compos.set(xi=[0.2, 0.8])
    phase.state.spec = 'PT'

    phase.state.set(p=50000000.0, t=400.0)
    p = phase.state.p
    vm = phase.state.vm
    t = phase.state.t
    ni = phase.compos.xi
    mw = phase.props.mw
    i = 0
    xai = None

    # Checking the values for all functions that make up sPC-SAFT.  The most important check is that the resulting f, f_v,
    # and f_i values equal their numeric derivatives. The
    print("---------------")
    print("p = {}".format(p))
    print("t = {}".format(t))
    print("vm = {}".format(vm))
    print("mw = {}".format(phase.props.mw))
    print("rho_m = {}".format(phase.props.rho_m))
    print("ni = {}".format(ni))
    print("rho = {}".format(phase.eos._rho(vm, ni)))
    print("v_a3 = {}".format(phase.eos._v_a3(vm)))
    print("m = {}".format(phase.eos._m(ni)))
    print("m_i = {}".format(phase.eos._m_i(ni)))
    print("di = {}".format(phase.eos._di(t)))
    print("d3 = {}".format(phase.eos._d3(t, ni)))
    print("d_i = {}".format(phase.eos._d_i(t, ni)))
    print("eta = {}".format(phase.eos._eta(t, vm, ni)))
    print("eta_v = {}".format(phase.eos._eta_v(t, vm, ni)))
    print("eta_i = {}".format(phase.eos._eta_i(t, vm, ni)))
    print("ghs = {}".format(phase.eos._ghs(t, vm, ni)))
    print("ghs_v = {}".format(phase.eos._ghs_v(t, vm, ni)))
    print("ghs_i = {}".format(phase.eos._ghs_i(t, vm, ni)))
    print("i1 = {}".format(phase.eos._i1(t, vm, ni)))
    print("i1_v = {}".format(phase.eos._i1_v(t, vm, ni)))
    print("i1_i = {}".format(phase.eos._i1_i(t, vm, ni)))
    print("i2 = {}".format(phase.eos._i2(t, vm, ni)))
    print("i2_v = {}".format(phase.eos._i2_v(t, vm, ni)))
    print("i2_i = {}".format(phase.eos._i2_i(t, vm, ni)))
    print("eps_ij = {}".format(phase.eos._epsij(t)))
    print("sig_ij = {}".format(phase.eos._sigij()))
    print("ai_i = {}".format(phase.eos._ai_i(2, ni)))
    print("bi_i = {}".format(phase.eos._bi_i(2, ni)))
    print("m2eps1sig3 = {}".format(phase.eos._m2eps1sig3(t, ni)))
    print("m2eps1sig3_i = {}".format(phase.eos._m2eps1sig3_i(t, ni)))
    print("m2eps2sig3 = {}".format(phase.eos._m2eps2sig3(t, ni)))
    print("m2eps2sig3_i = {}".format(phase.eos._m2eps2sig3_i(t, ni)))
    print("c0 = {}".format(phase.eos._c0(t, vm, ni)))
    print("c0_v = {}".format(phase.eos._c0_v(t, vm, ni)))
    print("c0_i = {}".format(phase.eos._c0_i(t, vm, ni)))
    print("c1 (as inverse of c0) = {}".format(1.0 / phase.eos._c0(t, vm, ni)))
    print("c1 = {}".format(phase.eos._c1(t, vm, ni)))
    print("c1_v = {}".format(phase.eos._c1_v(t, vm, ni)))
    print("c1_i = {}".format(phase.eos._c1_i(t, vm, ni)))
    print("fhs = {}".format(phase.eos._fhs(t, vm, ni)))
    print("fhs_v = {}".format(phase.eos._fhs_v(t, vm, ni)))
    print("fhs_i = {}".format(phase.eos._fhs_i(t, vm, ni)))
    print("fhc = {}".format(phase.eos._fhc(t, vm, ni)))
    print("fhc_v = {}".format(phase.eos._fhc_v(t, vm, ni)))
    print("fhc_i = {}".format(phase.eos._fhc_i(t, vm, ni)))
    print("fdisp = {}".format(phase.eos._fdisp(t, vm, ni)))
    print("fdisp1 = {}".format(phase.eos._fdisp1(t, vm, ni)))
    print("fdisp1_v = {}".format(phase.eos._fdisp1_v(t, vm, ni)))
    print("fdisp1_i = {}".format(phase.eos._fdisp1_i(t, vm, ni)))
    print("fdisp2 = {}".format(phase.eos._fdisp2(t, vm, ni)))
    print("fdisp2_v = {}".format(phase.eos._fdisp2_v(t, vm, ni)))
    print("fdisp2_i = {}".format(phase.eos._fdisp2_i(t, vm, ni)))
    print("fdisp_v = {}".format(phase.eos._fdisp_v(t, vm, ni)))
    print("fdisp_i = {}".format(phase.eos._fdisp_i(t, vm, ni)))
    print("f = {}".format(phase.eos._f(t, vm, ni, xai)))
    print("f_v = {}".format(phase.eos._f_v(t, vm, ni, xai)))
    print("f_v_num = {}".format(phase.eos._f_v_num(t, vm, ni, xai)))
    print("f_i = {}".format(phase.eos._f_i(t, vm, ni, xai)))
    print("f_i_num = {}".format(phase.eos._f_i_num_vect(t, vm, ni, xai)))
    print("ln_phi_j = {}".format(phase.eos.ln_phi_j(t, vm, ni, xai)))

    # Examples illustrating how the Corel class functions.
    print("---------------")
    print("Water vapor pressure correlation defined: {}".format(water.pvap.defined()))
    print("Water vapor pressure at triple point: {}".format(water.pvap(water.pvap.t_min)))
    print("Water vapor pressure at critical point: {}".format(water.pvap(water.pvap.t_max)))

    print("Water ideal gas heat capacity correlation defined: {}".format(water.cp_ig.defined()))
    print("Water ideal gas heat capacity at triple point: {}".format(water.cp_ig(water.cp_ig.t_min)))
    print("Water ideal gas heat capacity at critical point: {}".format(water.cp_ig(water.cp_ig.t_max)))

    print("Water heat of vaporization correlation defined: {}".format(water.hvap.defined()))
    print("Water heat of vaporization at triple point: {}".format(water.hvap(water.hvap.t_min)))
    print("Water heat of vaporization at critical point: {}".format(water.hvap(water.hvap.t_max)))

    phase1 = copy.deepcopy(phase)