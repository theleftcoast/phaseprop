"""Simplified PC-SAFT equation of state.

Attributes
----------
GS : instance of PCSAFTSpec
    PC-SAFT universal constants (Gross and Sadowski, [1, 2]_).

References
----------
[1] Gross, J; Sadowski, G. Perturbed-Chain SAFT:  An Equation of State Based on a Perturbation Theory for Chain
Molecules. Ind. Eng. Chem. Res. 2001, 40, 1244–1260.
[2] Gross, J.; Sadowski, G. Application of the Perturbed-Chain SAFT Equation of State to Associating Systems. Ind. Eng.
Chem. Res. 2002, 41, 5510-5515.
"""
import numpy as np
from const import PI, NA
from comps import Comp, PseudoComp, CompSet
from eos import BinaryInterParm, EOS
from assoc import AssocInter

class sPCSAFTSpec(object):
    """Constants that define a specific version of the sPC-SAFT equation of state."""

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
        if isinstance(other, sPCSAFTSpec):
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

GS = sPCSAFTSpec(a=gs_a, b=gs_b)


class sPCSAFTParms(object):
    def __init__(self, comp=None, spc_saft_spec=None, source=None,
                 seg_num=None, seg_diam=None, disp_energy=None, ck_const=0.12):
        if comp is None or spc_saft_spec is None:
            raise ValueError("comp and pc_saft_spec must be provided to create an instance of PCSAFTParms.")
        else:
            self.comp = comp
            self.spc_saft_spec = spc_saft_spec
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
    def spc_saft_spec(self):
        return self._spc_saft_spec

    @spc_saft_spec.setter
    def spc_saft_spec(self, value):
        try:
            self._spc_saft_spec
        except AttributeError:
            if isinstance(value, sPCSAFTSpec):
                self._spc_saft_spec = value
            else:
                raise TypeError("spc_saft_spec must be an instance of sPCSAFTSpec.")

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
        if isinstance(other, sPCSAFTParms):
            comp_eq = self.comp == other.comp
            spec_eq = self.spc_saft_spec == other.spc_saft_spec
            return comp_eq and spec_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.comp), hash(self.spc_saft_spec)))

    def defined(self):
        sn = self.seg_num is not None
        sd = self.seg_diam is not None
        de = self.disp_energy is not None
        ck = self.ck_const is not None
        return sn and sd and de and ck

    def init_parms(self):
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


class sPCSAFTPhysInter(object):
    """Physical interactions between components."""
    def __init__(self, comps=None, spc_saft_spec=None, adj_pure_comp_spec=None, adj_binary_spec=None):
        if comps is None or spc_saft_spec is None:
            raise ValueError("comps and spc_saft_spec must be provided to create an instance of PCSAFTPhysInter.")
        self.comps = comps
        self.spc_saft_spec = spc_saft_spec
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
    def spc_saft_spec(self):
        return self._spc_saft_spec

    @spc_saft_spec.setter
    def spc_saft_spec(self, value):
        try:
            self._spc_saft_spec
        except AttributeError:
            if isinstance(value, sPCSAFTSpec):
                self._spc_saft_spec = value
            else:
                raise TypeError("spc_saft_spec not an instance of sPCSAFTSpec.")

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
            new = sPCSAFTParms(comp=comp, spc_saft_spec=self._spc_saft_spec)
            if new not in result:
                result.append(new)
        return result

    def load_pure_comp_parms(self):
        # Load equation of state parameters for all Comp objects. The absence of a sPCSAFTParms object in a Comp object
        # triggers the init_parms procedure in sPCSAFTParms to generate a first-pass estimate as a basis for further
        # parameter optimization.
        for pcp in self._pure_comp_parms:
            if isinstance(pcp.comp.spc_saft_phys, sPCSAFTParms) and pcp.comp.spc_saft_phys.defined():
                pcp.seg_num = pcp.comp.spc_saft_phys.seg_num
                pcp.seg_diam = pcp.comp.spc_saft_phys.seg_diam
                pcp.disp_energy = pcp.comp.spc_saft_phys.disp_energy
            elif pcp.comp.spc_saft_phys is None:
                pcp.init_parms()
            else:
                raise TypeError("spc_saft_phys attribute for Comp object must be a sPCSAFTParms object.")

    def _create_binary_parms(self):
        # Create list of all possible component-to-component interactions as BinaryInterParm objects.
        result = []
        for comp_i in self._comps.comps:
            for comp_j in self._comps.comps:
                new = BinaryInterParm(comp_i, comp_j, temp_indep_coef=0.0)
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
        elif isinstance(input, sPCSAFTPhysInter):
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
        if isinstance(other, sPCSAFTPhysInter):
            comps_eq = self.comps == other.comps
            spec_eq = self.spc_saft_spec == other.spc_saft_spec
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
        return hash((self.comps, self.spc_saft_spec, self.seg_num, self.seg_diam, self.disp_energy, self.k_ij(298.15)))


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
            if isinstance(value, sPCSAFTPhysInter):
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
        return self.phys_inter.spc_saft_spec.a[i, 0] + \
               self.phys_inter.spc_saft_spec.a[i, 1] * (self._m(ni) - 1.0) / self._m(ni) + \
               self.phys_inter.spc_saft_spec.a[i, 2] * (self._m(ni) - 1.0) * (self._m(ni) - 2.0) / (self._m(ni) ** 2.0)

    def _ai_i(self, i, ni):
        # de Villers (2011), F-76
        return self._m_i(ni) * (-4.0 * self.phys_inter.spc_saft_spec.a[i, 2] +
                                self._m(ni) * (self.phys_inter.spc_saft_spec.a[i, 1] +
                                               3.0 * self.phys_inter.spc_saft_spec.a[i, 2])) / self._m(ni) ** 3.0

    def _bi(self, i, ni):
        # de Villers (2011), Equation F-70
        return self.phys_inter.spc_saft_spec.b[i, 0] + \
               self.phys_inter.spc_saft_spec.b[i, 1] * (self._m(ni) - 1.0) / self._m(ni) + \
               self.phys_inter.spc_saft_spec.b[i, 2] * (self._m(ni) - 1.0) * (self._m(ni) - 2.0) / (self._m(ni) ** 2.0)

    def _bi_i(self, i, ni):
        # de Villers (2011), F-83
        return self._m_i(ni) * (-4.0 * self.phys_inter.spc_saft_spec.b[i, 2] +
                                self._m(ni) * (self.phys_inter.spc_saft_spec.b[i, 1] +
                                               3.0 * self.phys_inter.spc_saft_spec.b[i, 2])) / self._m(ni) ** 3.0

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
