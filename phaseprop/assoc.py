"""Objects representing association sites and interactions."""

import numpy as np
import dataclasses
import comp
import eos

# Pre-defined equation of state (EOS) pick lists.
ASSOC_EOS = ['CPA', 'PC-SAFT', 'sPC-SAFT']


class AssocSite(object):
    """Association site."""
    def __init__(self, comp=None, site=None, type=None, desc=None):
        if comp is None or site is None or type is None:
            raise ValueError("comp, site, and type must be provided to create an instance of AssocSite.")
        else:
            self.comp = comp
            self.site = site
            self.type = type
            self.desc = desc

    @property
    def comp(self):
        return self._comp

    @comp.setter
    def comp(self, value):
        try:
            self._comp
        except AttributeError:
            if isinstance(value, comp.Comp):
                self._comp = value
            else:
                raise TypeError("Component must be an instance of Comp.")

    @property
    def site(self):
        return self._site

    @site.setter
    def site(self, value):
        try:
            self._site
        except AttributeError:
            if isinstance(value, str):
                self._site = value
            else:
                raise TypeError("Site must be a string.")

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        try:
            self._type
        except AttributeError:
            if value in ['ed', 'ea', 'glue', 'pi_stack']:
                self._type = value
            else:
                raise ValueError("Site type not valid.")

    def __eq__(self, other):
        if isinstance(other, AssocSite):
            comp_eq = self.comp == other.comp
            site_eq = self.site == other.site
            type_eq = self.type == other.type
            return comp_eq and site_eq and type_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.comp.name, self.site, self.type))

    def __str__(self):
        return "{}, {}, {}".format(self.comp.name, self.site, self.type)

    def can_interact(self, other):
        if isinstance(other, AssocSite):
            if self.type == 'ea' and other.type == 'ed':
                return True
            elif self.type == 'ed' and other.type == 'ea':
                return True
            elif self.type == 'glue' and other.type == 'glue':
                return True
            elif self.type == 'pi_stack' and other.type == 'pi_stack':
                return True
            else:
                return False
        return False


@dataclasses.dataclass
class AssocSiteSet(object):
    """Set of Association sites related to an individual component.

    Attributes
    ----------
    assoc_sites : list of AssocSite
        List of AssocSites that are a part of the AssocSiteSet instance.
    size : int
        The number of Comp or PseudoComp objects in 'assoc_sites'.
    """
    assoc_sites: list[AssocSite]

    def __post_init__(self):
        if len(self.assoc_sites) != len(set(self.assoc_sites)):
            raise ValueError("Comp objects must be a list of unique assoc_site objects.")

    @property
    def size(self):
        """int : The number of AssocSite objects in 'assoc_sites'."""
        return len(self.assoc_sites)


class AssocSiteInter(object):
    """Interaction between two association sites."""
    def __init__(self, site_a=None, site_b=None, eos=None, source=None,
                 assoc_energy=None, assoc_vol=None):
        if site_a is None or site_b is None or eos is None:
            raise ValueError("site_a, site_b, and eos must be provided to create an instance of AssocSite.")
        elif not isinstance(site_a, AssocSite) or not isinstance(site_b, AssocSite):
            raise ValueError("Association sites must be an instance of AssocSite.")
        elif not site_a.can_interact(site_b):
            raise ValueError("Association sites must be able to interact.")
        else:
            self.site_a = site_a
            self.site_b = site_b
            self.eos = eos
            self.source = source
            self.assoc_energy = assoc_energy
            self.assoc_vol = assoc_vol

    @property
    def site_a(self):
        return self._site_a

    @site_a.setter
    def site_a(self, value):
        try:
            self._site_a
        except AttributeError:
            self._site_a = value

    @property
    def site_b(self):
        return self._site_b

    @site_b.setter
    def site_b(self, value):
        try:
            self._site_b
        except AttributeError:
            self._site_b = value

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, value):
        try:
            self._eos
        except AttributeError:
            # TODO: Convert to checking for instance of CPA or sPC-SAFT.
            if value in ASSOC_EOS:
                self._eos = value
            else:
                raise ValueError("eos is not valid.")

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
    def assoc_energy(self):
        return self._assoc_energy

    @assoc_energy.setter
    def assoc_energy(self, value):
        if value is None:
            self._assoc_energy = value
        elif isinstance(value, float) and value >= 0.0:
            self._assoc_energy = value
        else:
            raise ValueError("assoc_energy must be a positive float.")

    @property
    def assoc_vol(self):
        return self._assoc_vol

    @assoc_vol.setter
    def assoc_vol(self, value):
        if value is None:
            self._assoc_vol = value
        elif isinstance(value, float) and value >= 0.0:
            self._assoc_vol = value
        else:
            raise ValueError("assoc_vol must be a positive float.")

    def __eq__(self, other):
        if isinstance(other, AssocSiteInter):
            aa_bb_eq = self.site_a == other.site_a and self.site_b == other.site_b
            ab_ba_eq = self.site_a == other.site_b and self.site_b == other.site_a
            eos_eq = self.eos == other.eos
            return (aa_bb_eq or ab_ba_eq) and eos_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.site_a), hash(self.site_b), self.eos))

    def __str__(self):
        return "Site A: ({}), Site B: ({}), EOS: {}, Source: {}, Volume: {}, Energy: {}".format(self.site_a,
                                                                                                self.site_b,
                                                                                                self.eos,
                                                                                                self.source,
                                                                                                self.assoc_vol,
                                                                                                self.assoc_energy)


class AssocInter(object):
    """Association interactions between multiple components."""
    def __init__(self, comps=None, eos=None, adj_parm_spec='none'):
        if comps is None or eos is None:
            raise ValueError("comps and eos must be provided to create an instance of AssocSite.")
        self.comps = comps
        self.eos = eos
        self.assoc_sites = self._create_assoc_sites()
        self.assoc_site_inter = self._create_assoc_site_inters()
        self._adj_assoc_energy_members = []
        self._adj_assoc_vol_members = []
        self.set_adj_parms(adj_parm_spec=adj_parm_spec)

    @property
    def comps(self):
        return self._comps

    @comps.setter
    def comps(self, value):
        try:
            self._comps
        except AttributeError:
            if isinstance(value, comp.CompSet):
                self._comps = value
            else:
                raise TypeError("comps must be an instance of CompSet.")

    @property
    def eos(self):
        # TODO: consider dropping the EOS spec...what is it used for anyway?
        return self._eos

    @eos.setter
    def eos(self, value):
        try:
            self._eos
        except AttributeError:
            if value in ASSOC_EOS:
                self._eos = value
            else:
                raise ValueError("eos is not valid.")

    @property
    def assoc_sites(self):
        return self._assoc_sites

    @assoc_sites.setter
    def assoc_sites(self, value):
        try:
            self._assoc_sites
        except AttributeError:
            self._assoc_sites = value

    @property
    def assoc_site_inter(self):
        return self._assoc_site_inter

    @assoc_site_inter.setter
    def assoc_site_inter(self, value):
        try:
            self._assoc_site_inter
        except AttributeError:
            self._assoc_site_inter = value

    @property
    def adj_assoc_energy_members(self):
        return self._adj_assoc_energy_members

    @property
    def adj_assoc_energy(self):
        ae = []
        for asi in self._adj_assoc_energy_members:
            ae.append(asi.assoc_energy)
        return ae

    @adj_assoc_energy.setter
    def adj_assoc_energy(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_assoc_energy_members):
            for i, asi in enumerate(self._adj_assoc_energy_members):
                if isinstance(value[i], float):
                    asi.assoc_energy = value[i]
                else:
                    raise ValueError("adj_assoc_energy values must be floats.")
        else:
            raise TypeError("adj_assoc_energy must be a list or tuple with {} elements."
                            .format(len(self._adj_assoc_energy_members)))

    @property
    def adj_assoc_vol_members(self):
        return self._adj_assoc_vol_members

    @property
    def adj_assoc_vol(self):
        av = []
        for asi in self._adj_assoc_vol_members:
            av.append(asi.assoc_vol)
        return av

    @adj_assoc_vol.setter
    def adj_assoc_vol(self, value):
        if isinstance(value, (list, tuple)) and len(value) == len(self._adj_assoc_vol_members):
            for i, asi in enumerate(self._adj_assoc_vol_members):
                if isinstance(value[i], float):
                    asi.assoc_vol = value[i]
                else:
                    raise ValueError("adj_assoc_vol values must be floats.")
        else:
            raise TypeError("adj_assoc_vol must be a list or tuple with {} elements."
                            .format(len(self._adj_assoc_vol_members)))

    def set_adj_parms(self, adj_parm_spec, incl_pure_comp=False):
        # Define which parameters are adjustable for each AssocSiteInter in assoc_site_inter.  'incl_pure_comp' changes
        # whether or not pure component assoc_vol or assoc_energy are adjustable in multicomponent mixtures (note that
        # pure component assoc_vol or assoc_energy are always adjustable in a single component mixture). Reset all
        # 'member' lists to empty lists as a default every time this method is called.
        self._adj_assoc_energy_members = []
        self._adj_assoc_vol_members = []
        if adj_parm_spec == 'all':
            self._adj_assoc_energy_members = self._assoc_site_inter[:]
            self._adj_assoc_vol_members = self._assoc_site_inter[:]
        elif adj_parm_spec == 'none':
            pass
        elif adj_parm_spec == 'assoc_energy':
            if incl_pure_comp is False and len(self._comps.comps) > 1:
                for asi in self._assoc_site_inter:
                    if asi.site_a.comp != asi.site_b.comp:
                        self._adj_assoc_energy_members.append(asi)
            else:
                self._adj_assoc_vol_members = self._assoc_site_inter[:]
        elif adj_parm_spec == 'assoc_vol':
            if incl_pure_comp is False and len(self._comps.comps) > 1:
                for asi in self._assoc_site_inter:
                    if asi.site_a.comp != asi.site_b.comp:
                        self._adj_assoc_vol_members.append(asi)
            else:
                self._adj_assoc_vol_members = self._assoc_site_inter[:]
        else:
            raise ValueError("adj_parm_spec not valid.")

    # TODO: This already implicitly does not allow a Comp to have two association sites.  However, it would be good to
    #  build this check directly into the specification for Comp and give the user a warning if there is a problem.
    def _create_assoc_sites(self):
        # Create list of all sites.
        result = []
        for comp in self._comps.comps:
            if comp.assoc_sites is not None:
                for site in comp.assoc_sites:
                    if site not in result:
                        result.append(site)
        return result

    def _create_assoc_site_inters(self):
        # Create list of all possible site-to-site interactions as AssocSiteInter objects.
        result = []
        for comp_i in self._comps.comps:
            for comp_j in self._comps.comps:
                if comp_i.assoc_sites is not None and comp_j.assoc_sites is not None:
                    for site_a in comp_i.assoc_sites:
                        for site_b in comp_j.assoc_sites:
                            if site_a.can_interact(site_b):
                                new = AssocSiteInter(site_a, site_b, self._eos)
                                if new not in result:
                                    result.append(new)
        return result

    def load_pure_comp(self):
        # Load source, assoc_vol, and assoc_energy from Comp objects.
        for comp in self._comps.comps:
            if comp.assoc_sites is not None:
                if self._eos == 'CPA':
                    self.load_site_inter(comp.cpa_assoc)
                elif self._eos == 'sPC-SAFT':
                    self.load_site_inter(comp.spc_saft_assoc)
                else:
                    raise ValueError("eos not valid.")

    def load_site_inter(self, input):
        # Load source, assoc_vol, and assoc_energy from list of AssocSiteInter ('asi') objects.
        if isinstance(input, AssocSiteInter):
            for _asi in self._assoc_site_inter:
                if input == _asi:
                    _asi.source = input.source
                    _asi.assoc_vol = input.assoc_vol
                    _asi.assoc_energy = input.assoc_energy
        elif isinstance(input, AssocInter):
            for asi in input.assoc_site_inter:
                for _asi in self._assoc_site_inter:
                    if asi == _asi:
                        _asi.source = asi.source
                        _asi.assoc_vol = asi.assoc_vol
                        _asi.assoc_energy = asi.assoc_energy
        elif isinstance(input, list):
            for item in input:
                if isinstance(item, AssocSiteInter):
                    for _asi in self._assoc_site_inter:
                        if _asi == item:
                            _asi.source = item.source
                            _asi.assoc_vol = item.assoc_vol
                            _asi.assoc_energy = item.assoc_energy
                else:
                    raise TypeError("Must pass a list of AssocSiteInter objects.")
        else:
            raise TypeError("Input not AssocSiteInter object, AssocInter object, or list of AssocSiteInter objects.")

    def assoc_energy(self):
        # Build array of association energy for all site combinations.
        ae = np.zeros((len(self.assoc_sites), len(self.assoc_sites)))
        for site in self._assoc_site_inter:
            i = self._assoc_sites.index(site.site_a)
            j = self._assoc_sites.index(site.site_b)
            ae[i, j] = site.assoc_energy
            ae[j, i] = site.assoc_energy
        return ae

    def assoc_vol(self):
        # Build array of association volumes for all site combinations.
        av = np.zeros((len(self.assoc_sites), len(self.assoc_sites)))
        for site in self._assoc_site_inter:
            i = self._assoc_sites.index(site.site_a)
            j = self._assoc_sites.index(site.site_b)
            av[i, j] = site.assoc_vol
            av[j, i] = site.assoc_vol
        return av

    def delta(self, g=None, b=None, d=None, t=None):
        # Build array of association strengths
        # g is the radial distribution function
        # Accociation energy is epsilon/KB.
        # np.multiply(self._assoc_vol(), self._assoc_energy())
        # t_dep_term = np.exp(self._assoc_energy()/(KB * t)) - 1.0
        if self._eos == 'CPA':
            if isinstance(g, float) and isinstance(b, np.ndarray) and isinstance(t, float):
                return np.exp(self.assoc_energy()/t) - 1.0
            else:
                raise TypeError('g and t must be floats and b_ij must be an np.ndarray')
        if self._eos in ['sPC-SAFT', 'PC-SAFT']:
            if isinstance(g, float) and isinstance(d, np.ndarray) and isinstance(t, float):
                return np.exp(self.assoc_energy()/t) - 1.0
            else:
                raise TypeError('g and t must be floats and d_ij must be an np.ndarray')

    def __eq__(self, other):
        if isinstance(other, AssocInter):
            comps_eq = self.comps == other.comps
            assoc_energy_eq = np.array_equal(self.assoc_energy(), other.assoc_energy())
            assoc_vol_eq = np.array_equal(self.assoc_vol(), other.assoc_vol())
            return comps_eq and assoc_energy_eq and assoc_vol_eq
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.comps, self.assoc_energy(), self.assoc_vol()))


class Assoc(eos.EOS):
    """Base class implementation of a general associating term (EOS).

    Notes
    -----
    The association term is implemented using the Q-function approach proposed by Michelsen and Hendriks [1]_. The value
    of Q at a stationary point is:

    Qsp = Ar(T, V, ni, xai)/(R*T) = F

    Thermodynamic properties can be estimated by taking derivatives of Qsp.  In all of these expressions, xai is the
    fraction of 'a' sites on molecule 'i' that do not form bonds with other active sites.  xai is a list of lists where
    the structure is as follows:

    xai = [[xa for  each site in Comp 1], [xa for each site in Comp 2], ..., [xa for each site in Comp n]]

    The length of xai is equal to the length of the associated CompSet (i.e. xai and ni have the same length).  The
    length of each xai element corresponds to the length of the assoc_sites attribute of each Comp in the associated
    CompSet.  If there are no sites on a molecule, then the corresponding element's truth value must be False (which is
    represented by None this library for simplicity).

    References
    -----
    [1] Michelsen, M. L.; Hendriks, E. M. Physical properties from association models. Fluid Phase Equilib. 2001, 180,
    165-174.
    [2] Kontogeorgis, G. M.; Folas, G. K. Thermodynamic Models for Industrial Applications, John Wiley & Sons, Ltd:
    West Sussex, UK, 2010.
    [3] de Villiers, A. J. Evaluation and improvement of the sPC-SAFT equation of state for complex mixtures. Ph.D.
    Dissertation, Stellenbosch University, Stellenbosch, South Africa, 2011.
    """

    def _qsp(self, t, v, ni, xai):
        return

    def _qsp_v(self, t, v, ni, xai):
        return

    def _qsp_t(self, t, v, ni, xai):
        return

    def _qsp_i(self, t, v, ni, xai):
        return

    def _ln_g_v(self, t, v, ni):
        return

    def _ln_g_i(self, t, v, ni):
        # This function MUST be defined in any child class that inherits the Association base class.
        return
