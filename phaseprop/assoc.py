"""Objects representing association sites and association interactions.

Attributes
----------
ASSOC_GROUPS : list
    Pre-defined association groups.
ASSOC_SCHEMES : dictionary
    Pre-defined association schemes.
ASSOC_SITE_TYPES : list
    Pre-defined association site types.
"""

import numpy as np
import textwrap as tw
import typing
import dataclasses
import comp
import eos


# Functional groups and unique compounds that contain association sites.
ASSOC_GROUPS = ['alkene',
                'alkyne',
                'aromatic',
                'primary alcohol',  # General groups with oxygen
                'secondary alcohol',
                'aromatic alcohol',
                'ether',
                'aldehyde',
                'ketone',
                'carboxylic acid',
                'ester',
                'anhydride',
                'carbonate',
                'primary amine',  # General groups with nitrogen
                'secondary amine',
                'tertiary amine',
                'aromatic amine',
                'nitrile',
                'aromatic nitrile',
                'pyridine',
                'thiol',  # General groups with sulfur
                'sulfide',
                'sulphone',
                'thiophene',
                'formamide',  # General groups with oxygen and nitrogen
                'acetamide',
                'morpholine',
                'nitro',
                'aromatic nitro',
                'water',  # Unique compounds with oxygen
                'methanol',
                'ethylene glycol',
                'furfural',
                'carbon dioxide',
                'ammonia',  # Uniquie compounds with nitrogen
                'methylpyrrolidone',
                'hydrogen sulfide',  # Unique compounds with sulfur
                'carbon dislufide',
                'carbonyl sulfide',
                'nitrogen',  # Other unique compounds
                'methane']

# Notation is 'ED:EA' and (ED, EA) (where ED is electron donor and EA is electron acceptor).
ASSOC_SCHEMES = {'1:0': (1, 0),
                 '2:0': (2, 0),
                 '3:0': (3, 0),
                 '4:0': (4, 0),
                 '1:1': (1, 1),
                 '2:1': (2, 1),
                 '3:1': (3, 1),
                 '4:1': (4, 1),
                 '1:2': (1, 2),
                 '2:2': (2, 2),
                 '3:2': (3, 2),
                 '4:2': (4, 2),
                 '1:3': (1, 3),
                 '2:3': (2, 3),
                 '3:3': (3, 3),
                 '4:3': (4, 3),
                 '1:4': (1, 4),
                 '2:4': (2, 4),
                 '3:4': (3, 4),
                 '4:4': (4, 4)}

ASSOC_SITE_TYPES = ['electron donor',
                    'electron acceptor']


@dataclasses.dataclass
class AssocSite(object):
    """Site capable of associating.

    Attributes
    ----------
    desc : str
        Detailed description of the site (atom, electron pairs, etc.).
    type : str
        Interaction characteristic of the site (electron donor, electron acceptor).
    """
    desc: str
    type: str
    parent_group: ... = None

    def __post_init__(self):
        if self.type not in ASSOC_SITE_TYPES:
            raise ValueError("Site type not valid.")

    def __str__(self):
        return "Description: {}, Type: {}".format(self.desc, self.type)

    def can_interact(self, other):
        if isinstance(other, AssocSite):
            if self.type == 'electron acceptor' and other.type == 'electron donor':
                return True
            elif self.type == 'electron donor' and other.type == 'electron acceptor':
                return True
            else:
                return False
        return False


@dataclasses.dataclass
class AssocGroup(object):
    """Functional grouping of atoms capable of associating.

    Attributes
    ----------
    desc : str
        Description of the functional group that hosts the association sites.
    scheme : str
        Description of the association site configuration.
    sites: List[AssocSite]
        List of related AssocSite objects (the actual sites themselves).

    Notes
    -----
    A functional group is an atom or a group of atoms that has similar chemical properties whenever it occurs in
    different compounds. It defines the characteristic physical and chemical properties of families of organic
    compounds _[1]. Association sites are commonly located on functional groups and generally have similar properties
    wherever they occurs in a chemical compounds. In this way, functional groups and association sites are tools to
    generalize behavior for organic molecules.

    UNIFAC main group and subgroup system _[2] is common categorization scheme for organic functional groups. The
    AssocGroup object captures functional groups and other inorganic compounds capable of association.

    References
    ----------
    [1] IUPAC. Compendium of Chemical Terminology, 2nd ed. (the "Gold Book"). Compiled by A. D. McNaught and A.
    Wilkinson. Blackwell Scientific Publications, Oxford (1997). Online version (2019-) created by S. J. Chalk. ISBN
    0-9678550-9-8. https://doi.org/10.1351/goldbook.
    [2] http://www.ddbst.com/published-parameters-unifac.html#ListOfSubGroupsAndTheirGroupSurfacesAndVolumes
    """
    desc: str
    scheme: str
    sites: typing.List[AssocSite] = dataclasses.field(default_factory=list)
    parent_comp: ... = None

    def __post_init__(self):
        if self.desc not in ASSOC_GROUPS:
            raise ValueError("Functional group not valid.")
        if self.scheme not in ASSOC_SCHEMES.keys():
            raise ValueError("Site type not valid.")

    def __str__(self):
        output = []
        output.append("Group: {}, Scheme: {}, Sites:\n".format(self.desc, self.scheme))
        for site in self.sites:
            if isinstance(site, AssocSite):
                output.append("    {}\n".format(tw.indent(str(site), "    ")))
        return "".join(output)

    def add_site(self, site):
        if isinstance(site, AssocSite):
            site.parent_group = self
            self.sites.append(site)
        else:
            raise TypeError('site is not an AssocSite')

    def sites_consistent(self):
        """Checks if sites are consistent with the association scheme."""
        ed = 0
        ea = 0
        for site in self.sites:
            if site.type == 'electron donor':
                ed += 1
            if site.type == 'electron acceptor':
                ea += 1
        if ASSOC_SCHEMES[self.scheme] == (ed, ea):
            return True
        return False


@dataclasses.dataclass
class AssocSiteInter(object):
    """Interaction between two association sites.

    Attributes
    ----------
    site_a : AssocSite
        AssocSite participating in the interaction.
    site_b : AssocSite
        AssocSite participating in the interaction.
    assoc_energy : float
        Association energy between sites.
    assoc_vol : float
        Association volume between sites.
    source : str, optional
        Source for the association interaction parameters (ACS citation format preferred).
    notes : str, optional
        Notes associated with the correlation.
    """
    site_a: AssocSite
    site_b: AssocSite
    assoc_energy: float
    assoc_vol: float
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def __str__(self):
        return "Comp A: {}, Group A: {}, Site A: {}\n" \
               "Comp B: {}, Group B: {}, Site B: {}\n" \
               "Energy: {}\n" \
               "Volume: {}".format(self.site_a.parent_group.parent_comp.name,
                                         self.site_a.parent_group.desc,
                                         self.site_a.desc,
                                         self.site_b.parent_group.parent_comp.name,
                                         self.site_b.parent_group.desc,
                                         self.site_b.desc,
                                         self.assoc_energy,
                                         self.assoc_vol)


@dataclasses.dataclass
class AssocInterReforged(object):
    """Association interactions between multiple components."""
    comps: comp.CompSet
    assoc_sites: typing.List[comp.AssocSites] = dataclasses.field(init=False)
    assoc_site_inter: typing.List[comp.AssocSiteInter] = dataclasses.field(init=False)

    def __post_init__(self):
        pass

    def _create_assoc_sites(self):
        # Create list of all sites.
        result = []
        for comp in self._comps.comps:
            if comp.assoc_sites is not None:
                for site in comp.assoc_sites:
                    if site not in result:
                        result.append(site)

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
