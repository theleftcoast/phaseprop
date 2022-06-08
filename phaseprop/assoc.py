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
                 '4:4': (4, 4),
                 'variable:0': (None, 0)}

# Pre-defined association site types.
ASSOC_SITE_TYPES = ['electron donor',
                    'electron acceptor']


@dataclasses.dataclass(eq=True, frozen=True)
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
    sites: Dict{AssocSite, float}
        Keys are AssocSite objects (the actual sites themselves) and values are the number of association sits.

    Notes
    -----
    A functional group is an atom or a group of atoms that has similar chemical properties whenever it occurs in
    different compounds. It defines the characteristic physical and chemical properties of families of organic
    compounds _[1]. Association sites are commonly located on functional groups and generally have similar properties
    wherever they occurs in a chemical compounds. In this way, functional groups and association sites are tools to
    generalize our represenation of an organic molecule.

    The UNIFAC main group and subgroup system _[2] is common categorization scheme for organic functional groups. The
    AssocGroup object captures functional groups and other inorganic compounds capable of association.

    The number of association sites is most often set to integer values. The exception is aromatic electron acceptors
    where a fractional number of association sites (greater than one) is allowed to characterize components and
    petroleum fractions which are capable of accepting multiple hydrogen bonds _[3].

    References
    ----------
    [1] IUPAC. Compendium of Chemical Terminology, 2nd ed. (the "Gold Book"). Compiled by A. D. McNaught and A.
    Wilkinson. Blackwell Scientific Publications, Oxford (1997). Online version (2019-) created by S. J. Chalk. ISBN
    0-9678550-9-8. https://doi.org/10.1351/goldbook.
    [2] http://www.ddbst.com/published-parameters-unifac.html#ListOfSubGroupsAndTheirGroupSurfacesAndVolumes
    [3] Marshall, B. D. A PC-SAFT Model for Hydrocarbons IV: Water-Hydrocarbon Phase Behavior Including Petroleum
    Pseudo-Components. Fluid Phase Equilibria 2019, 497, 79–86.
    """
    desc: str
    scheme: str
    sites: typing.Dict[AssocSite, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.desc not in ASSOC_GROUPS:
            raise ValueError("Functional group not valid.")
        if self.scheme not in ASSOC_SCHEMES.keys():
            raise ValueError("Site type not valid.")

    def __str__(self):
        output = []
        output.append("Group: {}, Scheme: {}, Sites:\n".format(self.desc, self.scheme))
        for site, number in self.sites.items():
            if isinstance(site, AssocSite):
                output.append("    {}\n".format(tw.indent(str(site), "    ")))
        return "".join(output)

    def add_site(self, site: AssocSite, number: float = 1.0):
        self.sites[site] = number

    def sites_consistent(self):
        """Checks if sites are consistent with the association scheme."""
        ed = 0
        ea = 0
        for site, number in self.sites.items():
            if site.type == 'electron donor':
                ed += number
            if site.type == 'electron acceptor':
                ea += number
        if self.scheme == 'variable:0' and ed >= 1.0 and ea == 0:
            return True
        elif ASSOC_SCHEMES[self.scheme] == (ed, ea):
            return True
        else:
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
    energy: float
    volume: float
    source: typing.Optional[str] = None
    notes: typing.Optional[str] = None

    def __str__(self):
        return "Group A: {}, Site A: {}\n" \
               "Group B: {}, Site B: {}\n" \
               "Energy: {}\n" \
               "Volume: {}".format(self.site_a.parent_group.desc,
                                   self.site_a.desc,
                                   self.site_b.parent_group.desc,
                                   self.site_b.desc,
                                   self.energy,
                                   self.volume)


@dataclasses.dataclass
class AssocInter(object):
    """Association interactions between multiple components."""
    assoc_sites: typing.List[AssocSite] = dataclasses.field(init=False)
    assoc_site_inter: typing.List[AssocSiteInter] = dataclasses.field(init=False)

    def __post_init__(self):
        pass

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
        pass


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
