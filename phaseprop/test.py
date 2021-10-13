import unittest

import eos

# 2B component
comp1 = eos.Comp('comp1')
comp1.mw = 20.0
ed1_c1 = eos.AssocSite(comp1, 'a', 'ed')
ea1_c1 = eos.AssocSite(comp1, 'b', 'ea')

ed1_c1_ea1_c1 = eos.AssocSiteInter(ed1_c1, ea1_c1, 'CPA')

# 3B component
comp2 = eos.Comp('comp2')
comp2.mw = 25.0
ed1_c2 = eos.AssocSite(comp2, 'a', 'ed')
ed2_c2 = eos.AssocSite(comp2, 'b', 'ed')
ea1_c2 = eos.AssocSite(comp2, 'c', 'ea')

# 4C component
comp3 = eos.Comp('comp3')
comp3.mw = 30.0
ed1_c3 = eos.AssocSite(comp3, 'a', 'ed')
ed2_c3 = eos.AssocSite(comp3, 'b', 'ed')
ea1_c3 = eos.AssocSite(comp3, 'c', 'ea')
ea2_c3 = eos.AssocSite(comp3, 'd', 'ea')

# Various site types
comp4 = eos.Comp('comp4')
comp4.mw = 35.0
ed1_c4 = eos.AssocSite(comp4, 'a', 'ed')
ed2_c4 = eos.AssocSite(comp4, 'b', 'ed')
ea1_c4 = eos.AssocSite(comp4, 'c', 'ea')
ea1_alt_c4 = eos.AssocSite(comp4, 'a', 'ea')
ea2_c4 = eos.AssocSite(comp4, 'd', 'ea')
pi1_c4 = eos.AssocSite(comp4, 'e', 'pi_stack')
pi2_c4 = eos.AssocSite(comp4, 'f', 'pi_stack')
glue1_c4 = eos.AssocSite(comp4, 'g', 'glue')
glue2_c4 = eos.AssocSite(comp4, 'h', 'glue')

# Various component sets
cs1 = [comp1, comp2, comp3]
cs1_alt = [comp1, comp2, comp3]
cs2 = [comp1, comp2, None]
cs3 = [comp1, comp2, '123']

# Typical combinations of associating compounds.
assoc_2b_2b = [comp1, comp1]
assoc_2b_3b = [comp1, comp2]
assoc_2b_4c = [comp1, comp3]
assoc_3b_3b = [comp2, comp2]
assoc_3b_4c = [comp2, comp3]
assoc_4c_4c = [comp3, comp3]

# Binary interaction parameter
bip1 = eos.BinaryInterParm(comp1, comp2, eos.CUBIC_SPECS['CPA'], 'CPA')

# Various compositions
xi = [0.2, 0.3, 0.5]
xi_type = 'abc'
xi_element_type = [0.2, 0.3, 'abc']
xi_sum = [0.2, 0.3, 0.4]
xi_neg = [0.2, 0.3, -0.1]
xi_len = [0.2, 0.3]
n = 1.2
n_type = 'abc'
n_neg = -0.1
ni = [2.0, 3.0, 5.0]
ni_type = 'abc'
ni_element_type = [0.2, 0.3, 'abc']
ni_neg = [0.2, 0.3, -0.1]
ni_len = [0.2, 0.3]
wi = [0.2, 0.3, 0.5]
wi_type = 'abc'
wi_element_type = [0.2, 0.3, 'abc']
wi_sum = [0.2, 0.3, 0.4]
wi_neg = [0.2, 0.3, -0.1]
wi_len = [0.2, 0.3]
m = 7500.0
m_type = 'abc'
m_neg = -0.1
mi = [0.2, 0.25, 0.3]
mi_type = 'abc'
mi_element_type = [0.2, 0.3, 'abc']
mi_neg = [0.2, 0.3, -0.1]
mi_len = [0.2, 0.3]


class TestSum(unittest.TestCase):
    def test_AssocSite_equal(self):
        """Two instances of AssocSite are equal."""
        self.assertTrue(ed1_c4 == ed1_c4)

    def test_AssocSite_not_equal(self):
        """Two instances of AssocSite are equal."""
        # Equal except for type.
        case_a = ed1_c4 == ea1_alt_c4
        # Equal except for site.
        case_b = ed1_c4 == ed2_c4
        # Equal except for comp.
        case_c = ed1_c4 == ed1_c3
        self.assertFalse(case_a and case_b and case_c)

    def test_AssocSite_can_interact_true(self):
        """Test if the can_interact() function returns True for valid site interactions."""
        # Electron donor can interact with an electron acceptor.
        case_a = ed1_c4.can_interact(ea1_c4)
        # Aromatic pi site can interact with another aromatic pi site.
        case_b = pi1_c4.can_interact(pi2_c4)
        # Glue site can interact with another glue site.
        case_c = glue1_c4.can_interact(glue2_c4)
        self.assertTrue(case_a and case_b and case_c)

    def test_AssocSite_can_interact_false(self):
        """Test if the can_interact() function returns False for invalid site interactions."""
        # Electron donor cannot interact with an electron donor site.
        case_a = ed1_c4.can_interact(ed2_c4)
        # Electron donor cannot interact with an aromatic pi site.
        case_b = ed1_c4.can_interact(pi1_c4)
        # Electron donor cannot interact with a glue site.
        case_c = ed1_c4.can_interact(glue1_c4)
        # Electron acceptor cannot interact with an aromatic pi site.
        case_d = ea1_c4.can_interact(pi1_c4)
        # Electron acceptor cannot interact with a glue site.
        case_e = ea1_c4.can_interact(glue1_c4)
        # Aromatic pi site cannot interact with a glue site.
        case_f = pi1_c4.can_interact(glue1_c4)
        self.assertFalse(case_a and case_b and case_c and case_d and case_e and case_f)

    def test_AssocSite_comp_TypeError(self):
        """Test if TypeError is raised if comp is not an instance of Comp."""
        with self.assertRaises(TypeError):
            eos.AssocSite('abc', 'a', 'glue')

    def test_AssocSite_comp_ValueError(self):
        """Test if ValueError is raised if comp is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.AssocSite(None, 'a', 'glue')

    def test_AssocSite_site_TypeError(self):
        """Test if TypeError is raised if site is not a string."""
        with self.assertRaises(TypeError):
            eos.AssocSite(comp1, 123 , 'glue')

    def test_AssocSite_site_ValueError(self):
        """Test if ValueError is raised if site is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.AssocSite(comp1, None, 'glue')

    def test_AssocSite_type_ValueError(self):
        """Test if TypeError is raised if site is not a string."""
        with self.assertRaises(ValueError):
            eos.AssocSite(comp1, 'a', 'g')

    def test_AssocSite_type_ValueError_alt(self):
        """Test if ValueError is raised if type is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.AssocSite(comp1, 'a', None)

    def test_AssocSite_immutable(self):
        """Test if changing the comp, site, or type is possible."""
        # Changing comp should not change the AssocSite instance.
        original_value = ed1_c4.comp
        ed1_c4.comp = comp1
        changed_value = ed1_c4.comp
        case_a = original_value == changed_value

        # Changing site should not change the AssocSite instance.
        original_value = ed1_c4.site
        ed1_c4.comp = 'abc'
        changed_value = ed1_c4.site
        case_b = original_value == changed_value

        # Changing type should not change the AssocSite instance.
        original_value = ed1_c4.type
        ed1_c4.comp = 'glue'
        changed_value = ed1_c4.type
        case_c = original_value == changed_value
        self.assertTrue(case_a and case_b and case_c)

    def test_AssocSiteInter_equal(self):
        """Two instances of AssocSiteInter are equal if sites are interchanged."""
        ab = eos.AssocSiteInter(ed1_c1, ea1_c1, 'CPA')
        ba = eos.AssocSiteInter(ea1_c1, ed1_c1, 'CPA')
        self.assertTrue(ab == ba)

    def test_AssocSiteInter_site_a_ValueError(self):
        """Test if ValueError is raised if site_a is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.AssocSiteInter(None, ea1_c1, 'CPA')

    def test_AssocSiteInter_site_a_ValueError_alt(self):
        """Test if ValueError is raised if site_a is not an instance of AssocSite."""
        with self.assertRaises(ValueError):
            eos.AssocSiteInter('123', ea1_c1, 'CPA')

    def test_AssocSiteInter_site_b_ValueError(self):
        """Test if ValueError is raised if site_a is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.AssocSiteInter(ed1_c1, None, 'CPA')

    def test_AssocSiteInter_site_b_ValueError_alt(self):
        """Test if ValueError is raised if site_b is not an instance of AssocSite."""
        with self.assertRaises(ValueError):
            eos.AssocSiteInter(ed1_c1, '123', 'CPA')

    def test_AssocSiteInter_eos_ValueError(self):
        """Test if ValueError is raised if eos is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.AssocSiteInter(ed1_c1, ea1_c1, None)

    def test_AssocSiteInter_eos_ValueError_alt(self):
        """Test if ValueError is raised if eos is not valid."""
        with self.assertRaises(ValueError):
            eos.AssocSiteInter(ed1_c4, ea1_c4, '123')

    def test_AssocSiteInter_can_interact(self):
        """Test if ValueError is raised if two AssocSite objects are passed which can't interact."""
        with self.assertRaises(ValueError):
            eos.AssocSiteInter(ed1_c4, ed2_c4, None)

    def test_AssocSiteInter_immutable(self):
        """Test if changing the site_a, site_b, or eos is possible."""
        # Changing site_a should not change the AssocSiteInter instance.
        original_value = ed1_c1_ea1_c1.site_a
        ed1_c1_ea1_c1.site_a = ed1_c4
        changed_value = ed1_c1_ea1_c1.site_a
        case_a = original_value == changed_value

        # Changing site_b should not change the AssocSiteInter instance.
        original_value = ed1_c1_ea1_c1.site_b
        ed1_c1_ea1_c1.site_b = ea1_c4
        changed_value = ed1_c1_ea1_c1.site_b
        case_b = original_value == changed_value

        # Changing eos should not change the AssocSiteInter instance.
        original_value = ed1_c1_ea1_c1.eos
        ed1_c1_ea1_c1.eos = 'sPC-SAFT'
        changed_value = ed1_c1_ea1_c1.eos
        case_c = original_value == changed_value
        self.assertTrue(case_a and case_b and case_c)

    def test_CompSet_comps_ValueError(self):
        """Test if ValueError is raised if comps is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.CompSet(None)

    def test_CompSet_comps_TypeError(self):
        """Test if TypeError is raised if a member of comps is not an instance of Comp."""
        with self.assertRaises(TypeError):
            eos.CompSet(cs3)

    def test_CompSet_equal(self):
        """Test if two instances of CompSet are equal."""
        self.assertTrue(eos.CompSet(cs1) == eos.CompSet(cs1_alt))

    def test_AssocInter_eos_ValueError(self):
        """Test if ValueError is raised if eos is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.AssocInter(eos.CompSet(cs1), None)

    def test_AssocInter_eos_ValueError_alt(self):
        """Test if ValueError is raised if eos is not valid."""
        with self.assertRaises(ValueError):
            eos.AssocInter(eos.CompSet(cs1), '123')

    def test_AssocInter_comps_ValueError(self):
        """Test if ValueError is raised if comps is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.AssocInter(None, 'CPA')

    def test_AssocInter_comps_TypeError(self):
        """Test if Type is raised if comps is not an instance of CompSet."""
        with self.assertRaises(TypeError):
            eos.AssocInter('123', 'CPA')

    def test_BinaryInterParm_equal(self):
        """Two instances of BinaryInterParm are equal if comps are interchanged."""
        ab = eos.BinaryInterParm(comp1, comp2, eos.CUBIC_SPECS['CPA'], 'CPA')
        ba = eos.BinaryInterParm(comp2, comp1, eos.CUBIC_SPECS['CPA'], 'CPA')
        self.assertTrue(ab == ba)

    def test_BinaryInterParm_comp_a_ValueError(self):
        """Test if ValueError is raised if comp_a is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.BinaryInterParm(None, comp2, 'CPA')

    def test_BinaryInterParm_comp_a_ValueError_alt(self):
        """Test if ValueError is raised if comp_a is not an instance of Comp."""
        with self.assertRaises(ValueError):
            eos.BinaryInterParm('123', comp2, 'CPA')

    def test_BinaryInterParm_comp_b_ValueError(self):
        """Test if ValueError is raised if comp_b is not passed when creating a new instance."""
        with self.assertRaises(ValueError):
            eos.BinaryInterParm(comp1, None, 'CPA')

    def test_BinaryInterParm_comp_b_ValueError_alt(self):
        """Test if ValueError is raised if comp_b is not an instance of Comp."""
        with self.assertRaises(ValueError):
            eos.BinaryInterParm(comp1, '123', 'CPA')

    def test_BinaryInterParm_immutable(self):
        """Test if changing the comp_a, comp_b, or eos is possible."""
        # Changing site_a should not change the AssocSiteInter instance.
        original_value = bip1.comp_a
        bip1.comp_a = comp2
        changed_value = bip1.comp_a
        case_a = original_value == changed_value

        # Changing site_b should not change the AssocSiteInter instance.
        original_value = bip1.comp_b
        bip1.comp_b = comp1
        changed_value = bip1.comp_b
        case_b = original_value == changed_value

        # Changing eos should not change the AssocSiteInter instance.
        original_value = bip1.eos
        bip1.eos = 'sPC-SAFT'
        changed_value = bip1.eos
        case_c = original_value == changed_value

        self.assertTrue(case_a and case_b and case_c)

    def test_Compos_input_check_a(self):
        """Test if comps is not an instance of CompSet."""
        with self.assertRaises(TypeError):
            eos.Compos(comps='abc')

    def test_Compos_input_check_b(self):
        """Test if an incorrect combination of inputs raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), xi=xi, m=m)

    def test_Compos_ni_input_check_a(self):
        """Test if ni is the wrong type raises a TypeError."""
        with self.assertRaises(TypeError):
            eos.Compos(comps=eos.CompSet(cs1), ni=ni_type)

    def test_Compos_ni_input_check_b(self):
        """Test if ni has an element of the wrong type raises a TypeError."""
        with self.assertRaises(TypeError):
            eos.Compos(comps=eos.CompSet(cs1), ni=ni_element_type)

    def test_Compos_ni_input_check_c(self):
        """Test if ni has negative element raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), ni=ni_neg)

    def test_Compos_ni_input_check_d(self):
        """Test if ni is not the same length as the CompSet raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), ni=ni_len)

    def test_Compos_mi_input_check_a(self):
        """Test if mi is the wrong type raises a TypeError."""
        with self.assertRaises(TypeError):
            eos.Compos(comps=eos.CompSet(cs1), mi=mi_type)

    def test_Compos_mi_input_check_b(self):
        """Test if mi has an element of the wrong type raises a TypeError."""
        with self.assertRaises(TypeError):
            eos.Compos(comps=eos.CompSet(cs1), mi=mi_element_type)

    def test_Compos_mi_input_check_c(self):
        """Test if mi has negative element raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), mi=mi_neg)

    def test_Compos_mi_input_check_d(self):
        """Test if mi is not the same length as the CompSet raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), mi=mi_len)

    def test_Compos_xi_n_input_check_a(self):
        """Test if xi is the wrong type raises a TypeError."""
        with self.assertRaises(TypeError):
            eos.Compos(comps=eos.CompSet(cs1), xi=xi_type, n=n)

    def test_Compos_xi_n_input_check_b(self):
        """Test if xi has an element of the wrong type raises a TypeError."""
        with self.assertRaises(TypeError):
            eos.Compos(comps=eos.CompSet(cs1), xi=xi_element_type, n=n)

    def test_Compos_xi_n_input_check_c(self):
        """Test if xi has negative element raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), xi=xi_neg, n=n)

    def test_Compos_xi_n_input_check_d(self):
        """Test if xi is not the same length as the CompSet raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), xi=xi_len, n=n)

    def test_Compos_xi_n_input_check_e(self):
        """Test if xi does not sum to one raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), xi=xi_sum, n=n)

    def test_Compos_xi_n_input_check_f(self):
        """Test if n is the wrong type raises a TypeError."""
        with self.assertRaises(TypeError):
            eos.Compos(comps=eos.CompSet(cs1), xi=xi, n=n_type)

    def test_Compos_xi_n_input_check_g(self):
        """Test if n is negative raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), xi=xi, n=n_neg)

    def test_Compos_wi_m_input_check_a(self):
        """Test if wi is the wrong type raises a TypeError."""
        with self.assertRaises(TypeError):
            eos.Compos(comps=eos.CompSet(cs1), wi=wi_type, m=m)

    def test_Compos_wi_m_input_check_b(self):
        """Test if wi has an element of the wrong type raises a TypeError."""
        with self.assertRaises(TypeError):
            eos.Compos(comps=eos.CompSet(cs1), wi=wi_element_type, m=m)

    def test_Compos_wi_m_input_check_c(self):
        """Test if wi has negative element raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), wi=wi_neg, m=m)

    def test_Compos_wi_m_input_check_d(self):
        """Test if wi is not the same length as the CompSet raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), wi=wi_len, m=m)

    def test_Compos_wi_m_input_check_e(self):
        """Test if wi does not sum to one raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), wi=wi_sum, m=m)

    def test_Compos_wi_m_input_check_f(self):
        """Test if m is the wrong type raises a TypeError."""
        with self.assertRaises(TypeError):
            eos.Compos(comps=eos.CompSet(cs1), wi=wi, m=m_type)

    def test_Compos_wi_m_input_check_g(self):
        """Test if m is negative raises a ValueError."""
        with self.assertRaises(ValueError):
            eos.Compos(comps=eos.CompSet(cs1), wi=wi, m=m_neg)

    def test_Comps_ni_spec(self):
        """Test if xi and n spec functions correctly."""
        compos = eos.Compos(comps=eos.CompSet(cs1), ni=ni)
        ni_ok = all(compos.ni == [2.0, 3.0, 5.0])
        xi_ok = all(compos.xi == [0.2, 0.3, 0.5])
        n_ok = compos.n == 10.0
        mi_ok = all(compos.mi == [0.04, 0.075, 0.15])
        self.assertTrue(ni_ok and xi_ok and n_ok and mi_ok)

    def test_Comps_xi_n_pec(self):
        """Test if xi and n spec functions correctly."""
        compos = eos.Compos(comps=eos.CompSet(cs1), xi=xi, n=n)
        ni_ok = all(compos.ni == [0.24, 0.36, 0.6])
        xi_ok = all(compos.xi == [0.2, 0.3, 0.5])
        n_ok = compos.n == 1.2
        mi_ok = all(compos.mi == [0.0048, 0.009, 0.018])
        self.assertTrue(ni_ok and xi_ok and n_ok and mi_ok)

    def test_Comps_mi_spec(self):
        """Test if mi spec functions correctly."""
        compos = eos.Compos(comps=eos.CompSet(cs1), mi=mi)
        ni_ok = all(compos.ni == [10.0, 10.0, 10.0])
        xi_ok = all(compos.xi == [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        n_ok = compos.n == 30.0
        mi_ok = all(compos.mi == [0.2, 0.25, 0.3])
        print(compos.ni)
        print(compos.xi)
        print(compos.n)
        print(compos.mi)
        print(ni_ok)
        print(xi_ok)
        print(n_ok)
        print(mi_ok)
        self.assertTrue(ni_ok and xi_ok and n_ok and mi_ok)


if __name__ == '__main__':
    unittest.main()
