"""Multiphase equilibrium and thermophysical property estimation."""

import comps

if __name__ == "__main__":
    # Define association sites corresponding to water as a symmetrically associating 4C molecule.
    # ea1 = AssocSite(comp=water, site='H1', type='ea')
    # ea2 = AssocSite(comp=water, site='H2', type='ea')
    # ed1 = AssocSite(comp=water, site='O1', type='ed')
    # ed2 = AssocSite(comp=water, site='O2', type='ed')
    #
    # water.assoc_sites = [ea1, ea2, ed1, ed2]
    #
    # ea1_ed1 = AssocSiteInter(site_a=ea1, site_b=ed1,
    #                          eos='sPC-SAFT',
    #                          source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
    #                          assoc_energy=1704.06, assoc_vol=0.3048)
    # ea1_ed2 = AssocSiteInter(site_a=ea1, site_b=ed2,
    #                          eos='sPC-SAFT',
    #                          source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
    #                          assoc_energy=1704.06, assoc_vol=0.3048)
    # ea2_ed1 = AssocSiteInter(site_a=ea2, site_b=ed1,
    #                          eos='sPC-SAFT',
    #                          source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
    #                          assoc_energy=1704.06, assoc_vol=0.3048)
    # ea2_ed2 = AssocSiteInter(site_a=ea2, site_b=ed2,
    #                          eos='sPC-SAFT',
    #                          source="Ind. Eng. Chem. Res. 2014, 53, 14493−14507.",
    #                          assoc_energy=1704.06, assoc_vol=0.3048)
    #
    # water.spc_saft_assoc = [ea1_ed1, ea1_ed2, ea2_ed1, ea2_ed2]

    # Information from DIPPR tables presented in Perry's Chemical Engineer's Handbook, 9th ed.
    # methanol = Comp('Methanol')
    # methanol.formula = "CH3OH"
    # methanol.family = "alcohol"
    # methanol.cas_no = "67-56-1"
    # methanol.mw = 32.04186
    # methanol.tc = 512.5
    # methanol.pc = 8084000.0
    # methanol.vc = 0.000117
    # methanol.acentric = 0.565831

    # methanol.spc_saft_phys = sPCSAFTParms(comp=methanol,
    #                                      spc_saft_spec=GS,
    #                                      source="Ind. Eng. Chem. Res. 2012, 45, 14903–14914.",
    #                                      seg_num=1.88238, seg_diam=3.0023, disp_energy=181.77)

    # # Define association sites corresponding to methanol as a 2B molecule.
    # ea1 = AssocSite(comp=methanol, site='H1', type='ea')
    # ed1 = AssocSite(comp=methanol, site='O1', type='ed')
    #
    # methanol.assoc_sites = [ea1, ed1]
    #
    # ea1_ed1 = AssocSiteInter(site_a=ea1, site_b=ed1,
    #                          eos='sPC-SAFT',
    #                          source="Ind. Eng. Chem. Res. 2012, 45, 14903–14914.",
    #                          assoc_energy=2738.03, assoc_vol=0.054664)
    #
    # mm = BinaryInterParm(comp_a=methane, comp_b=methane,
    #                      temp_indep_coef=0.0)
    #
    # me = BinaryInterParm(comp_a=methane, comp_b=ethane,
    #                      temp_indep_coef=0.03)
    #
    # mp = BinaryInterParm(comp_a=methane, comp_b=propane,
    #                      temp_indep_coef=0.03)
    #
    # ee = BinaryInterParm(comp_a=ethane, comp_b=ethane,
    #                      temp_indep_coef=0.0)
    #
    # ep = BinaryInterParm(comp_a=ethane, comp_b=propane,
    #                      temp_indep_coef=0.01)
    #
    # pp = BinaryInterParm(comp_a=propane, comp_b=propane,
    #                      temp_indep_coef=0.0)
    #
    # wm = BinaryInterParm(comp_a=water, comp_b=methane,
    #                      source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
    #                      temp_indep_coef=0.2306,
    #                      inv_temp_coef=-92.62)
    #
    # we = BinaryInterParm(comp_a=water, comp_b=ethane,
    #                      source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
    #                      temp_indep_coef=0.1773,
    #                      inv_temp_coef=-53.97)
    #
    # wp = BinaryInterParm(comp_a=water, comp_b=propane,
    #                      source='initial guess',
    #                      temp_indep_coef=0.05)
    #
    # mem = BinaryInterParm(comp_a=methanol, comp_b=methane,
    #                       source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
    #                       temp_indep_coef=0.01)
    #
    # mee = BinaryInterParm(comp_a=methanol, comp_b=ethane,
    #                       source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
    #                       temp_indep_coef=0.02)
    #
    # mep = BinaryInterParm(comp_a=methanol, comp_b=propane,
    #                       source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
    #                       temp_indep_coef=0.02)
    #
    # wme = BinaryInterParm(comp_a=water, comp_b=methanol,
    #                       source='J. Chem. Eng. Data 2017, 62, 2592–2605.',
    #                       temp_indep_coef=-0.066)
    #
    # cs = CompSet(comps=[ethane, propane])
    #
    # spc_saft_phys = sPCSAFTPhysInter(comps=cs, spc_saft_spec=GS)
    # spc_saft_phys.load_pure_comp_parms()
    # spc_saft_phys.load_binary_parms([ee, ep, pp])
    #
    # print("---------------")
    # print("seg_num: {}".format(spc_saft_phys.seg_num))
    # print("seg_diam: {}".format(spc_saft_phys.seg_diam))
    # print("disp_energy: {}".format(spc_saft_phys.disp_energy))
    # print("k_ij: {}".format(spc_saft_phys.k_ij(298.15)))
    #
    # spc_saft = sPCSAFT(phys_inter=spc_saft_phys, assoc_inter=None)
    #
    # phase = Phase(comps=cs, eos=spc_saft)
    # phase.compos.set(xi=[0.2, 0.8])
    # phase.state.spec = 'PT'
    #
    # phase.state.set(p=50000000.0, t=400.0)
    # p = phase.state.p
    # vm = phase.state.vm
    # t = phase.state.t
    # ni = phase.compos.xi
    # mw = phase.props.mw
    # i = 0
    # xai = None
    #
    # # Checking the values for all functions that make up sPC-SAFT.  The most important check is that the resulting f, f_v,
    # # and f_i values equal their numeric derivatives. The
    # print("---------------")
    # print("p = {}".format(p))
    # print("t = {}".format(t))
    # print("vm = {}".format(vm))
    # print("mw = {}".format(phase.props.mw))
    # print("rho_m = {}".format(phase.props.rho_m))
    # print("ni = {}".format(ni))
    # print("rho = {}".format(phase.eos._rho(vm, ni)))
    # print("v_a3 = {}".format(phase.eos._v_a3(vm)))
    # print("m = {}".format(phase.eos._m(ni)))
    # print("m_i = {}".format(phase.eos._m_i(ni)))
    # print("di = {}".format(phase.eos._di(t)))
    # print("d3 = {}".format(phase.eos._d3(t, ni)))
    # print("d_i = {}".format(phase.eos._d_i(t, ni)))
    # print("eta = {}".format(phase.eos._eta(t, vm, ni)))
    # print("eta_v = {}".format(phase.eos._eta_v(t, vm, ni)))
    # print("eta_i = {}".format(phase.eos._eta_i(t, vm, ni)))
    # print("ghs = {}".format(phase.eos._ghs(t, vm, ni)))
    # print("ghs_v = {}".format(phase.eos._ghs_v(t, vm, ni)))
    # print("ghs_i = {}".format(phase.eos._ghs_i(t, vm, ni)))
    # print("i1 = {}".format(phase.eos._i1(t, vm, ni)))
    # print("i1_v = {}".format(phase.eos._i1_v(t, vm, ni)))
    # print("i1_i = {}".format(phase.eos._i1_i(t, vm, ni)))
    # print("i2 = {}".format(phase.eos._i2(t, vm, ni)))
    # print("i2_v = {}".format(phase.eos._i2_v(t, vm, ni)))
    # print("i2_i = {}".format(phase.eos._i2_i(t, vm, ni)))
    # print("eps_ij = {}".format(phase.eos._epsij(t)))
    # print("sig_ij = {}".format(phase.eos._sigij()))
    # print("ai_i = {}".format(phase.eos._ai_i(2, ni)))
    # print("bi_i = {}".format(phase.eos._bi_i(2, ni)))
    # print("m2eps1sig3 = {}".format(phase.eos._m2eps1sig3(t, ni)))
    # print("m2eps1sig3_i = {}".format(phase.eos._m2eps1sig3_i(t, ni)))
    # print("m2eps2sig3 = {}".format(phase.eos._m2eps2sig3(t, ni)))
    # print("m2eps2sig3_i = {}".format(phase.eos._m2eps2sig3_i(t, ni)))
    # print("c0 = {}".format(phase.eos._c0(t, vm, ni)))
    # print("c0_v = {}".format(phase.eos._c0_v(t, vm, ni)))
    # print("c0_i = {}".format(phase.eos._c0_i(t, vm, ni)))
    # print("c1 (as inverse of c0) = {}".format(1.0 / phase.eos._c0(t, vm, ni)))
    # print("c1 = {}".format(phase.eos._c1(t, vm, ni)))
    # print("c1_v = {}".format(phase.eos._c1_v(t, vm, ni)))
    # print("c1_i = {}".format(phase.eos._c1_i(t, vm, ni)))
    # print("fhs = {}".format(phase.eos._fhs(t, vm, ni)))
    # print("fhs_v = {}".format(phase.eos._fhs_v(t, vm, ni)))
    # print("fhs_i = {}".format(phase.eos._fhs_i(t, vm, ni)))
    # print("fhc = {}".format(phase.eos._fhc(t, vm, ni)))
    # print("fhc_v = {}".format(phase.eos._fhc_v(t, vm, ni)))
    # print("fhc_i = {}".format(phase.eos._fhc_i(t, vm, ni)))
    # print("fdisp = {}".format(phase.eos._fdisp(t, vm, ni)))
    # print("fdisp1 = {}".format(phase.eos._fdisp1(t, vm, ni)))
    # print("fdisp1_v = {}".format(phase.eos._fdisp1_v(t, vm, ni)))
    # print("fdisp1_i = {}".format(phase.eos._fdisp1_i(t, vm, ni)))
    # print("fdisp2 = {}".format(phase.eos._fdisp2(t, vm, ni)))
    # print("fdisp2_v = {}".format(phase.eos._fdisp2_v(t, vm, ni)))
    # print("fdisp2_i = {}".format(phase.eos._fdisp2_i(t, vm, ni)))
    # print("fdisp_v = {}".format(phase.eos._fdisp_v(t, vm, ni)))
    # print("fdisp_i = {}".format(phase.eos._fdisp_i(t, vm, ni)))
    # print("f = {}".format(phase.eos._f(t, vm, ni, xai)))
    # print("f_v = {}".format(phase.eos._f_v(t, vm, ni, xai)))
    # print("f_v_num = {}".format(phase.eos._f_v_num(t, vm, ni, xai)))
    # print("f_i = {}".format(phase.eos._f_i(t, vm, ni, xai)))
    # print("f_i_num = {}".format(phase.eos._f_i_num_vect(t, vm, ni, xai)))
    # print("ln_phi_j = {}".format(phase.eos.ln_phi_j(t, vm, ni, xai)))
    #
    # # Examples illustrating how the Corel class functions.
    # print("---------------")
    # print("Water vapor pressure correlation defined: {}".format(water.pvap.defined()))
    # print("Water vapor pressure at triple point: {}".format(water.pvap(water.pvap.t_min)))
    # print("Water vapor pressure at boiling point: {}".format(water.pvap(373.15)))
    # print("Water vapor pressure at critical point: {}".format(water.pvap(water.pvap.t_max)))
    #
    # print("Water ideal gas heat capacity correlation defined: {}".format(water.cp_ig.defined()))
    # print("Water ideal gas heat capacity at triple point: {}".format(water.cp_ig(water.cp_ig.t_min)))
    # print("Water ideal gas heat capacity at critical point: {}".format(water.cp_ig(water.cp_ig.t_max)))
    #
    # print("Water heat of vaporization correlation defined: {}".format(water.hvap.defined()))
    # print("Water heat of vaporization at triple point: {}".format(water.hvap(water.hvap.t_min)))
    # print("Water heat of vaporization at critical point: {}".format(water.hvap(water.hvap.t_max)))
    print("holy fuck we made it")

    module = comps
    for key, value in module.__dict__.items():
        if isinstance(value, comps.Comp):
            print(value)
