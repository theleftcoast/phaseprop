"""sPC-SAFT parameters from Marshall et al. (2019-2021)."""

import comps
import refs
import spc_saft
import assoc


# Original Gross and Sadowski PC-SAFT parameters for most non-polar, non-associating compounds.
comps.methane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=1.0,
                                                                              seg_diam=3.7039,
                                                                              disp_energy=150.03,
                                                                              source=refs.gross_sadowski)

comps.ethane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=1.6069,
                                                                             seg_diam=3.5206,
                                                                             disp_energy=191.42,
                                                                             source=refs.gross_sadowski)

comps.propane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=2.002,
                                                                              seg_diam=3.6184,
                                                                              disp_energy=208.11,
                                                                              source=refs.gross_sadowski)

comps.n_butane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=2.3316,
                                                                               seg_diam=3.7086,
                                                                               disp_energy=222.88,
                                                                               source=refs.gross_sadowski)

comps.i_butane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=2.2616,
                                                                               seg_diam=3.7574,
                                                                               disp_energy=216.53,
                                                                               source=refs.gross_sadowski)

comps.n_pentane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=2.6896,
                                                                                seg_diam=3.7729,
                                                                                disp_energy=231.2,
                                                                                source=refs.gross_sadowski)

comps.i_pentane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=2.562,
                                                                                seg_diam=3.8296,
                                                                                disp_energy=230.75,
                                                                                source=refs.gross_sadowski)

comps.n_hexane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=3.0576,
                                                                               seg_diam=3.7983,
                                                                               disp_energy=236.77,
                                                                               source=refs.gross_sadowski)

comps.c_hexane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=2.5303,
                                                                               seg_diam=3.8499,
                                                                               disp_energy=278.11,
                                                                               source=refs.gross_sadowski)

comps.heptane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=3.4831,
                                                                              seg_diam=3.8049,
                                                                              disp_energy=238.4,
                                                                              source=refs.gross_sadowski)

comps.mc_hexane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=2.6637,
                                                                                seg_diam=3.9993,
                                                                                disp_energy=282.33,
                                                                                source=refs.gross_sadowski)

comps.octane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=3.8176,
                                                                             seg_diam=3.8373,
                                                                             disp_energy=242.78,
                                                                             source=refs.gross_sadowski)

comps.nonane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=4.2079,
                                                                             seg_diam=3.8448,
                                                                             disp_energy=244.51,
                                                                             source=refs.gross_sadowski)

comps.decane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=4.6627,
                                                                             seg_diam=3.8384,
                                                                             disp_energy=243.87,
                                                                             source=refs.gross_sadowski)

comps.dodecane.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=5.3060,
                                                                               seg_diam=3.8959,
                                                                               disp_energy=249.21,
                                                                               source=refs.gross_sadowski)

comps.nitrogen.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=1.2053,
                                                                               seg_diam=3.3130,
                                                                               disp_energy=90.96,
                                                                               source=refs.gross_sadowski)

comps.oxygen.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=1.1217,
                                                                             seg_diam=3.2098,
                                                                             disp_energy=114.96,
                                                                             source=refs.economou_2007)

comps.argon.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=0.9285,
                                                                               seg_diam=3.4784,
                                                                               disp_energy=122.23,
                                                                               source=refs.gross_sadowski)

# Marshall's mapping of pi-pi and unsaturated interactions onto dipolar free energy (ref marshall_2019 & marshall_2018).
aromatic_ed = assoc.AssocSite(desc='pi electrons',
                              type='electron donor')
benzene_polar_strength = 2.16
benzene_aromatic_carbons = 6
double_bond_polar_strength = 0.6
triple_bond_polar_strength = 1.5

comps.benzene.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=2.305,
                                                                        seg_diam=3.732,
                                                                        disp_energy=291.23,
                                                                        polar_strength=benzene_polar_strength *
                                                                                       benzene_aromatic_carbons /
                                                                                       benzene_aromatic_carbons,
                                                                        phantom_dipole=False,
                                                                        source=refs.marshall_2019)
benzene_aromatic_group = assoc.AssocGroup(desc='aromatic',
                                          scheme='variable:0')
benzene_aromatic_group.add_site(site=aromatic_ed,
                                number=benzene_aromatic_carbons / benzene_aromatic_carbons)
comps.benzene.spc_saft_parms.add_assoc_group(benzene_aromatic_group)

toluene_aromatic_carbons = 6
comps.toluene.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=2.612,
                                                                        seg_diam=3.814,
                                                                        disp_energy=293.33,
                                                                        polar_strength=benzene_polar_strength *
                                                                                       toluene_aromatic_carbons /
                                                                                       benzene_aromatic_carbons,
                                                                        phantom_dipole=False,
                                                                        source=refs.marshall_2019)
toluene_aromatic_group = assoc.AssocGroup(desc='aromatic',
                                          scheme='variable:0')
toluene_aromatic_group.add_site(site=aromatic_ed,
                                number=toluene_aromatic_carbons / benzene_aromatic_carbons)
comps.toluene.spc_saft_parms.add_assoc_group(benzene_aromatic_group)

# Marshall's water parameterization (ref marshall_2019a).
comps.water.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=1.179,
                                                                      seg_diam=2.9154,
                                                                      disp_energy=184.763,
                                                                      polar_strength=1.345,
                                                                      phantom_dipole=False,
                                                                      source=refs.marshall_2019a)
water_assoc_group = assoc.AssocGroup(desc='water',
                                     scheme='2:2')
water_h = assoc.AssocSite(desc='hydrogen atom',
                          type='electron acceptor')
water_o = assoc.AssocSite(desc='oxygen lone pair',
                          type='electron donor')
water_assoc_group.add_site(site=water_h,
                           number=2)
water_assoc_group.add_site(site=water_o,
                           number=2)
comps.water.spc_saft_parms.add_assoc_group(group=water_assoc_group)
water_h_o = assoc.AssocSiteInter(site_a=water_h,
                                 site_b=water_o,
                                 energy=1716.3,
                                 volume=0.0615,
                                 source=refs.marshall_2019a)
comps.water.spc_saft_parms.add_assoc_inter(water_h_o)

# Marshall's alcohol parameterizations (ref marshall_2020).
comps.methanol.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=2.358,
                                                                         seg_diam=2.809,
                                                                         disp_energy=185.297,
                                                                         source=refs.marshall_2020a)

methanol_assoc_group = assoc.AssocGroup(desc='methanol',
                                        scheme='2:1')
methanol_h = assoc.AssocSite(desc='hydrogen atom',
                             type='electron acceptor')
methanol_o = assoc.AssocSite(desc='oxygen lone pair',
                             type='electron donor')
methanol_assoc_group.add_site(site=methanol_h,
                              number=1)
methanol_assoc_group.add_site(site=methanol_o,
                              number=1)
comps.methanol.spc_saft_parms.add_assoc_group(group=methanol_assoc_group)
methanol_h_o = assoc.AssocSiteInter(site_a=methanol_h,
                                    site_b=methanol_o,
                                    energy=2226.94,
                                    volume=0.043,
                                    source=refs.marshall_2020)

# Marshall's ketone parameterizations (ref marshall_2021)
ketone_assoc_group = assoc.AssocGroup(desc='ketone',
                                      scheme='2:0')
ketone_o = assoc.AssocSite(desc='oxygen lone pair',
                           type='electron donor')
ketone_assoc_group.add_site(site=ketone_o,
                            number=2)
ketone_polar_strength = 4.898

comps.acetone.spc_saft_parms = spc_saft.sPCSAFTParms(seg_num=1.993,
                                                                        seg_diam=3.691,
                                                                        disp_energy=2599.99,
                                                                        polar_strength=ketone_polar_strength,
                                                                        phantom_dipole=False,
                                                                        source=refs.marshall_2021)
comps.acetone.spc_saft_parms.add_assoc_group(group=ketone_assoc_group)

