'''
dict_filename: directory for output file name
build_file_path: Build the file path for output file
read_hdf5_data: Basic function to read hdf5 file
'''

def generate_filename(keys: str, prefix: str, suffix: str):
    return {key: (prefix,suffix) for key in keys}

init_keys = ['R0','S_rshape', 'S_rshape_mod', 'Sce0', 'Sce_mod',
             'Te0', 'Ts0', 'coefDr', 'coefDth', 'coefNu',
             'ne0', 'ns0', 'nseq', 'mask_CORE', 'mask_SOL',
             'rhostar', 'nustar', 'Antenna_PhiG_mode_0',
             'Antenna_PhiQ_mode_0']

magnet_keys = ['B', 'B_gradphi','B_gradr', 'B_gradtheta',
               'Bphi', 'Br', 'Btheta',
               'R', 'Z', 'dBdr', 'dBdtheta',
               'g11', 'g12', 'g13', 'g21', 'g22', 'g23',
               'g31', 'g32', 'g33', 'Btheta',
               'gradx1gradx1','gradx1gradx2','gradx1gradx3',
               'gradx2gradx1','gradx2gradx2','gradx2gradx3',
               'gradx3gradx1','gradx3gradx2','gradx3gradx3',
               'intdtheta_Js','intdthetadphi_Js','jacob_space',
               'mu0J_gradphi','mu0j_gradr','mu0J_gradtheta',
               'mu0J_phi','mu0J_r','mu0J_theta',
               'psi', 'safety_factor', 'shear']

mesh_keys = ['Nr', 'Nmu', 'Nphi', 'Ntheta', 'Nvpar',
             'mug', 'phig', 'rg', 'thetag', 'vparg']

phi2d_keys = ['Phirphi', 'Phirth','Phirth_n0','Phithphi', 'time_diag']

Apar2d_keys = ['Aparrphi', 'Aparrth', 'Aparrth_n0', 'Aparthphi']

rprof_part_keys = ['Enpot_FSavg', 'Gamma_vD_r_FSavg', 'Gamma_vE_r_FSavg',
                   'Phi00', 'Phi_FSavg', 'Q_vD_r_FSavg', 'Q_vE_r_FSavg',
                   'RStheta_vD_r_FSavg', 'RStheta_vE_r_FSavg',
                   'dens_FSavg', 'dens_rtheta', 'dens_trapped_FSavg',
                   'nVpar_FSavg', 'nvpol_mag_B_FSavg', 'nvpol_mag_FSavg',
                   'nvpol_mag_rtheta', 'stress_FSavg', 'stress_rtheta' ]

rprof_GC2D_keys = ['IturbGC_rtheta', 'NTV_rtheta', 'PparGC_rtheta', 'PperpGC_rtheta',
                   'QGC_par_vD_rtheta', 'QGC_par_vE_rtheta', 'QGC_par_vEn0_rtheta',
                   'QGC_perp_vD_rtheta', 'QGC_perp_vE_rtheta', 'QGC_perp_vEn0_rtheta',
                   'RSphiGC_vE_rtheta', 'RSpolGC_vE_rtheta', 'VparGC_rtheta',
                   'densGC_rtheta', 'nIturbGC_rtheta', 'nvpolGC_vD_rtheta',
                   'nvpolGC_vE_rtheta', 'nvpolGC_vEn0_rtheta', 'nvpolGC_vpar_rtheta',
                   'nvrGC_vD_rtheta', 'nvrGC_vE_rtheta', 'spreadingGC_rtheta']

sources_keys = ['E_par_Sce', 'E_par_pass_Sce', 'E_par_trapp_Sce',
                'E_perp_Sce', 'E_perp_pass_Sce', 'E_perp_trapp_Sce',
                'Energy_Sce', 'Mass_Sce', 'Mass_pass_Sce',
                'Mass_trapp_Sce', 'Moment_Sce', 'Vorticity_Sce']

f2D_keys = ['fFSavg_irS_vparmu', 'fFSavg_irp_m15_vparmu',
            'fFSavg_irp_m30_vparmu', 'fFSavg_irp_m40_vparmu',
            'fFSavg_irp_p15_vparmu', 'fFSavg_irp_p30_vparmu',
            'fFSavg_irp_p40_vparmu', 'fFSavg_irp_vparmu',
            'fphivpar_passing',  'fphivpar_trapped',
            'frtheta_passing', 'frtheta_trapped',
            'frvpar_passing', 'frvpar_trapped',
            'fthvpar_passing', 'fthvpar_trapped',
            'fvparmu']


phi3d_keys = ['Phi_3D']
Apar3d_keys = ['Apar_3D']
Ppar3d_keys = ['Ppar_GC_3D']
Pperp3d_keys = ['Pperp_GC_3D']
Pperp_trap3d_keys = ['Perp_trap_GC_3D']
VGC3d_keys = ['V_GC_GC_3D']
dens3d_keys = ['n_GC_3D']
ntrap3d_keys = ['n_trap_GC_3D']

spread3d_keys = ['spreading_3D_3D']
Iturb3d_keys = ['Iturb_3D_3D']
nIturb3d_keys = ['nIturb_3D_3D']

prefix_suffix_map = {
    'init_keys': ('init_state', 'init_state_r000.h5'),
    'magnet_keys': ('init_state', 'magnet_config_r000.h5'),
    'mesh_keys': ('init_state', 'mesh5d_r000.h5'),
    'phi2d_keys': ('Phi2D', 'Phi2D_d{t1:05d}.h5'),
    'Apar2d_keys': ('Phi2D', 'Apar2D_d{t1:05d}.h5'),
    'rprof_part_keys': ('rprof', 'rprof_part_d{t1:05d}.h5'),
    'rprof_GC2D_keys': ('rprof', 'rprof_GC2D_d{t1:05d}.h5'),
    'f2D_keys' : ('f2D', 'f2D_d{t1:05d}.h5'),
    'sources_keys' : ('sources', 'sources_r000.h5'),
    'phi3d_keys'   : ('Phi3D', 'Phi_3D_d{t1:05d}.h5'),
    'Apar3d_keys'   : ('Phi3D', 'Apar_3D_d{t1:05d}.h5'),
    'Ppar3d_keys'   : ('moment3D', 'Ppar_GC_3D_d{t1:05d}.h5'),
    'Pperp3d_keys'   : ('moment3D', 'Pperp_GC_3D_d{t1:05d}.h5'),
    'Pperp_trap3d_keys'   : ('moment3D', 'Pperp_trap_GC_3D_d{t1:05d}.h5'),
    'VGC3d_keys'   : ('moment3D', 'V_GC_3D_d{t1:05d}.h5'),
    'dens3d_keys'   : ('moment3D', 'n_GC_3D_d{t1:05d}.h5'),
    'ntrap3d_keys'   : ('moment3D', 'n_trap_GC_3D_d{t1:05d}.h5'),
    'spread3d_keys'  : ('yw_flux', 'spreading_3D_3D_d{t1:05d}.h5'),
    'Iturb3d_keys'  : ('yw_flux', 'Iturb_3D_3D_d{t1:05d}.h5'),
    'nIturb3d_keys'  : ('yw_flux', 'nIturb_3D_3D_d{t1:05d}.h5')
}


dict_filename = {}

for key_group_name, (prefix, suffix) in prefix_suffix_map.items():
    key_group = globals()[key_group_name]
    dict_filename.update(generate_filename(key_group, prefix, suffix))

