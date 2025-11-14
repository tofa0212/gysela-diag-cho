from functools import partial, lru_cache
from typing import Callable, Dict
import numpy as np
import mylib


# How to use? (example)
# data = process_data('Pres2D', dirname, t1, spn, mylib)

class _GlobalCache:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_GlobalCache, cls).__new__(cls)
            cls._instance.cache = {}
        return cls._instance
    
    def get_static(self, dirname:str, key: str, mylib):
        cache_key = f"{dirname}_{key}" # (dirname, key)
        if cache_key not in self.cache:
            self.cache[cache_key] = mylib.read_data(dirname, key)
        return self.cache[cache_key]
    
    def clear(self):
        self.cache.clear()
        
_cache = _GlobalCache()

def _read_static(dirname: str, key: str, mylib):
    return _cache.get_static(dirname, key, mylib)


def process_pres2d(dirname: str, t1: int, spnum: int, mylib):
    Ppar, Pperp = mylib.read_data(dirname, 'PparGC_rtheta', 'PperpGC_rtheta', t1=t1, spnum=spnum)
    return Ppar + Pperp/2

def process_pres_aniso(dirname: str, t1: int, spnum: int, mylib):
    Ppar, Pperp = mylib.read_data(dirname, 'PparGC_rtheta', 'PperpGC_rtheta', t1=t1, spnum=spnum)
    return Ppar / (Pperp)

def process_qgc_ve(dirname: str, t1: int, spnum: int, mylib):
    Ppar, Pperp = mylib.read_data(dirname, 'QGC_perp_vE_rtheta', 'QGC_par_vE_rtheta', t1=t1, spnum=spnum)
    return Ppar + Pperp

def process_QGC_vD(dirname: str, t1: int, spnum: int, mylib):
    Ppar, Pperp = mylib.read_data(dirname, 'QGC_perp_vD_rtheta', 'QGC_par_vD_rtheta', t1=t1, spnum=spnum)
    return Ppar + Pperp

def process_vDr(dirname: str, t1: int, spnum: int, mylib):
    n, vD = mylib.read_data(dirname, 'densGC_rtheta', 'nvrGC_vD_rtheta', t1=t1, spnum=spnum)
    Ppar, Pperp = mylib.read_data(dirname, 'PparGC_rtheta', 'PperpGC_rtheta', t1=t1, spnum=spnum)
    return vD / (n)

def process_vEr(dirname: str, t1: int, spnum: int, mylib):
    n, vE = mylib.read_data(dirname, 'densGC_rtheta', 'nvrGC_vE_rtheta', t1=t1, spnum=spnum)
    Ppar, Pperp = mylib.read_data(dirname, 'PparGC_rtheta', 'PperpGC_rtheta', t1=t1, spnum=spnum)
    return vE / (n)

def process_frtheta(dirname: str, t1: int, spnum: int, mylib):      
    tmp1, tmp2 = mylib.read_data(dirname, 'frtheta_passing', 'frtheta_trapped', t1=t1, spnum=spnum)
    return tmp1 + tmp2

def process_frvpar(dirname: str, t1: int, spnum: int, mylib):
    tmp1, tmp2 = mylib.read_data(dirname, 'frvpar_passing', 'frvpar_trapped', t1=t1, spnum=spnum)
    return tmp1 + tmp2

def process_fphivpar(dirname: str, t1: int, spnum: int, mylib):
    tmp1, tmp2 = mylib.read_data(dirname, 'fphivpar_passing', 'fphivpar_trapped', t1=t1, spnum=spnum)
    return tmp1 + tmp2

def process_fthvpar(dirname: str, t1: int, spnum: int, mylib):
    tmp1, tmp2 = mylib.read_data(dirname, 'fthvpar_passing', 'fthvpar_trapped', t1=t1, spnum=spnum)
    return tmp1 + tmp2

def process_test_vE(dirname: str, t1: int, spnum: int, mylib):
    tmp1, tmp2 = mylib.read_data(dirname, 'nIturbGC_rtheta', 'spreadingGC_rtheta', t1=t1, spnum=spnum)
    return tmp2 / tmp1 

def process_test_vE2(dirname: str, t1: int, spnum: int, mylib):
    tmp1, tmp2 = mylib.read_data(dirname, 'densGC_rtheta', 'nvrGC_vE_rtheta', t1=t1, spnum=spnum)
    return tmp2 / tmp1

def process_test_vE3(dirname: str, t1: int, spnum: int, mylib):
    tmp1, tmp2 = mylib.read_data(dirname, 'nIturbGC_rtheta', 'spreadingGC_rtheta', t1=t1, spnum=spnum)
    dataa = tmp2 / tmp1
    tmp1, tmp2 = mylib.read_data(dirname, 'densGC_rtheta', 'nvrGC_vE_rtheta', t1=t1, spnum=spnum)   
    datab = tmp2 / tmp1
    return datab - dataa

def process_spreading_speed(dirname: str, t1: int, spnum: int, mylib):
    Iturb, spreading = mylib.read_data(dirname, 'IturbGC_rtheta', 'spreadingGC_rtheta', t1=t1)
    return spreading / (Iturb + 1e-10*spreading) #np.min(np.abs(spreading)))

def process_vExB(dirname: str, t1: int, spnum: int,mylib):
    # psi = mylib.read_data(dirname, 'psi')
    # rg = mylib.read_data(dirname, 'rg')
    # B0 = mylib.read_data(dirname, 'B')[:, :] 
    # Btheta = mylib.read_data(dirname, 'Btheta')[:, :]
    
    rg = _read_static(dirname, 'rg', mylib)
    phi0 = mylib.read_data(dirname, 'Phi00', t1=t1)
    vExB = np.gradient(phi0, rg)
    return vExB

def process_wExB(dirname: str, t1: int, spnum: int,mylib):
    # psi = mylib.read_data(dirname, 'psi')
    # B0 = mylib.read_data(dirname, 'B')[:, :] 
    # Btheta = mylib.read_data(dirname, 'Btheta')[:, :]
    psi = _read_static(dirname, 'psi', mylib)
    B0 = _read_static(dirname, 'B', mylib)
    Btheta = _read_static(dirname, 'Btheta', mylib)
    
    phi0 = mylib.read_data(dirname, 'Phi00', t1=t1)
    vExB = np.gradient(phi0[:], psi)
    wExB = np.squeeze(np.gradient(vExB, psi)*(Btheta)**2/B0)
    return -(wExB)

def process_RS_elec(dirname: str, t1: int, spnum: int, mylib):
    # rg = mylib.read_data(dirname, 'rg')
    # thg = mylib.read_data(dirname, 'thetag')
    # B = mylib.read_data(dirname, 'B')
    # jacob_space = mylib.read_data(dirname, 'jacob_space')
    
    rg = _read_static(dirname, 'rg', mylib)
    thg = _read_static(dirname, 'thetag', mylib)
    B = _read_static(dirname, 'B', mylib)
    jacob_space = _read_static(dirname, 'jacob_space', mylib)
    Phi3D = mylib.read_data(dirname, 'Phi_3D', t1=t1-1501, spnum=spnum)
    dr_Phi3D = np.gradient(Phi3D, rg, axis=2)
    dth_Phi3D = np.gradient(Phi3D, thg, axis=1)
    B_jacob = (B * jacob_space)
    dr_vExB = -dth_Phi3D / (B_jacob[np.newaxis, :, :])
    dth_vExB = dr_Phi3D / (B_jacob[np.newaxis, :, :])
    return dr_vExB * dth_vExB

def process_RS_mix(dirname: str, t1: int, spnum: int, mylib):
    # rg = mylib.read_data(dirname, 'rg')
    # thg = mylib.read_data(dirname, 'thetag')
    # B = mylib.read_data(dirname, 'B')
    # jacob_space = mylib.read_data(dirname, 'jacob_space')
    rg = _read_static(dirname, 'rg', mylib)
    thg = _read_static(dirname, 'thetag', mylib)
    B = _read_static(dirname, 'B', mylib)
    jacob_space = _read_static(dirname, 'jacob_space', mylib)
    Phi3D = mylib.read_data(dirname, 'Phi_3D', t1=t1-1501, spnum=spnum)
    Pperp_3D = mylib.read_data(dirname, 'Pperp_GC_3D', t1=t1-1501, spnum=spnum)
    dens_3D = mylib.read_data(dirname, 'n_GC_3D', t1=t1-1501, spnum=spnum)
    dr_Phi3D = np.gradient(Phi3D, rg, axis=2)
    dth_Pperp3D = np.gradient(Pperp_3D, thg, axis=1)
    B_jacob = (B * jacob_space)
    dth_vExB = dr_Phi3D / ((B_jacob)[np.newaxis, :, :])
    dr_vdiag_3d = -dth_Pperp3D / ((B_jacob)[np.newaxis, :, :]*dens_3D)
    return dr_vdiag_3d * dth_vExB

def process_RS_tot(dirname: str, t1: int, spnum: int, mylib):
    rg = _read_static(dirname, 'rg', mylib)
    thg = _read_static(dirname, 'thetag', mylib)
    B = _read_static(dirname, 'B', mylib)
    jacob_space = _read_static(dirname, 'jacob_space', mylib)
    Phi3D = mylib.read_data(dirname, 'Phi_3D', t1=t1-1501, spnum=spnum)
    Pperp_3D = mylib.read_data(dirname, 'Pperp_GC_3D', t1=t1-1501, spnum=spnum)
    dens_3D = mylib.read_data(dirname, 'n_GC_3D', t1=t1-1501, spnum=spnum)
    B_jacob = (B * jacob_space)
    dr_Phi3D = np.gradient(Phi3D, rg, axis=2)
    dth_Phi3D = np.gradient(Phi3D, thg, axis=1)
    dth_Pperp3D = np.gradient(Pperp_3D, thg, axis=1)
    dr_vExB = -dth_Phi3D / (B_jacob[np.newaxis, :, :])
    dth_vExB = dr_Phi3D / (B_jacob[np.newaxis, :, :])
    dr_vdiag_3d = -dth_Pperp3D / ((B_jacob)[np.newaxis, :, :]*dens_3D)
    
    return (dr_vdiag_3d +dr_vExB) * dth_vExB
    
def process_vpol_vE(dirname: str, t1: int, spnum: int, mylib):
    nvpol_vE_rtheta = mylib.read_data(dirname, 'nvpolGC_vE_rtheta', t1=t1, spnum=spnum)
    dens_rtheta = mylib.read_data(dirname, 'densGC_rtheta', t1=t1, spnum=spnum)
    
    return nvpol_vE_rtheta / dens_rtheta

def process_vpol(dirname: str, t1: int, spnum: int, mylib):
    dens_rtheta = mylib.read_data(dirname, 'densGC_rtheta', t1=t1, spnum=spnum)
    nvpol_vE = mylib.read_data(dirname, 'nvpolGC_vE_rtheta', t1=t1, spnum=spnum)
    nvpol_vD = mylib.read_data(dirname, 'nvpolGC_vD_rtheta', t1=t1, spnum=spnum)
    nvpol_mag = mylib.read_data(dirname, 'nvpol_mag_rtheta', t1=t1, spnum=spnum)
    nvpol_vpar = mylib.read_data(dirname, 'nvpolGC_vpar_rtheta', t1=t1, spnum=spnum)
    return (nvpol_vE + nvpol_vD + nvpol_mag + nvpol_vpar) / dens_rtheta


# 프로세서 함수들을 딕셔너리로 관리
DATA_PROCESSORS: Dict[str, Callable] = {
    'Pres2D': process_pres2d,
    'Pres_aniso': process_pres_aniso,
    'QGC_vE': process_qgc_ve,
    'QGC_vD': process_QGC_vD,
    'vDr': process_vDr,
    'vEr': process_vEr,
    'frtheta': process_frtheta,
    'frvpar': process_frvpar,
    'fphivpar': process_fphivpar,
    'fthvpar': process_fthvpar,
    'test_vE': process_test_vE,
    'test_vE2': process_test_vE2,
    'test_vE3': process_test_vE3,
    'spreading_speed': process_spreading_speed,
    'wExB': process_wExB,
    'vExB': process_vExB,
    'RS_elec': process_RS_elec,
    'RS_mix': process_RS_mix,
    'RS_tot': process_RS_tot,
    'vpol_vE': process_vpol_vE,
    'vpol': process_vpol,
}

def process_data(dtype: str, dirname: str, t1: int, spnum: int, mylib):
    if dtype in DATA_PROCESSORS:
        return DATA_PROCESSORS[dtype](dirname, t1, spnum, mylib)
    else:
        return mylib.read_data(dirname, dtype, t1=t1, spnum=spnum)

def clear_cache():
    """Optional: Clear cache if needed for memory management"""
    _cache.clear()