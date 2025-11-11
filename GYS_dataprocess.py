from functools import partial
from typing import Callable, Dict
import numpy as np
import mylib

# 사용법
# data = process_data('Pres2D', dirname, t1, spn, mylib)


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
    psi = mylib.read_data(dirname, 'psi')
    B0 = mylib.read_data(dirname, 'B')[:, :] 
    Btheta = mylib.read_data(dirname, 'Btheta')[:, :]
    phi0, tdiag = mylib.read_data(dirname, 'Phi00', 'time_diag', t1=t1)
    vExB = np.gradient(phi0[:], psi)
    return vExB
def process_wExB(dirname: str, t1: int, spnum: int,mylib):
    psi = mylib.read_data(dirname, 'psi')
    B0 = mylib.read_data(dirname, 'B')[:, :] 
    Btheta = mylib.read_data(dirname, 'Btheta')[:, :]
    phi0, tdiag = mylib.read_data(dirname, 'Phi00', 'time_diag', t1=t1)
    vExB = np.gradient(phi0[:], psi)
    wExB = np.squeeze(np.gradient(vExB, psi)*(Btheta)**2/B0)
    return -(wExB)


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
    'vExB': process_vExB
}

def process_data(dtype: str, dirname: str, t1: int, spnum: int, mylib):
    if dtype in DATA_PROCESSORS:
        return DATA_PROCESSORS[dtype](dirname, t1, spnum, mylib)
    else:
        return mylib.read_data(dirname, dtype, t1=t1, spnum=spnum)
