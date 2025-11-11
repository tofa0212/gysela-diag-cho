import h5py
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mylib

def main(dirname, sp):
    Te0, Ts0, ne0, S0, Smod = mylib.read_data(
        dirname, 'Te0', 'Ts0', 'ns0', 'S_rshape', 'S_rshape_mod', spnum=sp) #)
    q, shear, psi = mylib.read_data(dirname, 'safety_factor', 'shear', 'psi')

    R0, rhostar = mylib.read_data(dirname, 'R0', 'rhostar')
    rg = mylib.read_data(dirname, 'rg')
    
    rg *= rhostar
    R0 *= rhostar
    xind = np.where((rg > 0.) & (rg<2))
    
    P, neq, tdiag = mylib.read_data(dirname,
                                        'stress_FSavg', 'dens_FSavg', 'time_diag', t1 = 1501)
    
    Ti = (P/neq)
    RLT = -np.gradient(Ti, rg)/Ti*R0
    Smod1D = np.squeeze(Smod[0, xind])

    fig1, ax1 = mylib.init_plot_params()
    ax1.plot(rg[xind], RLT[xind], label=r'$R/L_T$', color='r') # /Ts0*R0
    ax1.plot(rg[xind], R0*(-np.gradient(ne0, rg)/ne0)[xind], label=r'$R/L_n$', color='b') # /ne0*R0
    # ax1.plot(rg[xind], -R0*(np.gradient(Ts0, rg)/Ts0)[xind], label=r'$R/L_T$', color='r') # /Ts0*R0
    # ax1.plot(rg[xind], R0*(-np.gradient(ne0, rg)/ne0)[xind], label=r'$R/L_n$', color='b') # /ne0*R0
    # ax1.plot(rg[xind], Smod1D/np.max(Smod1D), label=r'$S_{mod}$', color='b')
    ax1.set_xlabel(r'r/a', size=28)
    ax1.set_xlim(0, 1.15)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=20)

    # psi_int = np.zeros(np.shape(rg))
    # for i in range(np.size(rg)):
    #     psi_int[i] = np.trapz((rg/q)[0:i], rg[0:i])
    fig2, ax2 = mylib.init_plot_params()
    ax2.plot(rg[xind], q[xind], color='b')
    # ax2.plot(rg[xind], psi[xind], color='r')
    ax2.set_xlabel(r'r/a', size=20)
    ax2.set_ylabel('Safety Factor', size=20)
    ax2.set_xlim(0, 1)
    x0 = np.interp(1.75, q, rg)
    
    print(np.shape(Smod))

    fig3, ax3 = mylib.init_plot_params()
    ax3.plot(rg[xind], S0[xind], color='r', label=r'$S_{static}$')
    ax3.plot(rg[xind], np.squeeze(np.average(Smod[:, xind], axis=0)), color='b', label=r'$S_{mod}$')
    ax3.set_xlabel(r'r/a', size=20)
    ax3.set_ylabel('Source', size=20)
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax3.set_xlim(0, 1)
    fig3.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=14)
    ax3.yaxis.get_offset_text().set_size(14)


    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the init in given directory")
    parser.add_argument("dirname", type=str, help='Name of directory')
    parser.add_argument('-sp', type=str, help='Species number', default=0)
    args = parser.parse_args()

    abs_path = os.path.abspath(args.dirname)

    main(abs_path, args.sp)
