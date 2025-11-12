"""
Initial Profile Plotter

Plot initial equilibrium profiles from GYSELA simulation data:
- Temperature and density gradient scale lengths (R/L_T, R/L_n)
- Safety factor profile
- Source profiles (static and modulated)
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mylib


def plot_gradient_profiles(dirname: str, spnum: int = 0,
                           timestep: int = 1501,
                           r_max: float = 1.15,
                           font_size: int = 18):
    """
    Plot temperature and density gradient scale length profiles.

    Args:
        dirname: Data directory path
        spnum: Species number (0=ions, 1=electrons)
        timestep: Timestep for temperature calculation
        r_max: Maximum radius for plotting
        font_size: Base font size

    Returns:
        (fig, ax): Figure and axes objects
    """
    # Load normalized grid using mylib
    grid_data = mylib.load_normalized_grid(dirname, r_min=0.0, r_max=2.0,
                                           spnum=spnum, return_mask=True)
    rg = grid_data['rg_full']
    R0 = grid_data['R0']
    xind = grid_data['mask']

    # Load initial density profile
    ne0 = mylib.read_data(dirname, 'ns0', spnum=spnum)

    # Calculate temperature and R/L_T using mylib functions
    Ti = mylib.load_temperature_profile(dirname, timestep=timestep, spnum=spnum)
    RLT = mylib.calc_gradient_scale_length(Ti, rg, R0)

    # Calculate R/L_n
    RLn = mylib.calc_gradient_scale_length(ne0, rg, R0)

    # Create plot
    fig, ax = mylib.init_plot_params(figsize=(8, 6), font_size=font_size)

    ax.plot(rg[xind], RLT[xind], label=r'$R/L_T$', color='r', linewidth=2)
    ax.plot(rg[xind], RLn[xind], label=r'$R/L_n$', color='b', linewidth=2)

    ax.set_xlabel(r'r/a', size=font_size + 10)
    ax.set_xlim(0, r_max)
    ax.set_ylabel(r'Gradient Scale Length', size=font_size + 10)
    ax.tick_params(labelsize=font_size)
    ax.legend(fontsize=font_size + 2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, ax


def plot_safety_factor(dirname: str, r_max: float = 1.0,
                       font_size: int = 18):
    """
    Plot safety factor profile.

    Args:
        dirname: Data directory path
        r_max: Maximum radius for plotting
        font_size: Base font size

    Returns:
        (fig, ax): Figure and axes objects
    """
    # Load safety factor and radial grid
    q = mylib.read_data(dirname, 'safety_factor')

    grid_data = mylib.load_normalized_grid(dirname, r_min=0.0, r_max=2.0,
                                           return_mask=True)
    rg = grid_data['rg_full']
    xind = grid_data['mask']

    # Create plot
    fig, ax = mylib.init_plot_params(figsize=(8, 6), font_size=font_size)

    ax.plot(rg[xind], q[xind], color='b', linewidth=2)
    ax.set_xlabel(r'r/a', size=font_size + 2)
    ax.set_ylabel('Safety Factor (q)', size=font_size + 2)
    ax.set_xlim(0, r_max)
    ax.tick_params(labelsize=font_size)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Find radius where q = 1.75 (useful reference)
    try:
        r_q175 = np.interp(1.75, q, rg)
        print(f"Radius at q=1.75: r/a = {r_q175:.3f}")
    except:
        pass

    return fig, ax


def plot_source_profiles(dirname: str, spnum: int = 0,
                        r_max: float = 1.0,
                        font_size: int = 14):
    """
    Plot static and modulated source profiles.

    Args:
        dirname: Data directory path
        spnum: Species number (0=ions, 1=electrons)
        r_max: Maximum radius for plotting
        font_size: Base font size

    Returns:
        (fig, ax): Figure and axes objects

    Note:
        S_rshape_mod is optional - only plotted if available in data
    """
    # Load static source
    S_static = mylib.read_data(dirname, 'S_rshape', spnum=spnum)

    # Try to load modulated source (may not exist in all branches)
    has_modulated = False
    try:
        S_mod = mylib.read_data(dirname, 'S_rshape_mod', spnum=spnum)
        has_modulated = True
        print("✓ S_rshape_mod found - plotting modulated source")
    except (FileNotFoundError, KeyError) as e:
        print("⚠ S_rshape_mod not found - plotting static source only")
        print(f"  (This is normal for non-modulated simulations)")

    # Load grid
    grid_data = mylib.load_normalized_grid(dirname, r_min=0.0, r_max=2.0,
                                           spnum=spnum, return_mask=True)
    rg = grid_data['rg_full']
    xind = grid_data['mask']

    # Create plot
    fig, ax = mylib.init_plot_params(figsize=(8, 6), font_size=font_size)

    # Plot static source
    ax.plot(rg[xind], S_static[xind], color='r', linewidth=2,
            label=r'$S_{\mathrm{static}}$')

    # Plot modulated source if available
    if has_modulated:
        # Average over time dimension if 2D
        if S_mod.ndim == 2:
            S_mod_avg = np.average(S_mod[:, xind], axis=0)
        else:
            S_mod_avg = S_mod[xind]

        ax.plot(rg[xind], S_mod_avg, color='b', linewidth=2,
                label=r'$S_{\mathrm{mod}}$')
        print(f"S_mod shape: {S_mod.shape}")

    ax.set_xlabel(r'r/a', size=font_size + 6)
    ax.set_ylabel('Source', size=font_size + 6)
    ax.set_xlim(0, r_max)

    # Use scientific notation for y-axis
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_size(font_size)

    ax.tick_params(labelsize=font_size)
    ax.legend(fontsize=font_size)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, ax


def main(dirname: str, spnum: int = 0,
         timestep: int = 1501,
         show_plots: bool = True,
         save_dir: str = None):
    """
    Generate all initial profile plots.

    Args:
        dirname: Data directory path
        spnum: Species number (0=ions, 1=electrons)
        timestep: Timestep for temperature calculation
        show_plots: Whether to display plots interactively
        save_dir: Directory to save figures (None = don't save)
    """
    print(f"\n{'='*60}")
    print(f"Plotting Initial Profiles")
    print(f"Directory: {dirname}")
    print(f"Species: {spnum} ({'ions' if spnum == 0 else 'electrons'})")
    print(f"{'='*60}\n")

    # Create output directory if saving
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving figures to: {save_dir}")

    # Plot gradient scale lengths
    print("1. Plotting gradient scale lengths (R/L_T, R/L_n)...")
    fig1, ax1 = plot_gradient_profiles(dirname, spnum=spnum, timestep=timestep)
    if save_dir:
        fig1.savefig(os.path.join(save_dir, 'init_gradients.png'),
                     dpi=300, bbox_inches='tight')

    # Plot safety factor
    print("2. Plotting safety factor profile...")
    fig2, ax2 = plot_safety_factor(dirname)
    if save_dir:
        fig2.savefig(os.path.join(save_dir, 'init_safety_factor.png'),
                     dpi=300, bbox_inches='tight')

    # Plot source profiles
    print("3. Plotting source profiles...")
    fig3, ax3 = plot_source_profiles(dirname, spnum=spnum)
    if save_dir:
        fig3.savefig(os.path.join(save_dir, 'init_sources.png'),
                     dpi=300, bbox_inches='tight')

    print("\n✓ All plots generated successfully")

    if show_plots:
        plt.show()

    return fig1, fig2, fig3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot initial equilibrium profiles from GYSELA simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('dirname', type=str,
                       help='Data directory path')
    parser.add_argument('-s', '--spnum', type=int, default=0,
                       help='Species number (0=ions, 1=electrons)')
    parser.add_argument('-t', '--timestep', type=int, default=1501,
                       help='Timestep for temperature calculation')
    parser.add_argument('--save', type=str, default=None,
                       help='Directory to save figures')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (useful with --save)')

    args = parser.parse_args()

    abs_path = os.path.abspath(args.dirname)

    main(abs_path,
         spnum=args.spnum,
         timestep=args.timestep,
         show_plots=not args.no_show,
         save_dir=args.save)
