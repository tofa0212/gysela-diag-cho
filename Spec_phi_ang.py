import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import mylib
import matplotlib.ticker as ticker
from tqdm import tqdm
from scipy.signal import welch
from typing import Tuple, Optional
import warnings


def validate_angle_range(min_deg: float, max_deg: float) -> Tuple[float, float]:
    """Validate and normalize angle range"""
    min_rad = np.deg2rad(min_deg) % (2 * np.pi)
    max_rad = np.deg2rad(max_deg) % (2 * np.pi)
    return min_rad, max_rad


def get_angle_indices(angles_rad: np.ndarray, min_rad: float, max_rad: float) -> np.ndarray:
    """Get indices of angles within specified range"""
    angles_norm = angles_rad % (2 * np.pi)
    
    if min_rad <= max_rad:
        angle_mask = (angles_norm >= min_rad) & (angles_norm <= max_rad)
    else:  # Wrap around case (e.g., 350° to 10°)
        angle_mask = (angles_norm >= min_rad) | (angles_norm <= max_rad)
    
    angle_indices = np.where(angle_mask)[0]
    
    if len(angle_indices) == 0:
        warnings.warn(f"No angles found in range [{np.rad2deg(min_rad):.1f}°, {np.rad2deg(max_rad):.1f}°]")
    
    return angle_indices


def get_radial_indices(rg: np.ndarray, min_r: float, max_r: float) -> np.ndarray:
    """Get indices of radial points within specified range"""
    if min_r > max_r:
        raise ValueError(f"min_r ({min_r}) must be less than max_r ({max_r})")
    
    radial_mask = (rg >= min_r) & (rg <= max_r)
    radial_indices = np.where(radial_mask)[0]
    
    if len(radial_indices) == 0:
        warnings.warn(f"No radial points found in range [r/a={min_r:.3f}, {max_r:.3f}]")
        # Fallback: use middle point
        radial_indices = np.array([len(rg) // 2])
    
    return radial_indices


def process_phi_timeseries(dirname: str, n: int, dn: int, n1: int, 
                           xind: np.ndarray, angle_indices: np.ndarray,
                           bal_ang: np.ndarray, rg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process phi data over time steps
    
    Returns:
        Phi2D: [num_radial, num_time] array of phi fluctuations
        time_arr: [num_time] array of time values
    """
    Phi2D = np.zeros([len(rg), n])
    time_arr = np.zeros(n)
    
    print(f"Processing {n} time steps: {n1} to {n1 + (n-1)*dn} with increment {dn}")
    print(f"Using {len(angle_indices)} angle points and {len(rg)} radial points")
    
    for ni in tqdm(range(n), desc="Processing time steps", unit="step"):
        itime = n1 + ni * dn
        
        try:
            # Read phi data
            phirth_orig, phirth_n0, tdiag = mylib.read_data(
                dirname, 'Phirth', 'Phirth_n0', 'time_diag', t1=itime
            )
            
            # Remove n=0 mode and slice to selected radial range
            pdata = (phirth_orig - phirth_n0)[:, xind]
            
            # Average over selected angle range
            if len(angle_indices) > 0 and pdata.shape[0] > max(angle_indices):
                anal_data = pdata[angle_indices, :]
                phi_profile = np.mean(np.abs(anal_data), axis=0)  # Shape: [num_radial]
                
                if len(phi_profile) == len(rg):
                    Phi2D[:, ni] = phi_profile
                else:
                    warnings.warn(f"Dimension mismatch at t={itime}: expected {len(rg)}, got {len(phi_profile)}")
                    Phi2D[:, ni] = np.nan
            else:
                Phi2D[:, ni] = np.nan
            
            time_arr[ni] = tdiag[0]
            
        except Exception as e:
            warnings.warn(f"Error at time step {itime}: {e}")
            time_arr[ni] = np.nan
            Phi2D[:, ni] = np.nan
    
    return Phi2D, time_arr


def plot_contour(Phi2D: np.ndarray, time_arr: np.ndarray, rg: np.ndarray,
                min_angle_deg: float, max_angle_deg: float) -> Tuple[plt.Figure, plt.Axes]:
    """Create contour plot of phi fluctuations"""
    fig, ax = mylib.init_plot_params(font_size=16)
    
    X, Y = np.meshgrid(rg, time_arr)
    valid_indices = ~np.isnan(Phi2D)
    
    if np.any(valid_indices):
        valid_values = Phi2D[valid_indices]
        vmax = np.percentile(valid_values, 99) if len(valid_values) > 0 else 1.0
        
        contour = ax.pcolormesh(X, Y, Phi2D.T, cmap='hot',
                               vmin=0, vmax=vmax, shading='gouraud')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(r'$|\Phi|$ (angle-averaged)', size=16)
        cbar.ax.tick_params(labelsize=14)
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
    
    # Format axes
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_size(14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    ax.set_xlabel(r'$r/a$', size=20)
    ax.set_ylabel(r'Time [$\omega_c^{-1}$]', size=20)
    ax.set_title(f'Phi Fluctuation ({min_angle_deg:.0f}° to {max_angle_deg:.0f}°)', size=18)
    
    fig.tight_layout()
    return fig, ax


def compute_spectrum(Phi2D: np.ndarray, time_arr: np.ndarray, rg: np.ndarray,
                    radial_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute frequency spectrum for specified radial range
    
    Returns:
        frequencies: Frequency array
        psd: Power spectral density
        r_center: Center of radial range used
    """
    # Remove NaN time steps
    valid_time_idx = ~np.isnan(time_arr)
    
    if np.sum(valid_time_idx) < 2:
        raise ValueError("Need at least 2 valid time steps for spectrum analysis")
    
    time_valid = time_arr[valid_time_idx]
    Phi2D_valid = Phi2D[:, valid_time_idx]
    
    # Calculate time step and sampling frequency
    dt = np.mean(np.diff(time_valid))
    if dt <= 0:
        raise ValueError(f"Non-positive time step: dt={dt}")
    
    fs = 1.0 / dt
    print(f"\nSpectrum analysis parameters:")
    print(f"  Time step (dt): {dt:.2e} [ω_c^-1]")
    print(f"  Sampling frequency (fs): {fs:.2e} [ω_c]")
    
    # Average over selected radial range
    signal = np.mean(Phi2D_valid[radial_indices, :], axis=0)
    r_center = np.mean(rg[radial_indices])
    
    print(f"  Radial range: r/a ∈ [{rg[radial_indices[0]]:.3f}, {rg[radial_indices[-1]]:.3f}]")
    print(f"  Center: r/a = {r_center:.3f}")
    print(f"  Number of radial points averaged: {len(radial_indices)}")
    print(f"  Time series length: {len(signal)}")
    
    if len(signal) < 10:
        warnings.warn(f"Short time series (length {len(signal)}). Results may be unreliable.")
    
    # Compute PSD using Welch method
    nperseg = min(len(signal), 256 if len(signal) >= 256 else len(signal))
    frequencies, psd = welch(signal, fs=fs, nperseg=nperseg, scaling='density')
    
    return frequencies, psd, r_center


def analyze_spectrum_slope(frequencies: np.ndarray, psd: np.ndarray,
                           freq_min: float = 3e-4, freq_max: float = 2e-3) -> Optional[float]:
    """
    Analyze power law slope in specified frequency range
    
    Returns:
        slope: Power law exponent (or None if insufficient data)
    """
    print(f"\nSpectrum slope analysis:")
    print(f"  Frequency range: [{freq_min:.1e}, {freq_max:.1e}] [ω_c]")
    
    # Find indices in frequency range
    slope_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    slope_indices = np.where(slope_mask)[0]
    
    if len(slope_indices) < 2:
        warnings.warn("Not enough data points for slope fitting")
        return None
    
    freqs_fit = frequencies[slope_indices]
    psd_fit = psd[slope_indices]
    
    # Filter out non-positive values for log
    valid_mask = psd_fit > 0
    if np.sum(valid_mask) < 2:
        warnings.warn("Not enough positive PSD values for slope fitting")
        return None
    
    freqs_fit = freqs_fit[valid_mask]
    psd_fit = psd_fit[valid_mask]
    
    # Fit in log-log space
    log_freqs = np.log(freqs_fit)
    log_psd = np.log(psd_fit)
    
    slope, intercept = np.polyfit(log_freqs, log_psd, 1)
    
    print(f"  Number of points used: {len(log_freqs)}")
    print(f"  Fitted slope: {slope:.2f}")
    
    return slope, intercept, freqs_fit


def plot_spectrum(frequencies: np.ndarray, psd: np.ndarray, r_center: float,
                 min_angle_deg: float, max_angle_deg: float,
                 slope_data: Optional[Tuple] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot frequency spectrum with optional power law fit"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.loglog(frequencies, psd, 'b-', linewidth=1.5, label='Spectrum')
    
    # Add power law fit if available
    if slope_data is not None:
        slope, intercept, freqs_fit = slope_data
        psd_fit = np.exp(intercept) * (freqs_fit ** slope)
        ax.loglog(freqs_fit, psd_fit, 'r--', linewidth=2,
                 label=f'Power law: $\\omega^{{{slope:.2f}}}$')
        ax.legend(fontsize=12)
    
    ax.set_xlabel(r'Frequency [$\omega_c$]', size=16)
    ax.set_ylabel(r'PSD [$|\Phi|^2 / \omega_c$]', size=16)
    ax.set_title(f'Frequency Spectrum at r/a = {r_center:.3f}\n' +
                f'(Angle avg.: {min_angle_deg:.0f}° to {max_angle_deg:.0f}°)', size=14)
    
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    fig.tight_layout()
    return fig, ax


def contour_phi_time_and_spectrum(dirname: str, n: int, dn: int, n1: int,
                                  min_angle_deg: float, max_angle_deg: float,
                                  min_r: float = 0.4, max_r: float = 0.6,
                                  compute_slope: bool = True):
    """
    Main function: Process phi data and create contour plot and spectrum
    
    Args:
        dirname: Directory containing simulation data
        n: Number of time steps to process
        dn: Increment between time steps
        n1: Starting time step index
        min_angle_deg: Minimum angle in degrees
        max_angle_deg: Maximum angle in degrees
        min_r: Minimum normalized radius (r/a) for spectrum
        max_r: Maximum normalized radius (r/a) for spectrum
        compute_slope: Whether to compute and plot power law slope
    """
    # Load basic grid data
    rg_orig = mylib.read_data(dirname, 'rg')
    R0, inv_rho = mylib.read_data(dirname, 'R0', 'rhostar')
    
    rg_max = 1.0 / inv_rho
    xind = np.squeeze(np.where((rg_orig <= rg_max) & (rg_orig > 0.5 * rg_max)))
    rg = rg_orig[xind] / rg_max  # Normalized radius
    
    # Load ballooning angle
    bal_ang_orig = mylib.esti_bal_angle(dirname)
    bal_ang = bal_ang_orig[:, xind]
    
    # Get angle indices
    min_rad, max_rad = validate_angle_range(min_angle_deg, max_angle_deg)
    angles_rad = bal_ang_orig[:, 0]
    angle_indices = get_angle_indices(angles_rad, min_rad, max_rad)
    
    if len(angle_indices) == 0:
        print("Error: No valid angle indices. Aborting.")
        return
    
    # Process time series
    Phi2D, time_arr = process_phi_timeseries(
        dirname, n, dn, n1, xind, angle_indices, bal_ang, rg
    )
    
    # Create contour plot
    fig_contour, ax_contour = plot_contour(
        Phi2D, time_arr, rg, min_angle_deg, max_angle_deg
    )
    
    # Compute spectrum
    try:
        radial_indices = get_radial_indices(rg, min_r, max_r)
        frequencies, psd, r_center = compute_spectrum(
            Phi2D, time_arr, rg, radial_indices
        )
        
        # Analyze slope if requested
        slope_data = None
        if compute_slope:
            result = analyze_spectrum_slope(frequencies, psd)
            if result is not None:
                slope_data = result
        
        # Plot spectrum
        fig_spec, ax_spec = plot_spectrum(
            frequencies, psd, r_center, min_angle_deg, max_angle_deg, slope_data
        )
        
    except Exception as e:
        print(f"Error in spectrum analysis: {e}")
        import traceback
        traceback.print_exc()
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze phi fluctuations: contour plot and frequency spectrum',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('dirname', type=str, help='Directory containing simulation data')
    
    # Time parameters
    parser.add_argument('-n1', type=int, default=0, 
                       help='Starting time step index')
    parser.add_argument('-n', type=int, default=100, 
                       help='Number of time steps to process')
    parser.add_argument('-dn', type=int, default=1, 
                       help='Increment between time steps')
    
    # Angular range
    parser.add_argument('--min_angle', type=float, default=-30.0,
                       help='Minimum angle (degrees) for averaging')
    parser.add_argument('--max_angle', type=float, default=30.0,
                       help='Maximum angle (degrees) for averaging')
    
    # Radial range for spectrum
    parser.add_argument('--min_r', type=float, default=0.4,
                       help='Minimum normalized radius (r/a) for spectrum')
    parser.add_argument('--max_r', type=float, default=0.6,
                       help='Maximum normalized radius (r/a) for spectrum')
    
    # Analysis options
    parser.add_argument('--no_slope', action='store_true',
                       help='Skip power law slope analysis')
    
    args = parser.parse_args()
    
    contour_phi_time_and_spectrum(
        dirname=args.dirname,
        n=args.n,
        dn=args.dn,
        n1=args.n1,
        min_angle_deg=args.min_angle,
        max_angle_deg=args.max_angle,
        min_r=args.min_r,
        max_r=args.max_r,
        compute_slope=not args.no_slope
    )