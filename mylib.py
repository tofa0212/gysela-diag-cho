"""
GYSELA Data Analysis Library - Refactored Version

This module provides utilities for reading, processing, and visualizing
GYSELA simulation data stored in HDF5 format.
"""

import numpy as np
import os
import glob
import re
import h5py
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import ticker
from fractions import Fraction
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass

from gysela_name import dict_filename


# ============================================================================
# Constants
# ============================================================================
DEFAULT_TIME_PERIOD = 4.0e4
DEFAULT_TIME_OFFSET = 1.45e5
DEFAULT_WINDOW_SIZE = 101
EPSILON = 1e-12
DEFAULT_FALLBACK_DIR = '/zhisongqu_data/gysela_irene/diag_test/DIS_ITG_INIT'


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
    """Configuration for file path handling"""
    fallback_dir: Optional[str] = DEFAULT_FALLBACK_DIR
    
    
_config = Config()


def set_fallback_dir(path: Optional[str]) -> None:
    """
    Set the fallback directory for file searches.
    
    Args:
        path: Path to fallback directory, or None to disable fallback
    """
    _config.fallback_dir = path


# ============================================================================
# File I/O Functions
# ============================================================================
def _extract_run_number(filepath: str) -> int:
    """
    Extract run number from filepath.
    
    Args:
        filepath: Path containing run number pattern (e.g., 'r123')
        
    Returns:
        Run number as integer, or -1 if not found
    """
    match = re.search(r'r(\d+)', filepath)
    return int(match.group(1)) if match else -1


def _find_file_with_wildcard(base_path: str, filename: str) -> Optional[str]:
    """
    Find file with run number wildcard, returning the highest run number.
    
    Args:
        base_path: Directory to search in
        filename: Filename pattern with run number (e.g., 'data_r001.h5')
        
    Returns:
        Path to file with highest run number, or None if not found
    """
    pattern = os.path.join(base_path, re.sub(r'r\d+', 'r*', filename))
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    files.sort(key=_extract_run_number)
    return files[-1]


def build_file_path(dirname: str, arg: str, t1: Optional[int] = None, 
                   spnum: int = 0) -> str:
    """
    Build file path from directory and data identifier.
    
    Args:
        dirname: Base directory path
        arg: Data identifier key from dict_filename
        t1: Time index for filename formatting
        spnum: Species number
        
    Returns:
        Full path to the data file
        
    Raises:
        KeyError: If arg is not found in dict_filename
        FileNotFoundError: If file not found in primary or fallback location
    """
    if arg not in dict_filename:
        raise KeyError(f"Unknown data identifier: {arg}")
    
    subpath, fn_template = dict_filename[arg]
    
    # Format filename
    if t1 is not None:
        filename = fn_template.format(t1=t1)
    else:
        filename = fn_template
    
    # Build primary path
    base_path = os.path.join(os.path.abspath(dirname), f'sp{spnum}', subpath)
    file_path = os.path.join(base_path, filename)
    
    # Check if file exists
    if os.path.exists(file_path):
        return file_path
    
    # Try with wildcard
    wildcard_path = _find_file_with_wildcard(base_path, filename)
    if wildcard_path:
        return wildcard_path
    
    # Try fallback directory
    if _config.fallback_dir is not None:
        fallback_base = os.path.join(
            os.path.abspath(_config.fallback_dir), 
            f'sp{spnum}', 
            subpath
        )
        fallback_path = os.path.join(fallback_base, filename)
        
        if os.path.exists(fallback_path):
            return fallback_path
        
        # Try fallback with wildcard
        fallback_wildcard = _find_file_with_wildcard(fallback_base, filename)
        if fallback_wildcard:
            return fallback_wildcard
    
    raise FileNotFoundError(
        f"File not found: {file_path}\n"
        f"(Also checked with wildcard and fallback directory)"
    )


def read_hdf5_data(filepath: str, *args: str) -> Dict[str, np.ndarray]:
    """
    Read data from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        *args: Dataset names to read
        
    Returns:
        Dictionary mapping dataset names to arrays
        
    Raises:
        IOError: If file cannot be read
        KeyError: If dataset not found in file
    """
    try:
        with h5py.File(filepath, 'r') as f:
            data = {arg: np.array(f[arg]) for arg in args}
            return data
    except (IOError, OSError) as e:
        raise IOError(f"Error reading file {filepath}: {e}")
    except KeyError as e:
        available_keys = list(h5py.File(filepath, 'r').keys())
        raise KeyError(
            f"Dataset {e} not found in {filepath}. "
            f"Available datasets: {available_keys}"
        )


def read_data(dirname: str, *args: str, t1: Optional[int] = None, 
             spnum: int = 0) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Read data from GYSELA output files.
    
    Args:
        dirname: Base directory path
        *args: Data identifier(s) to read
        t1: Time index for filename formatting
        spnum: Species number
        
    Returns:
        Single array if one arg provided, list of arrays otherwise
        
    Raises:
        ValueError: If no args provided
        FileNotFoundError: If file not found
        IOError: If file cannot be read
    """
    if len(args) == 0:
        raise ValueError("At least one data identifier must be provided")
    
    filepath = build_file_path(dirname, args[0], t1, spnum)
    data = read_hdf5_data(filepath, *args)
    
    return [data[arg] for arg in args] if len(args) > 1 else data[args[0]]


def count_files(directory: str) -> int:
    """
    Count number of files in a directory.
    
    Args:
        directory: Path to directory
        
    Returns:
        Number of files (not including subdirectories)
    """
    file_count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                file_count += 1
    return file_count


# ============================================================================
# FFT and Signal Processing Functions
# ============================================================================
def _apply_fft_filter_inplace(fft_data: np.ndarray, ax: int, 
                             filt: List[int]) -> None:
    """
    Apply FFT filtering in-place by zeroing out specified modes.
    
    Args:
        fft_data: FFT-transformed data (modified in-place)
        ax: Axis along which FFT was performed
        filt: List of mode numbers to zero out
    """
    for n in filt:
        if ax == 0:
            fft_data[n, :] = 0.0
            fft_data[-n, :] = 0.0
        else:
            fft_data[:, n] = 0.0
            fft_data[:, -n] = 0.0


def fft_filter(data: np.ndarray, ax: int = 0, 
              filt: Optional[List[int]] = None) -> np.ndarray:
    """
    Apply FFT filtering to remove specific frequency modes.
    
    Args:
        data: Input array to filter
        ax: Axis along which to perform FFT (0 or 1)
        filt: List of mode numbers to filter out. Default: [0, 1]
        
    Returns:
        RMS of filtered data along the specified axis
        
    Raises:
        ValueError: If ax is not 0 or 1
    """
    if ax not in [0, 1]:
        raise ValueError(f"ax must be 0 or 1, got {ax}")
    
    if filt is None:
        filt = [0, 1]
    
    fft_data = np.fft.fft(data, axis=ax)
    _apply_fft_filter_inplace(fft_data, ax, filt)
    
    ifft_data = np.fft.ifft(fft_data, axis=ax)
    res = np.sqrt(np.mean(np.abs(ifft_data)**2, axis=ax))
    
    return res


def mode_decomp(data: np.ndarray, ax: int = 0, 
               mode: Optional[List[int]] = None) -> np.ndarray:
    """
    Decompose data into specified Fourier modes.
    
    Args:
        data: Input array to decompose
        ax: Axis along which to perform FFT (0 or 1)
        mode: List of mode indices to extract. Default: [0, 1]
        
    Returns:
        FFT coefficients for specified modes
        
    Raises:
        ValueError: If no valid mode indices provided or ax invalid
    """
    if ax not in [0, 1]:
        raise ValueError(f"ax must be 0 or 1, got {ax}")
    
    if mode is None:
        mode = [0, 1]
    
    fft_data = np.fft.fft(data, axis=ax)
    n_modes = fft_data.shape[ax]
    
    # Filter valid modes (handle negative indices correctly)
    valid_modes = [m for m in mode if -n_modes <= m < n_modes]
    
    if len(valid_modes) == 0:
        raise ValueError(
            f"No valid mode indices provided. "
            f"Valid range: [{-n_modes}, {n_modes-1}]"
        )
    
    # Extract modes
    if ax == 0:
        res = fft_data[valid_modes, :]
    else:
        res = fft_data[:, valid_modes]
    
    return res


def spatial_broadening_analysis(data: np.ndarray, dx: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Analyze spatial broadening through Fourier space variance.
    
    Args:
        data: 2D array (time × space)
        dx: Spatial coordinate spacing
        
    Returns:
        Tuple of (spectral_variances, fft_frequencies, energy_spectrum)
    """
    times = data.shape[0]
    spatial_points = data.shape[1]
    
    # Fourier transform for energy distribution
    fft_data = np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)
    fft_freq = np.fft.fftshift(np.fft.fftfreq(spatial_points, d=dx))
    
    # Energy spectrum
    energy_spectrum = np.abs(fft_data)**2
    
    # Calculate spectral variance for each time point
    spectral_variances = np.zeros(times)
    
    for t in range(times):
        # Normalize to probability distribution
        prob_distribution = energy_spectrum[t] / np.sum(energy_spectrum[t])
        
        # Calculate variance in spatial frequency
        mean_freq = np.sum(fft_freq * prob_distribution)
        spectral_variances[t] = np.sum(
            ((fft_freq - mean_freq)**2) * prob_distribution
        )
    
    return spectral_variances, fft_freq, energy_spectrum


def apply_time_window(data_2d: np.ndarray, 
                     window_size: int = DEFAULT_WINDOW_SIZE) -> np.ndarray:
    """
    Apply moving average window to remove low-frequency trends.
    
    Args:
        data_2d: 2D array to process
        window_size: Size of moving average window (should be odd)
        
    Returns:
        Processed array with trend removed
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    data_processed = data_2d.copy()
    window = np.ones(window_size) / window_size
    
    for i in range(data_2d.shape[0]):
        pad_data = np.pad(
            data_2d[i, :], 
            pad_width=window_size//2, 
            mode='reflect'
        )
        average = np.convolve(pad_data, window, mode='valid')
        data_processed[i, :] = data_2d[i, :] - average
    
    return data_processed


# ============================================================================
# Common Data Loading Functions
# ============================================================================
def load_normalized_grid(dirname: str, spnum: int = 0,
                        r_min: float = 0.0, r_max: float = np.inf,
                        return_mask: bool = False) -> Dict[str, Any]:
    """
    Load and normalize radial grid with optional range selection.

    This is the most commonly used grid loading pattern, appearing in 13+ files.

    Args:
        dirname: Base directory path
        spnum: Species number
        r_min: Minimum normalized radial coordinate
        r_max: Maximum normalized radial coordinate
        return_mask: If True, return boolean mask; if False, return indices

    Returns:
        Dictionary containing:
            - 'rg': Normalized and selected radial grid array
            - 'rg_full': Full normalized radial grid (before selection)
            - 'R0': Normalized major radius
            - 'rhostar': rho_star normalization factor
            - 'mask': Boolean mask or indices for selected range

    Example:
        >>> grid = load_normalized_grid('/path/to/data', r_min=0.7, r_max=1.2)
        >>> rg = grid['rg']
        >>> R0 = grid['R0']
    """
    # Load basic grid data
    rg_raw = read_data(dirname, 'rg', spnum=spnum)
    R0, rhostar = read_data(dirname, 'R0', 'rhostar', spnum=spnum)

    # Normalize
    rg_normalized = rg_raw * rhostar
    R0_normalized = R0 * rhostar

    # Create mask/indices for selected range
    if return_mask:
        mask = (rg_normalized >= r_min) & (rg_normalized <= r_max)
    else:
        mask = np.squeeze(np.where((rg_normalized >= r_min) & (rg_normalized <= r_max)))

    # Select radial range
    rg_selected = rg_normalized[mask]

    return {
        'rg': rg_selected,
        'rg_full': rg_normalized,
        'R0': R0_normalized,
        'rhostar': rhostar,
        'mask': mask,
    }


def load_temperature_profile(dirname: str, timestep: int, spnum: int = 0,
                            radial_mask: Optional[np.ndarray] = None,
                            return_components: bool = False) -> Union[
                                np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]
                            ]:
    """
    Load temperature profile (T = P/n) at given timestep.

    Common pattern appearing in 8+ files for loading ion/electron temperature.

    Args:
        dirname: Base directory path
        timestep: Time step index
        spnum: Species number (0=ions, 1=electrons)
        radial_mask: Optional mask/indices to select radial points
        return_components: If True, return (P, n, T); if False, return T only

    Returns:
        If return_components=False: Temperature array
        If return_components=True: Tuple of (pressure, density, temperature)

    Example:
        >>> Ti = load_temperature_profile('/path/to/data', timestep=100)
        >>> P, n, Ti = load_temperature_profile('/path/to/data', 100,
        ...                                      return_components=True)
    """
    # Load pressure and density
    P, n = read_data(dirname, 'stress_FSavg', 'dens_FSavg',
                    t1=timestep, spnum=spnum)

    # Calculate temperature
    T = P / n

    # Apply radial mask if provided
    if radial_mask is not None:
        P = P[radial_mask]
        n = n[radial_mask]
        T = T[radial_mask]

    if return_components:
        return P, n, T
    else:
        return T


def load_2d_temperature(dirname: str, timestep: int, spnum: int = 0,
                       radial_mask: Optional[np.ndarray] = None,
                       angle_mask: Optional[np.ndarray] = None,
                       return_density: bool = False) -> Union[
                           Tuple[np.ndarray, np.ndarray],
                           Tuple[np.ndarray, np.ndarray, np.ndarray]
                       ]:
    """
    Load 2D temperature fields with anisotropy (parallel/perpendicular).

    Common pattern for loading 2D temperature data from GC moments.

    Args:
        dirname: Base directory path
        timestep: Time step index
        spnum: Species number
        radial_mask: Optional mask for radial selection
        angle_mask: Optional mask for angular selection
        return_density: If True, also return density field

    Returns:
        If return_density=False: (T_par, T_perp)
        If return_density=True: (T_par, T_perp, density)

    Example:
        >>> T_par, T_perp = load_2d_temperature('/path/to/data', timestep=100)
        >>> T_par, T_perp, n = load_2d_temperature('/path/to/data', 100,
        ...                                         return_density=True)
    """
    # Load pressure components and density
    P_par, P_perp, dens = read_data(
        dirname, 'PparGC_rtheta', 'PperpGC_rtheta', 'densGC_rtheta',
        t1=timestep, spnum=spnum
    )

    # Calculate temperatures
    T_par = P_par / dens
    T_perp = P_perp / dens

    # Apply masks if provided
    if radial_mask is not None:
        T_par = T_par[:, radial_mask]
        T_perp = T_perp[:, radial_mask]
        dens = dens[:, radial_mask]

    if angle_mask is not None:
        T_par = T_par[angle_mask, :]
        T_perp = T_perp[angle_mask, :]
        dens = dens[angle_mask, :]

    if return_density:
        return T_par, T_perp, dens
    else:
        return T_par, T_perp


def load_geometry_data(dirname: str, spnum: int = 0,
                      radial_mask: Optional[np.ndarray] = None,
                      components: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Load geometry data with optional masking.

    Args:
        dirname: Base directory path
        spnum: Species number
        radial_mask: Optional mask for radial selection
        components: List of components to load. Default: ['B', 'jacob_space', 'thetag']
                   Available: 'B', 'jacob_space', 'thetag', 'R', 'Z', 'psi',
                             'B_gradtheta', 'B_gradphi', 'Btheta'

    Returns:
        Dictionary with requested components, properly masked

    Example:
        >>> geom = load_geometry_data('/path/to/data',
        ...                           components=['B', 'jacob_space', 'thetag'])
        >>> B = geom['B']
        >>> jacob = geom['jacob_space']
    """
    if components is None:
        components = ['B', 'jacob_space', 'thetag']

    # Load requested components
    data = {}
    for comp in components:
        data[comp] = read_data(dirname, comp, spnum=spnum)

    # Apply radial mask to 2D arrays if provided
    if radial_mask is not None:
        for key, value in data.items():
            if value.ndim == 2:
                data[key] = value[:, radial_mask]
            elif value.ndim == 1 and len(value) > len(np.atleast_1d(radial_mask)):
                # For 1D arrays that match radial dimension
                data[key] = value[radial_mask]

    return data


def load_common_geometry(dirname: str, spnum: int = 0,
                        min_r: float = 0.7, max_r: float = 1.1) -> Dict[str, Any]:
    """
    Load common geometry data for analysis.
    
    This function loads and processes:
    - Radial grid (normalized by rho_star)
    - R0 (major radius)
    - Ballooning angle
    - Radial indices for specified range
    
    Args:
        dirname: Base directory path
        spnum: Species number
        min_r: Minimum normalized radial coordinate
        max_r: Maximum normalized radial coordinate
        
    Returns:
        Dictionary containing:
            - 'rg': Selected radial grid array
            - 'R0': Major radius (normalized)
            - 'inv_rho': Inverse rho_star
            - 'bal_ang': Ballooning angle array (selected range)
            - 'xind': Radial indices
            - 'min_r': Minimum radius used
            - 'max_r': Maximum radius used
            
    Example:
        >>> geom = load_common_geometry('/path/to/data', min_r=0.7, max_r=1.1)
        >>> rg = geom['rg']
        >>> bal_ang = geom['bal_ang']
    """
    # Load basic geometry
    rg = read_data(dirname, 'rg', spnum=spnum)
    R0, inv_rho = read_data(dirname, 'R0', 'rhostar', spnum=spnum)
    
    # Normalize radial coordinate
    rg *= inv_rho
    
    # Select radial range
    xind = np.squeeze(np.where((rg <= max_r) & (rg >= min_r)))
    R0 *= inv_rho
    rg_selected = rg[xind]
    
    # Load ballooning angle
    bal_ang = esti_bal_angle(dirname)
    bal_ang_selected = bal_ang[:, xind]
    
    return {
        'rg': rg_selected,
        'R0': R0,
        'inv_rho': inv_rho,
        'bal_ang': bal_ang_selected,
        'xind': xind,
        'min_r': min_r,
        'max_r': max_r,
    }


def get_angle_indices(angles: np.ndarray,
                     min_angle: float,
                     max_angle: float,
                     in_degrees: bool = True) -> np.ndarray:
    """
    Get indices for angle range with automatic wrap-around handling.

    This common pattern appears in 7+ files. Handles wrap-around cases
    (e.g., [-30°, 30°] crossing 0) automatically.

    Args:
        angles: Angle array (radians or degrees depending on in_degrees)
        min_angle: Minimum angle
        max_angle: Maximum angle
        in_degrees: If True, inputs are in degrees; if False, in radians

    Returns:
        Array of indices within specified range

    Raises:
        ValueError: If no angles found in specified range

    Example:
        >>> theta_indices = get_angle_indices(theta_grid, -30, 30, in_degrees=True)
        >>> data_selected = data[theta_indices, :]
    """
    # Convert to radians if needed
    if in_degrees:
        min_rad = np.deg2rad(min_angle)
        max_rad = np.deg2rad(max_angle)
        angle_rad = angles if not in_degrees else np.deg2rad(angles)
    else:
        min_rad = min_angle
        max_rad = max_angle
        angle_rad = angles

    # Normalize to [0, 2π) for robust comparison
    angles_norm = angle_rad % (2 * np.pi)
    min_norm = min_rad % (2 * np.pi)
    max_norm = max_rad % (2 * np.pi)

    # Handle wrap-around
    if min_norm <= max_norm:
        mask = (angles_norm >= min_norm) & (angles_norm <= max_norm)
    else:
        # Wrap-around case: angles >= min OR angles <= max
        mask = (angles_norm >= min_norm) | (angles_norm <= max_norm)

    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(
            f"No angles found in range [{min_angle}, {max_angle}]. "
            f"Available angle range: [{np.rad2deg(angle_rad.min()):.1f}°, "
            f"{np.rad2deg(angle_rad.max()):.1f}°]"
        )

    return indices


def setup_angle_analysis(bal_ang: np.ndarray,
                        min_angle_deg: float = -30.0,
                        max_angle_deg: float = 30.0) -> np.ndarray:
    """
    Setup angle indices for analysis based on ballooning angle range.
    
    This function identifies poloidal angle indices within a specified
    degree range, handling wrap-around cases (e.g., [-30, 30] degrees
    crossing 0).
    
    Args:
        bal_ang: Ballooning angle array (theta x radius)
        min_angle_deg: Minimum angle in degrees
        max_angle_deg: Maximum angle in degrees
        
    Returns:
        Array of angle indices within the specified range
        
    Raises:
        ValueError: If no angles found in specified range
        
    Example:
        >>> bal_ang = esti_bal_angle('/path/to/data')
        >>> angle_indices = setup_angle_analysis(bal_ang, -30, 30)
        >>> selected_data = data[angle_indices, :]
    """
    # Use angles from the first radial point as reference
    angles_rad = bal_ang[:, 0]
    
    # Convert degree range to radians
    min_angle_rad = np.deg2rad(min_angle_deg)
    max_angle_rad = np.deg2rad(max_angle_deg)
    
    # Normalize angles to [0, 2π) for robust comparison
    angles_norm = angles_rad % (2 * np.pi)
    min_angle_norm = min_angle_rad % (2 * np.pi)
    max_angle_norm = max_angle_rad % (2 * np.pi)
    
    if min_angle_norm <= max_angle_norm:
        # Normal case: min_angle <= angles <= max_angle
        angle_mask = (angles_norm >= min_angle_norm) & (angles_norm <= max_angle_norm)
    else:
        # Wrap-around case: angles >= min_angle OR angles <= max_angle
        angle_mask = (angles_norm >= min_angle_norm) | (angles_norm <= max_angle_norm)
    
    angle_indices = np.where(angle_mask)[0]
    
    if len(angle_indices) == 0:
        raise ValueError(
            f"No angles found in range [{min_angle_deg}, {max_angle_deg}] degrees. "
            f"Available angle range: [{np.rad2deg(angles_rad.min()):.1f}, "
            f"{np.rad2deg(angles_rad.max()):.1f}] degrees"
        )
    
    return angle_indices


def load_and_select_data(dirname: str, dtype: str, t1: int, 
                        geom: Dict[str, Any], spnum: int = 0,
                        average_toroidal: bool = True) -> np.ndarray:
    """
    Load data and apply radial selection from geometry.
    
    Args:
        dirname: Base directory path
        dtype: Data type identifier
        t1: Time step index
        geom: Geometry dictionary from load_common_geometry
        spnum: Species number
        average_toroidal: If True, average over toroidal dimension for 3D data
        
    Returns:
        Data array with radial selection applied
        
    Example:
        >>> geom = load_common_geometry('/path/to/data')
        >>> data = load_and_select_data('/path/to/data', 'Phirth', 10, geom)
    """
    data = read_data(dirname, dtype, t1=t1, spnum=spnum)
    
    # Average over toroidal angle if 3D
    if average_toroidal and data.ndim == 3:
        data = np.mean(data, axis=0)
    
    # Select radial range
    xind = geom['xind']
    if data.ndim == 2:
        data = data[:, xind]
    elif data.ndim == 1:
        # Handle 1D data (like radial profiles)
        data = data[xind] if len(data) > len(xind) else data
    
    return data


# ============================================================================
# Physical Calculations
# ============================================================================
def calc_gradient_scale_length(profile: np.ndarray,
                               radial_grid: np.ndarray,
                               R0: float) -> np.ndarray:
    """
    Calculate inverse gradient scale length R/L_X = -R0 * (dX/dr) / X

    Common pattern appearing in 6+ files for calculating gradient scale
    lengths of temperature, density, or pressure profiles.

    Args:
        profile: Quantity profile (T, n, P, etc.) [n_radial]
        radial_grid: Radial coordinate array [n_radial]
        R0: Major radius normalization factor

    Returns:
        R_over_L: Inverse gradient scale length [n_radial]

    Example:
        >>> Ti = load_temperature_profile(dirname, timestep)
        >>> RLT = calc_gradient_scale_length(Ti, rg, R0)
    """
    # Calculate gradient
    grad = np.gradient(profile, radial_grid)

    # Calculate R/L with safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        R_over_L = -grad / profile * R0
        # Set to 0 where profile is 0
        R_over_L = np.nan_to_num(R_over_L, nan=0.0, posinf=0.0, neginf=0.0)

    return R_over_L


def calc_exb_velocity_2d(phi: np.ndarray,
                        rg: np.ndarray,
                        thetag: np.ndarray,
                        B: np.ndarray,
                        jacob_space: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate ExB velocity components from electrostatic potential.

    Common pattern in Reynolds stress and transport calculations.

    Args:
        phi: Electrostatic potential [n_phi, n_theta, n_r] or [n_theta, n_r]
        rg: Radial grid
        thetag: Poloidal angle grid
        B: Magnetic field magnitude
        jacob_space: Jacobian of coordinate transformation

    Returns:
        v_r: Radial ExB velocity (same shape as phi)
        v_theta: Poloidal ExB velocity (same shape as phi)

    Formula:
        v_r = -dPhi/dtheta / (B * J)
        v_theta = dPhi/dr / (B * J)

    Example:
        >>> Phi3D = mylib.read_data(dirname, 'Phi_3D', t1=timestep)
        >>> v_r, v_theta = calc_exb_velocity_2d(Phi3D, rg, thetag, B, jacob)
    """
    # Determine dimensionality
    is_3d = (phi.ndim == 3)

    # Calculate gradients
    if is_3d:
        dr_phi = np.gradient(phi, rg, axis=2)
        dth_phi = np.gradient(phi, thetag, axis=1)
    else:
        dr_phi = np.gradient(phi, rg, axis=1)
        dth_phi = np.gradient(phi, thetag, axis=0)

    # Calculate B * J
    B_jacob = B * jacob_space

    # Calculate velocities
    if is_3d:
        v_r = -dth_phi / B_jacob[np.newaxis, :, :]
        v_theta = dr_phi / B_jacob[np.newaxis, :, :]
    else:
        v_r = -dth_phi / B_jacob
        v_theta = dr_phi / B_jacob

    return v_r, v_theta


def calc_diamagnetic_velocity_2d(pressure: np.ndarray,
                                 density: np.ndarray,
                                 rg: np.ndarray,
                                 thetag: np.ndarray,
                                 B: np.ndarray,
                                 jacob_space: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate diamagnetic velocity components from pressure and density.

    Common pattern in Reynolds stress and transport calculations.

    Args:
        pressure: Pressure field [n_theta, n_r]
        density: Density field [n_theta, n_r]
        rg: Radial grid
        thetag: Poloidal angle grid
        B: Magnetic field magnitude
        jacob_space: Jacobian of coordinate transformation

    Returns:
        v_r: Radial diamagnetic velocity [n_theta, n_r]
        v_theta: Poloidal diamagnetic velocity [n_theta, n_r]

    Formula:
        v_r = -dP/dtheta / (B * J * n)
        v_theta = dP/dr / (B * J * n)

    Example:
        >>> P_perp, n = mylib.read_data(dirname, 'PperpGC_rtheta',
        ...                              'densGC_rtheta', t1=timestep)
        >>> v_r, v_theta = calc_diamagnetic_velocity_2d(P_perp, n, rg, thetag,
        ...                                              B, jacob)
    """
    # Calculate gradients of pressure
    dr_P = np.gradient(pressure, rg, axis=1)
    dth_P = np.gradient(pressure, thetag, axis=0)

    # Calculate B * J * n
    B_jacob_n = B * jacob_space * density

    # Calculate velocities
    v_r = -dth_P / B_jacob_n
    v_theta = dr_P / B_jacob_n

    return v_r, v_theta


def calc_reynolds_stress_from_velocities(v_r: np.ndarray,
                                         v_theta: np.ndarray,
                                         rg: np.ndarray,
                                         thetag: np.ndarray) -> np.ndarray:
    """
    Calculate Reynolds stress from velocity field.

    Formula: RS = -v_theta * d(v_r)/dr + v_r * d(v_theta)/dtheta

    Args:
        v_r: Radial velocity component
        v_theta: Poloidal velocity component
        rg: Radial grid
        thetag: Poloidal angle grid

    Returns:
        RS: Reynolds stress field (same shape as velocities)

    Example:
        >>> v_r_ExB, v_theta_ExB = calc_exb_velocity_2d(...)
        >>> RS_ExB = calc_reynolds_stress_from_velocities(v_r_ExB, v_theta_ExB,
        ...                                               rg, thetag)
    """
    # Determine dimensionality
    is_3d = (v_r.ndim == 3)

    if is_3d:
        # 3D case: [n_phi, n_theta, n_r]
        dr_vr = np.gradient(v_r, rg, axis=2)
        dth_vtheta = np.gradient(v_theta, thetag, axis=1)
    else:
        # 2D case: [n_theta, n_r]
        dr_vr = np.gradient(v_r, rg, axis=1)
        dth_vtheta = np.gradient(v_theta, thetag, axis=0)

    # Calculate Reynolds stress components
    RS1 = -v_theta * dr_vr
    RS2 = v_r * dth_vtheta

    return RS1 + RS2


# ============================================================================
# Ballooning Angle Functions
# ============================================================================
def esti_bal_angle(filepath: str) -> np.ndarray:
    """
    Estimate ballooning angle from magnetic field components.
    
    Args:
        filepath: Path to file containing magnetic field data
        
    Returns:
        Ballooning angle array
        
    Raises:
        ValueError: If invalid dS/dt values detected
    """
    Bp, Bt = read_data(filepath, 'B_gradtheta', 'B_gradphi')
    
    # Prevent division by zero
    Bp_safe = np.where(np.abs(Bp) < EPSILON, EPSILON * np.sign(Bp), Bp)
    dSdt = Bt / Bp_safe
    
    if np.any(np.isnan(dSdt)) or np.any(np.isinf(dSdt)):
        raise ValueError("Invalid dS/dt values detected")
    
    bal_angle = np.zeros_like(dSdt)
    
    # Cumulative integration
    cum_int = np.cumsum(dSdt, axis=0)
    tot_int = cum_int[-1, :]
    
    # Normalize to [0, 2π]
    for j in range(dSdt.shape[1]):
        bal_angle[:, j] = 2.0 * np.pi * cum_int[:, j] / tot_int[j]
        bal_angle[:, j] -= bal_angle[0, j]
    
    return bal_angle


def interp_bal(bal_angle: np.ndarray, data: np.ndarray, 
              thetag: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Interpolate data to uniform grid using ballooning angle (1D version).
    
    Args:
        bal_angle: Ballooning angle array
        data: Data array to interpolate
        thetag: Target angle grid. If None, uses uniform [0, 2π]
        
    Returns:
        Interpolated data
        
    Raises:
        ValueError: If shapes don't match
    """
    if bal_angle.shape != data.shape:
        raise ValueError(
            f"Shape mismatch: bal_angle {bal_angle.shape} vs "
            f"data {data.shape}"
        )
    
    f_interp = interp1d(
        bal_angle, data, 
        kind='cubic', 
        fill_value="extrapolate"
    )
    
    if thetag is None:
        new_ang = np.linspace(0, 2*np.pi, len(data))
    else:
        new_ang = thetag
    
    interp_data = f_interp(new_ang)
    
    return interp_data


def interp2_bal(bal_angle: np.ndarray, data: np.ndarray, 
               thetag: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Interpolate 2D data to uniform grid using ballooning angle.
    
    Performs interpolation for each column independently with periodic
    boundary conditions.
    
    Args:
        bal_angle: Ballooning angle array (same shape as data)
        data: 2D data array to interpolate
        thetag: Target angle grid. If None, uses uniform [0, 2π]
        
    Returns:
        Interpolated data on uniform grid
        
    Raises:
        ValueError: If shapes don't match
    """
    if bal_angle.shape != data.shape:
        raise ValueError(
            f"Shape mismatch: bal_angle {bal_angle.shape} vs "
            f"data {data.shape}"
        )
    
    interp_data = np.zeros_like(data)
    
    if thetag is None:
        new_ang = np.linspace(0, 2*np.pi, data.shape[0])
    else:
        new_ang = thetag
    
    # Interpolate each column with periodic extension
    for j in range(data.shape[1]):
        ext_ang = np.concatenate([
            bal_angle[:, j], 
            [bal_angle[0, j] + 2*np.pi]
        ])
        ext_data = np.concatenate([data[:, j], [data[0, j]]])
        
        f_interp = interp1d(
            ext_ang, ext_data, 
            kind='cubic', 
            fill_value="extrapolate"
        )
        interp_data[:, j] = f_interp(new_ang)
    
    return interp_data


def inv_interp2_bal(bal_angle: np.ndarray, interp_data: np.ndarray, 
                   thetag: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Inverse interpolation: transform data from uniform grid back to 
    ballooning angle grid.
    
    Args:
        bal_angle: Original ballooning angle array
        interp_data: Data on uniform grid (from interp2_bal)
        thetag: Uniform angle grid used in interpolation. If None, uses [0, 2π]
        
    Returns:
        Data approximation on original ballooning angle grid
        
    Raises:
        ValueError: If shapes don't match
    """
    if interp_data.shape != bal_angle.shape:
        raise ValueError(
            f"Shape mismatch: interp_data {interp_data.shape} vs "
            f"bal_angle {bal_angle.shape}"
        )
    
    original_data_approx = np.zeros_like(interp_data)
    
    if thetag is None:
        new_ang = np.linspace(0, 2*np.pi, interp_data.shape[0])
    else:
        new_ang = thetag
    
    # Inverse interpolation for each column
    for j in range(interp_data.shape[1]):
        f_inv_interp = interp1d(
            new_ang, interp_data[:, j],
            kind='cubic',
            fill_value="extrapolate"
        )
        original_data_approx[:, j] = f_inv_interp(bal_angle[:, j])
    
    return original_data_approx


def fft_bal_filter(data: np.ndarray, bal_ang: np.ndarray, 
                  ax: int = 0, filt: Optional[List[int]] = None) -> np.ndarray:
    """
    Apply FFT filtering in ballooning coordinate system.
    
    Args:
        data: Data array to filter
        bal_ang: Ballooning angle array
        ax: Axis along which to perform FFT
        filt: List of mode numbers to filter out. Default: [0, 1]
        
    Returns:
        RMS of filtered data
    """
    if filt is None:
        filt = [0, 1]
    
    # Interpolate to uniform grid
    interp_data = interp2_bal(bal_ang, data)
    
    # Apply FFT filter
    fft_data = np.fft.fft(interp_data, axis=ax)
    _apply_fft_filter_inplace(fft_data, ax, filt)
    
    # Inverse FFT
    ifft_data = np.fft.ifft(fft_data, axis=ax)
    res = np.sqrt(np.mean(np.abs(ifft_data)**2, axis=ax))
    
    return res


# ============================================================================
# Plotting Utility Functions
# ============================================================================
def init_plot_params(figsize: Tuple[float, float] = (8, 6), 
                    font_size: int = 12) -> Tuple[plt.Figure, plt.Axes]:
    """
    Initialize plot with standard parameters.
    
    Args:
        figsize: Figure size in inches (width, height)
        font_size: Font size for axis labels
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='both', labelsize=font_size)
    return fig, ax


def auto_color_limits(data: np.ndarray,
                     mode: str = 'symmetric',
                     percentile: float = 99,
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None) -> Tuple[float, float]:
    """
    Calculate color scale limits automatically.

    Common pattern appearing in 8+ files for setting color limits.

    Args:
        data: Data array for color mapping
        mode: Scaling mode - 'symmetric', 'positive', or 'percentile'
        percentile: Percentile for outlier removal (used in 'percentile' mode)
        vmin: Override minimum (if provided, ignores mode)
        vmax: Override maximum (if provided, ignores mode)

    Returns:
        (vmin, vmax): Color scale limits

    Modes:
        'symmetric': ±max(abs(data)) for diverging colormaps (RdBu_r, etc.)
        'positive': 0 to max for sequential colormaps
        'percentile': Based on percentiles to ignore outliers

    Example:
        >>> vmin, vmax = auto_color_limits(data, mode='symmetric')
        >>> plt.pcolormesh(X, Y, data, vmin=vmin, vmax=vmax, cmap='RdBu_r')
    """
    # Handle provided values
    if vmin is not None and vmax is not None:
        return vmin, vmax

    # Remove NaN/Inf for calculations
    valid_data = data[np.isfinite(data)]

    if len(valid_data) == 0:
        return 0.0, 1.0  # Fallback

    if mode == 'symmetric':
        abs_max = np.max(np.abs(valid_data))
        calc_vmin = -abs_max
        calc_vmax = abs_max
    elif mode == 'positive':
        calc_vmin = 0.0
        calc_vmax = np.max(valid_data)
    elif mode == 'percentile':
        calc_vmin = np.percentile(valid_data, 100 - percentile)
        calc_vmax = np.percentile(valid_data, percentile)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'symmetric', 'positive', or 'percentile'")

    # Override with provided values
    final_vmin = vmin if vmin is not None else calc_vmin
    final_vmax = vmax if vmax is not None else calc_vmax

    return final_vmin, final_vmax


def add_colorbar(mappable, ax: plt.Axes,
                label: str = '',
                font_size: int = 12,
                **kwargs) -> plt.colorbar.Colorbar:
    """
    Add consistently formatted colorbar to plot.

    Common pattern appearing in 10+ files.

    Args:
        mappable: Matplotlib mappable object (result of pcolormesh, contourf, etc.)
        ax: Axes to attach colorbar to
        label: Colorbar label text
        font_size: Font size for label and ticks
        **kwargs: Additional arguments passed to plt.colorbar()

    Returns:
        cbar: Colorbar object

    Example:
        >>> mesh = ax.pcolormesh(X, Y, data, cmap='RdBu_r')
        >>> cbar = add_colorbar(mesh, ax, label='Temperature [eV]', font_size=14)
    """
    cbar = plt.colorbar(mappable, ax=ax, **kwargs)

    if label:
        cbar.set_label(label, size=font_size + 2)

    cbar.ax.tick_params(labelsize=font_size)

    return cbar


def fin_plot_params(fig: plt.Figure, ax: plt.Axes) -> None:
    """
    Finalize plot layout.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes (unused but kept for API compatibility)
    """
    fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    fig.tight_layout()


def setup_contour_plot(fig: plt.Figure, ax: plt.Axes, 
                      font_size: int = 28,
                      with_colorbar: bool = True,
                      colorbar_label: Optional[str] = None,
                      with_time_axis: bool = False,
                      t_p: float = DEFAULT_TIME_PERIOD) -> Tuple[
                          plt.Figure, plt.Axes
                      ]:
    """
    Setup contour plot with colorbar and axis formatting.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        font_size: Font size for labels
        with_colorbar: Whether to add colorbar
        colorbar_label: Label for colorbar
        with_time_axis: Whether to format time axis
        t_p: Time period for axis formatting
        
    Returns:
        Tuple of (figure, axes)
    """
    if with_colorbar:
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=font_size)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=font_size)
    
    if with_time_axis:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_size(font_size)
    
    return fig, ax


def set_time_axis_ticks(ax: plt.Axes, time_array: np.ndarray,
                       t_p: float = DEFAULT_TIME_PERIOD,
                       time_offset: float = DEFAULT_TIME_OFFSET) -> plt.Axes:
    """
    Set time axis ticks as fractions of period.
    
    Args:
        ax: Matplotlib axes
        time_array: Array of time values
        t_p: Time period
        time_offset: Time offset for normalization
        
    Returns:
        Modified axes
    """
    ytick_list = np.arange(time_array[0], time_array[-1] + 1, t_p / 4)
    ytick_labels = [
        float_to_fraction((ttick - time_offset), t_p) 
        for ttick in ytick_list
    ]
    ax.set_yticks(ytick_list, ytick_labels)
    return ax


def add_side_sine_wave(fig: plt.Figure, main_ax: plt.Axes,
                      time_array: np.ndarray,
                      t_p: float = DEFAULT_TIME_PERIOD,
                      time_offset: float = 1.65e5,
                      width_ratio: float = 0.2,
                      position: int = 1) -> Tuple[
                          plt.Figure, plt.Axes, plt.Axes
                      ]:
    """
    Add sine wave visualization to the side of a plot.
    
    Args:
        fig: Matplotlib figure
        main_ax: Main plot axes
        time_array: Array of time values
        t_p: Time period for sine wave
        time_offset: Phase offset
        width_ratio: Width ratio of sine wave panel
        position: Position (unused, kept for compatibility)
        
    Returns:
        Tuple of (figure, main_axes, sine_axes)
    """
    from matplotlib.gridspec import GridSpec
    
    gs = GridSpec(1, 2, width_ratios=[1, width_ratio], wspace=0.1)
    fig.clf()
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Transfer content from main_ax to ax1
    for item in main_ax.get_children():
        ax1.add_artist(item)
    
    # Create sine wave
    y_values = np.linspace(time_array[0], time_array[-1], 1000)
    sine_wave = np.sin(2 * np.pi * (y_values - time_offset) / t_p)
    
    ax2.plot(sine_wave, y_values)
    ax2.fill_betweenx(
        y_values, sine_wave, 
        where=(sine_wave > 0), 
        color='red', alpha=0.5
    )
    ax2.fill_betweenx(
        y_values, sine_wave,
        where=(sine_wave < 0),
        color='blue', alpha=0.5
    )
    
    ax2.set_xlim([-1, 1])
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    
    # Hide spines
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    return fig, ax1, ax2


# ============================================================================
# Formatting Utility Functions
# ============================================================================
def sci_note(value: float, precision: int = 2) -> str:
    """
    Format value in scientific notation for LaTeX.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        LaTeX formatted string
    """
    if value == 0:
        return "0"
    
    exponent = int(np.floor(np.log10(np.abs(value))))
    coefficient = value / 10**exponent
    
    return rf"${coefficient:.{precision}f} \times 10^{{{exponent}}}$"


def float_to_fraction(value: float, base: float) -> str:
    """
    Convert float to fraction relative to base value.
    
    Args:
        value: Numerator value
        base: Denominator base value
        
    Returns:
        LaTeX formatted fraction string
    """
    fraction = Fraction(value / base).limit_denominator()
    
    if fraction.denominator == 1:
        return rf'${fraction.numerator}$' if fraction.numerator != 0 else '0'
    else:
        return rf'${fraction.numerator}/{fraction.denominator}$'


def float_to_fraction2(value: float, base: float, cdenom: int) -> str:
    """
    Convert float to fraction with common denominator.
    
    Args:
        value: Numerator value
        base: Base value for normalization
        cdenom: Common denominator
        
    Returns:
        LaTeX formatted fraction string
    """
    fraction = Fraction(value / base).limit_denominator()
    
    if fraction.numerator == 0:
        return '0'
    else:
        fact = int(cdenom / fraction.denominator)
        print(fact, fraction.numerator, fraction.denominator)
        return rf'$\frac{{{fact * fraction.numerator}}}{{{cdenom}}}$'


# ============================================================================
# Module Info
# ============================================================================
__version__ = "2.1.0"
__all__ = [
    # Configuration
    'set_fallback_dir',
    # File I/O
    'build_file_path',
    'read_hdf5_data',
    'read_data',
    'count_files',
    # Common data loading
    'load_normalized_grid',
    'load_temperature_profile',
    'load_2d_temperature',
    'load_geometry_data',
    'load_common_geometry',
    'get_angle_indices',
    'setup_angle_analysis',
    'load_and_select_data',
    # Physical calculations
    'calc_gradient_scale_length',
    'calc_exb_velocity_2d',
    'calc_diamagnetic_velocity_2d',
    'calc_reynolds_stress_from_velocities',
    # Signal processing
    'fft_filter',
    'mode_decomp',
    'spatial_broadening_analysis',
    'apply_time_window',
    # Ballooning angle
    'esti_bal_angle',
    'interp_bal',
    'interp2_bal',
    'inv_interp2_bal',
    'fft_bal_filter',
    # Plotting
    'init_plot_params',
    'fin_plot_params',
    'setup_contour_plot',
    'set_time_axis_ticks',
    'add_side_sine_wave',
    'auto_color_limits',
    'add_colorbar',
    # Formatting
    'sci_note',
    'float_to_fraction',
    'float_to_fraction2',
]