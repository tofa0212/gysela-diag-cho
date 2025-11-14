import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import mylib
import matplotlib.ticker as ticker
from tqdm import tqdm
from typing import Tuple, Optional
import h5py

# print("Current directory:", os.getcwd())
# print("Files in directory:", os.listdir('.'))
# import sys
# print("Python path:", sys.path)
# Import the data processing module
try:
    from GYS_dataprocess import process_data
except ImportError:
    print("Warning: GYS_dataprocess module not found. Using basic data loading only.")
    
    def process_data(dtype: str, dirname: str, t1: int, spnum: int, mylib):
        """Fallback function when GYS_dataprocess is not available"""
        return mylib.read_data(dirname, dtype, t1=t1, spnum=spnum)


class ContourAnalyzer:
    """Class to handle contour analysis of plasma data"""
    
    def __init__(self, dirname: str, dtype: str, spnum: int = 0, 
                 min_r: float = 0.7, max_r: float = 1.1,
                 min_angle_deg: float = -30.0, max_angle_deg: float = 30.0):
        self.dirname = dirname
        self.dtype = dtype
        self.spnum = spnum
        self.min_angle_deg = min_angle_deg
        self.max_angle_deg = max_angle_deg
        self.min_r = min_r
        self.max_r = max_r
        self.reference_data = None  # Store reference data when needed
        
        # Load common data
        self._load_common_data()
        
    def _load_common_data(self):
        """Load geometry and grid data"""
        self.rg = mylib.read_data(self.dirname, 'rg')
        self.R0, self.inv_rho = mylib.read_data(self.dirname, 'R0', 'rhostar')
        
        # Normalize radial coordinate
        self.rg *= self.inv_rho
        
        # Select radial range
        self.xind = np.squeeze(np.where((self.rg <= self.max_r) & (self.rg >= self.min_r)))
        self.R0 *= self.inv_rho
        self.rg = self.rg[self.xind]
        
        # Load ballooning angle data
        self.bal_ang = mylib.esti_bal_angle(self.dirname)
        self.bal_ang = self.bal_ang[:, self.xind]
        
        # Setup angle analysis
        self._setup_angle_analysis()
        
    def _setup_angle_analysis(self):
        """Setup angle indices for analysis"""
        angles_rad = self.bal_ang[:, 0]  # Use angles from the first radial point
        
        # Convert degree range to radians
        min_angle_rad = np.deg2rad(self.min_angle_deg)
        max_angle_rad = np.deg2rad(self.max_angle_deg)
        
        # Normalize angles and range to [0, 2*pi) for robust comparison
        angles_norm = angles_rad % (2 * np.pi)
        min_angle_norm = min_angle_rad % (2 * np.pi)
        max_angle_norm = max_angle_rad % (2 * np.pi)
        
        if min_angle_norm <= max_angle_norm:
            # Normal case: min_angle <= angles <= max_angle
            angle_mask = (angles_norm >= min_angle_norm) & (angles_norm <= max_angle_norm)
        else:
            # Wrap-around case: angles >= min_angle OR angles <= max_angle
            angle_mask = (angles_norm >= min_angle_norm) | (angles_norm <= max_angle_norm)
            
        self.angle_indices = np.where(angle_mask)[0]
        
        if len(self.angle_indices) == 0:
            print(f"Warning: No angles found in the specified range [{self.min_angle_deg}, {self.max_angle_deg}] degrees.")
    
    def load_reference_data(self, ref_time: int, apply_fft_filter: bool = True, 
                           fft_filter_modes: int = 1):
        """Load and process reference data for subtraction"""
        try:
            print(f"Loading reference data from time step {ref_time}...")
            
            # Use the unified data processing function for reference
            ref_data = process_data(self.dtype, self.dirname, ref_time, self.spnum, mylib)
            
            # Handle special case for Phi data with n0 component
            ref_data_n0 = None
            if self.dtype in ['Phirth', 'phi']:
                try:
                    ref_data_n0 = mylib.read_data(self.dirname, 'Phirth_n0', t1=ref_time, spnum=self.spnum)
                except:
                    pass  # n0 component not available
                    
            # Select radial range
            if ref_data.ndim == 2:
                ref_data = ref_data[:, self.xind]
                if ref_data_n0 is not None:
                    ref_data_n0 = ref_data_n0[:, self.xind]
            else:
                # Handle 1D data (like radial profiles)
                ref_data = ref_data[self.xind] if len(ref_data) > len(self.xind) else ref_data
            
            if apply_fft_filter and ref_data.ndim == 2:
                # Apply ballooning coordinate transformation and FFT filtering
                databal = mylib.interp2_bal(self.bal_ang, ref_data)
                data_fft = np.fft.fft(databal, axis=0)
                
                # Remove low-frequency modes
                for j in range(0, fft_filter_modes):
                    data_fft[j, :] = 0
                    data_fft[-j, :] = 0
                    
                pdata_bal = np.real(np.fft.ifft(data_fft, axis=0))  # Use real part for reference
                self.reference_data = mylib.inv_interp2_bal(self.bal_ang, pdata_bal)
            else:
                # Use data as is or subtract background if available
                if ref_data_n0 is not None and ref_data.ndim == 2:
                    self.reference_data = ref_data - ref_data_n0
                else:
                    self.reference_data = ref_data
                    
            print(f"Reference data loaded successfully. Shape: {self.reference_data.shape}")
            
        except Exception as e:
            print(f"Error loading reference data from time step {ref_time}: {e}")
            self.reference_data = None
            
    def process_timestep(self, itime: int, apply_fft_filter: bool = True, 
                        fft_filter_modes: int = 1, subtract_reference: bool = False) -> Tuple[np.ndarray, float]:
        """Process a single timestep using GYS_dataprocess module"""
        try:
            # Use the unified data processing function
            data = process_data(self.dtype, self.dirname, itime, self.spnum, mylib)
            tdiag = mylib.read_data(self.dirname, 'time_diag', t1=itime, spnum=self.spnum)
            
            # Handle special case for Phi data with n0 component
            data_n0 = None
            if self.dtype in ['Phirth', 'phi']:
                try:
                    data_n0 = mylib.read_data(self.dirname, 'Phirth_n0', t1=itime, spnum=self.spnum)
                except:
                    pass  # n0 component not available
                   
            # Average over toroidal angle if 3D                    
            if data.ndim == 3:
                data = np.mean(data, axis=0)  
                
            # Select radial range
            if data.ndim == 2:
                data = data[:, self.xind]
                if data_n0 is not None:
                    data_n0 = data_n0[:, self.xind]
            else:
                # Handle 1D data (like radial profiles)
                data = data[self.xind] if len(data) > len(self.xind) else data
            
            # Apply reference subtraction before other processing if requested
            if subtract_reference and self.reference_data is not None:
                if data.ndim == 2 and self.reference_data.ndim == 2:
                    data = data - self.reference_data
                elif data.ndim == 1 and self.reference_data.ndim == 1:
                    data = data - self.reference_data
                else:
                    print(f"Warning: Dimension mismatch between data ({data.ndim}D) and reference ({self.reference_data.ndim}D)")
            
            if apply_fft_filter and data.ndim == 2:
                # Apply ballooning coordinate transformation and FFT filtering
                databal = mylib.interp2_bal(self.bal_ang, data)
                data_fft = np.fft.fft(databal, axis=0)
                
                # Remove low-frequency modes
                for j in range(0, fft_filter_modes):
                    data_fft[j, :] = 0
                    data_fft[-j, :] = 0
                    
                pdata_bal = np.fft.ifft(data_fft, axis=0)
                pdata = np.abs(mylib.inv_interp2_bal(self.bal_ang, pdata_bal))
            else:
                # Use data as is or subtract background if available (and not using reference subtraction)
                if data_n0 is not None and data.ndim == 2 and not subtract_reference:
                    pdata = np.abs(data - data_n0)
                else:
                    pdata = data  # Use data directly without taking absolute value for signed data
            
            # Average over selected angle range
            if data.ndim == 2 and len(self.angle_indices) > 0:
                anal_data = pdata[self.angle_indices, :]
                profile = np.mean(anal_data, axis=0)
            elif data.ndim == 1:
                # For 1D data, just return as is
                profile = pdata
            else:
                profile = np.full(len(self.rg), np.nan)
                
            return profile, tdiag[0]
            
        except Exception as e:
            print(f"Error processing timestep {itime} for {self.dtype}: {e}")
            return np.full(len(self.rg), np.nan), np.nan
            
    def analyze_time_evolution(self, n1: int, n: int, dn: int, 
                             apply_fft_filter: bool = True,
                             fft_filter_modes: int = 1,
                             reference_time: Optional[int] = None,
                             time_derivative: bool = False,
                             rad_derivative: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analyze time evolution and create contour data"""
        
        # Load reference data if specified
        if reference_time is not None:
            self.load_reference_data(reference_time, apply_fft_filter, fft_filter_modes)
            subtract_reference = True
        else:
            subtract_reference = False
        
        print(f"Processing {n} time steps starting from {n1} with increment {dn}...")
        print(f"Data type: {self.dtype}")
        print(f"Radial range: [{self.min_r:.2f}, {self.max_r:.2f}]")
        print(f"Angle range: [{self.min_angle_deg:.1f}, {self.max_angle_deg:.1f}] degrees")
        print(f"Found {len(self.angle_indices)} angle points in the specified range.")
        print(f"FFT filtering: {'ON' if apply_fft_filter else 'OFF'}")
        print(f"Reference subtraction: {'ON (t=' + str(reference_time) + ')' if reference_time is not None else 'OFF'}")
        
        # Initialize arrays
        Data2D = np.zeros([len(self.rg), n])
        time_arr = np.zeros(n)
        
        # Process each timestep
        for ni in tqdm(range(n), desc="Processing time steps", unit="step"):
            itime = n1 + ni * dn
            profile, time_val = self.process_timestep(itime, apply_fft_filter, fft_filter_modes, subtract_reference)
            Data2D[:, ni] = profile
            time_arr[ni] = time_val
        
        if time_derivative:
            print("Calculating time derivative of the data...")
            from scipy.ndimage import uniform_filter1d
            interval = 1
            smoothed = uniform_filter1d(Data2D, size=interval, axis=1, mode='nearest')
            Data2D = np.gradient(smoothed, time_arr, axis=1)
            
        if rad_derivative:
            print("Calculating radial derivative of the data...")
            Data2D = np.gradient(Data2D, self.rg, axis=0)
        
        return Data2D, time_arr, self.rg
        
    def plot_contour(self, Data2D: np.ndarray, time_arr: np.ndarray, rg: np.ndarray,
                    colormap: str = 'hot', cb_option: str = 'percentile',
                    cb_low: float = 0.0, cb_up: float = 1.0,
                    percentile_low: float = 1.0, percentile_high: float = 99.0,
                    save_path: Optional[str] = None, show_plot: bool = True):
        """Create contour plot with configurable percentile limits"""
        
        fig, ax1 = mylib.init_plot_params(figsize=(5,8), font_size=16)
        
        # Create meshgrid
        X, Y = np.meshgrid(rg, time_arr)
        
        # Calculate colorbar limits using percentiles
        valid_data = Data2D[~np.isnan(Data2D)]
        if len(valid_data) > 0:
            if cb_option == 'fixed':
                cbar_lim_lower = cb_low
                cbar_lim_upper = cb_up
                print(f"Using fixed colorbar limits: [{cbar_lim_lower:.6e}, {cbar_lim_upper:.6e}]")
            elif cb_option == 'percentile':
                if percentile_low < 0:
                    cbar_lim_lower = -np.percentile(np.abs(valid_data), -percentile_low)
                else:
                    cbar_lim_lower = np.percentile(np.abs(valid_data), percentile_low)
                cbar_lim_upper = np.percentile(np.abs(valid_data), percentile_high)
                print(f"Colorbar limits: [{cbar_lim_lower:.6e}, {cbar_lim_upper:.6e}] "
                        f"(percentiles: {percentile_low}%, {percentile_high}%)")
        else:
            cbar_lim_lower = 0.0
            cbar_lim_upper = 1.0
            print("Warning: No valid data found. Using default colorbar limits.")
        
        # cbar_lim_lower = 0 #-0.00015
        # cbar_lim_upper = 0.015
        if np.any(~np.isnan(Data2D)):
            # Create contour plot with proper percentile limits
            f1 = ax1.pcolormesh(X, Y, np.transpose(Data2D), cmap=colormap, shading='gouraud',
                               vmin=cbar_lim_lower, vmax=cbar_lim_upper)
            
            cbar = plt.colorbar(f1, ax=ax1)
            cbar.ax.tick_params(labelsize=14)
            # cbar.set_label(f'{self.dtype}', size=16)
        else:
            ax1.text(0.5, 0.5, 'No valid data to plot', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax1.transAxes)
        
        # Format axes
        ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1.yaxis.get_offset_text().set_size(14)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        
        ax1.set_xlabel(r'$r/a$', size=20)
        ax1.set_ylabel(r'Time [$\omega_c^{-1}$]', size=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close()


def main(fn: str, dtype: str = 'Phirth', n1: int = 0, n: int = 1, dn: int = 1,
         spnum: int = 0, min_r: float = 0.7, max_r: float = 1.1,
         min_angle_deg: float = -30.0, max_angle_deg: float = 30.0,
         apply_fft_filter: bool = True, fft_filter_modes: int = 1,
         reference_time: Optional[int] = None,
         time_derivative: bool = False,
         rad_derivative: bool = False,
         colormap: str = 'hot', cb_option: str = 'percentile',
         cb_low: float = 0.0, cb_up: float = 1.0,
         percentile_low: float = 1.0, percentile_high: float = 99.0,
         save_path: Optional[str] = None):
    """Main analysis function"""
    
    dirname = os.path.dirname(fn) if os.path.dirname(fn) else '.'
    
    # Initialize analyzer
    analyzer = ContourAnalyzer(dirname, dtype, spnum, min_r, max_r, min_angle_deg, max_angle_deg)
    
    # Analyze time evolution
    Data2D, time_arr, rg = analyzer.analyze_time_evolution(
        n1, n, dn, apply_fft_filter, fft_filter_modes, reference_time,
        time_derivative, rad_derivative
    )
    
    # Create plot with specified percentile limits
    analyzer.plot_contour(Data2D, time_arr, rg, colormap, cb_option,
                          cb_low, cb_up,
                          percentile_low, percentile_high, save_path)
    
    return Data2D, time_arr, rg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generalized Plasma Data Contour Analyzer')
    
    # Basic parameters
    parser.add_argument('fn', type=str, help='Data directory or file path')
    parser.add_argument('-d', '--dtype', type=str, default='Phirth', 
                       help='Data type to analyze (default: Phirth)')
    parser.add_argument('-n1', '--start_time', type=int, default=0, 
                       help='First time step (default: 0)')
    parser.add_argument('-n', '--num_steps', type=int, default=1, 
                       help='Number of time steps (default: 1)')
    parser.add_argument('-dn', '--time_increment', type=int, default=1, 
                       help='Time step increment (default: 1)')
    parser.add_argument('-s', '--spnum', type=int, default=0, 
                       help='Species number (default: 0)')
    
    # Spatial parameters
    parser.add_argument('--min_r', type=float, default=0.7, 
                       help='Minimum radial coordinate (default: 0.7)')
    parser.add_argument('--max_r', type=float, default=1.1, 
                       help='Maximum radial coordinate (default: 1.1)')
    parser.add_argument('--min_angle', type=float, default=-30.0, 
                       help='Minimum angle in degrees (default: -30.0)')
    parser.add_argument('--max_angle', type=float, default=30.0, 
                       help='Maximum angle in degrees (default: 30.0)')
    
    # Processing parameters
    parser.add_argument('--no_fft_filter', action='store_true', 
                       help='Disable FFT filtering')
    parser.add_argument('--fft_modes', type=int, default=1, 
                       help='Number of low-frequency modes to filter (default: 1)')
    parser.add_argument('--ref_time', type=int, default=None, 
                       help='Reference time step for subtraction (default: None)')
    parser.add_argument('--time_derivative', action='store_true',
                       help='Compute time derivative of the data')
    parser.add_argument('--rad_derivative', action='store_true',
                       help='Compute radial derivative of the data')
    
    # Visualization parameters
    parser.add_argument('--colormap', type=str, default='hot', 
                       help='Colormap for contour plot (default: hot)')
    parser.add_argument('--cb_option', type=str, default='percentile', choices=['fixed', 'percentile'],
                       help='Colorbar limit option: fixed or percentile (default: percentile)')
    parser.add_argument('--cb_low', type=float, default=0.0, 
                       help='Lower colorbar limit if fixed (default: 0.0)')
    parser.add_argument('--cb_up', type=float, default=1.0, 
                       help='Upper colorbar limit if fixed (default: 1.0)')
    parser.add_argument('--percentile_low', type=float, default=1.0, 
                       help='Lower percentile for colorbar limit (default: 1.0)')
    parser.add_argument('--percentile_high', type=float, default=99.0, 
                       help='Upper percentile for colorbar limit (default: 99.0)')
    parser.add_argument('--save', type=str, default=None, 
                       help='Save plot to specified path')
    parser.add_argument('--no_show', action='store_true', 
                       help='Do not show plot (useful when saving)')
    
    args = parser.parse_args()
    
    # Validate percentile arguments
    # if not (0 <= args.percentile_low < args.percentile_high <= 100):
        # raise ValueError("Percentiles must satisfy: 0 <= percentile_low < percentile_high <= 100")
    
    # Run analysis
    main(args.fn, args.dtype, args.start_time, args.num_steps, args.time_increment,
         args.spnum, args.min_r, args.max_r, args.min_angle, args.max_angle,
         not args.no_fft_filter, args.fft_modes, args.ref_time, args.time_derivative,
         args.rad_derivative, args.colormap, args.cb_option,
         args.cb_low, args.cb_up,
         args.percentile_low, args.percentile_high, args.save)