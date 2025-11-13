import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import mylib
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.ticker as ticker

# Import the data processing module
try:
    from GYS_dataprocess import process_data
except ImportError:
    print("Warning: GYS_dataprocess module not found. Using basic data loading only.")
    
    def process_data(dtype: str, dirname: str, t1: int, spnum: int, mylib):
        """Fallback function when GYS_dataprocess is not available"""
        return mylib.read_data(dirname, dtype, t1=t1, spnum=spnum)


class MultiVariableAnalyzer:
    """Class to analyze multiple variables' time series"""
    
    def __init__(self, dirname: str, spnum: int = 0, 
                 min_r: float = 0.7, max_r: float = 1.1,
                 min_angle_deg: float = -30.0, max_angle_deg: float = 30.0):
        self.dirname = dirname
        self.spnum = spnum
        self.min_r = min_r
        self.max_r = max_r
        self.min_angle_deg = min_angle_deg
        self.max_angle_deg = max_angle_deg
        
        # Load common geometry data
        self._load_common_data()
        
    def _load_common_data(self):
        """Load geometry and grid data"""
        # Use mylib function to load normalized grid
        grid_data = mylib.load_normalized_grid(
            self.dirname,
            spnum=self.spnum,
            r_min=self.min_r,
            r_max=self.max_r,
            return_mask=False  # Get indices, not boolean mask
        )

        self.rg = grid_data['rg']
        self.R0 = grid_data['R0']
        self.inv_rho = grid_data['rhostar']
        self.xind = grid_data['mask']

        # Load ballooning angle data
        self.bal_ang = mylib.esti_bal_angle(self.dirname)
        self.bal_ang = self.bal_ang[:, self.xind]

        # Setup angle analysis
        self._setup_angle_analysis()
        
    def _setup_angle_analysis(self):
        """Setup angle indices for analysis using mylib function"""
        angles_rad = self.bal_ang[:, 0]

        # Use mylib function for angle selection with wrap-around handling
        try:
            self.angle_indices = mylib.get_angle_indices(
                angles_rad,
                self.min_angle_deg,
                self.max_angle_deg,
                in_degrees=True
            )
        except ValueError as e:
            print(f"Warning: {e}")
            self.angle_indices = np.array([], dtype=int)
    
    def process_variable_timestep(self, dtype: str, itime: int, apply_fft: bool = True, 
                                 fft_modes: int = 1) -> np.ndarray:
        """Process a single timestep for a specific variable"""
        try:
            # Get data using the unified processing function
            data = process_data(dtype, self.dirname, itime, self.spnum, mylib)
            
            # Handle special processing for wExB
            if dtype == 'wExB':
                psi = mylib.read_data(self.dirname, 'psi')
                B0 = mylib.read_data(self.dirname, 'B')[:, :] 
                Btheta = mylib.read_data(self.dirname, 'Btheta')[:, :]
                phi0, _ = mylib.read_data(self.dirname, 'Phi00', 'time_diag', t1=itime, spnum=self.spnum)
                vExB = np.gradient(phi0[:], psi)
                data = np.squeeze(np.gradient(vExB, psi) * (Btheta)**2 / B0)
            
            # Handle Phi data with n0 component
            data_n0 = None
            if dtype in ['Phirth', 'phi']:
                try:
                    data_n0 = mylib.read_data(self.dirname, 'Phirth_n0', t1=itime, spnum=self.spnum)
                except:
                    pass
                    
            # Select radial range
            if data.ndim == 2:
                data = data[:, self.xind]
                if data_n0 is not None:
                    data_n0 = data_n0[:, self.xind]
            else:
                data = data[self.xind] if len(data) > len(self.xind) else data
            
            # Apply FFT filtering if requested and data is 2D
            if apply_fft and data.ndim == 2:
                databal = mylib.interp2_bal(self.bal_ang, data)
                data_fft = np.fft.fft(databal, axis=0)
                
                for j in range(0, max(1, fft_modes)):
                    data_fft[j, :] = 0
                    data_fft[-j, :] = 0
                    
                pdata_bal = np.abs(np.fft.ifft(data_fft, axis=0))
                pdata = mylib.inv_interp2_bal(self.bal_ang, pdata_bal)
            else:
                if data_n0 is not None and data.ndim == 2:
                    pdata = np.abs(data - data_n0)
                else:
                    pdata = np.abs(data)
            
            return pdata
            
        except Exception as e:
            print(f"Error processing {dtype} at timestep {itime}: {e}")
            return np.full(len(self.rg), np.nan)
    
    def extract_radial_profile(self, data: np.ndarray, r_center: float, r_width: float) -> float:
        """Extract radial average from 2D data"""
        if data.ndim == 1:
            return np.nanmean(data)
            
        # Find radial indices for averaging
        r_min = r_center - r_width
        r_max = r_center + r_width
        r_indices = np.where((self.rg >= r_min) & (self.rg <= r_max))[0]
        
        if len(r_indices) == 0:
            return np.nan
            
        # Average over angle range first, then radial range
        if len(self.angle_indices) > 0:
            angle_avg = np.nanmean(data[self.angle_indices, :], axis=0)
            radial_avg = np.nanmean(angle_avg[r_indices])
        else:
            radial_avg = np.nanmean(data[:, r_indices])
            
        return radial_avg
    
    def analyze_multiple_variables(self, var_list: List[str], var_settings: Dict[str, Dict], 
                                  reference_times: Dict[str, int], n1: int, n: int, dn: int,
                                  r_center: float = 0.9, r_width: float = 0.05) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Analyze time series for multiple variables"""
        
        print(f"Analyzing {len(var_list)} variables: {var_list}")
        print(f"Time range: {n1} to {n1 + (n-1)*dn} (step: {dn})")
        print(f"Radial range for extraction: {r_center:.3f} ± {r_width:.3f}")
        
        # Print variable-specific settings
        print("\nVariable settings:")
        for var in var_list:
            settings = var_settings.get(var, {'fft': True, 'modes': 1})
            ref_info = f", ref_t={reference_times[var]}" if var in reference_times else ""
            print(f"  {var}: FFT={'ON' if settings['fft'] else 'OFF'}, modes={settings['modes']}{ref_info}")
        print()
        
        # Load reference data if specified
        reference_data = {}
        for var in var_list:
            if var in reference_times:
                ref_time = reference_times[var]
                settings = var_settings.get(var, {'fft': True, 'modes': 1})
                print(f"Loading reference data for {var} at t={ref_time}...")
                ref_data = self.process_variable_timestep(var, ref_time, settings['fft'], settings['modes'])
                reference_data[var] = ref_data
        
        # Initialize arrays
        time_arr = np.zeros(n)
        var_data = {var: np.zeros(n) for var in var_list}
        
        # Process each timestep
        for ni in tqdm(range(n), desc="Processing time steps", unit="step"):
            itime = n1 + ni * dn
            
            # Get time value
            try:
                tdiag = mylib.read_data(self.dirname, 'time_diag', t1=itime, spnum=self.spnum)
                time_arr[ni] = tdiag[0]
            except:
                time_arr[ni] = itime
            
            # Process each variable
            for var in var_list:
                settings = var_settings.get(var, {'fft': True, 'modes': 1})
                data = self.process_variable_timestep(var, itime, settings['fft'], settings['modes'])
                
                # Apply reference subtraction if specified
                if var in reference_data:
                    if data.shape == reference_data[var].shape:
                        data = data - reference_data[var]
                    else:
                        print(f"Warning: Shape mismatch for {var} reference subtraction")
                
                # Extract radial profile
                profile_value = self.extract_radial_profile(data, r_center, r_width)
                var_data[var][ni] = profile_value
        
        return time_arr, var_data
    
    def plot_multiple_variables(self, time_arr: np.ndarray, var_data: Dict[str, np.ndarray],
                               var_list: List[str], r_center: float, r_width: float,
                               normalize_vars: bool = False, dual_y_axis: bool = False,
                               save_path: Optional[str] = None, plot_title: Optional[str] = None):
        """Plot multiple variables"""
        
        fig, ax1 = mylib.init_plot_params(font_size=16)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        if dual_y_axis and len(var_list) == 2:
            # Use dual y-axis for two variables
            var1, var2 = var_list[0], var_list[1]
            
            # Plot first variable on left y-axis
            line1 = ax1.plot(time_arr, var_data[var1], color=colors[0], linewidth=2, label=var1)
            ax1.set_xlabel(r'Time [$\omega_c^{-1}$]', size=20)
            ax1.set_ylabel(var1, size=18, color=colors[0])
            ax1.tick_params(axis='y', labelcolor=colors[0])
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
            
            # Create second y-axis
            ax2 = ax1.twinx()
            line2 = ax2.plot(time_arr, var_data[var2], color=colors[1], linewidth=2, label=var2)
            ax2.set_ylabel(var2, size=18, color=colors[1])
            ax2.tick_params(axis='y', labelcolor=colors[1])
            ax2.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', fontsize=14)
            
        else:
            # Single y-axis for all variables
            for i, var in enumerate(var_list):
                data = var_data[var]
                
                # Normalize if requested
                if normalize_vars:
                    data_max = np.nanmax(np.abs(data))
                    if data_max > 0:
                        data = data / data_max
                        label = f'{var} (normalized)'
                    else:
                        label = f'{var}'
                else:
                    label = var
                
                color = colors[i % len(colors)]
                ax1.plot(time_arr, data, color=color, linewidth=2, label=label)
            
            ax1.set_xlabel(r'Time [$\omega_c^{-1}$]', size=20)
            ylabel = 'Normalized Values' if normalize_vars else 'Variable Values'
            ax1.set_ylabel(ylabel, size=18)
            ax1.legend(fontsize=14)
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        
        if plot_title:
            title = plot_title
        else:
            title = f'Multi-Variable Time Series (r = {r_center:.2f} ± {r_width:.2f})'
        
        ax1.set_title(title, size=18)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Multi-variable plot saved as {save_path}")
            
        plt.show()


def parse_var_settings(settings_str: str) -> Dict[str, Dict]:
    """Parse variable settings string"""
    var_settings = {}
    
    if not settings_str:
        return var_settings
    
    try:
        for var_setting in settings_str.split(';'):
            var_setting = var_setting.strip()
            if not var_setting:
                continue
                
            parts = var_setting.split(':')
            if len(parts) != 2:
                continue
                
            var_name = parts[0].strip()
            settings_part = parts[1].strip()
            
            var_config = {'fft': True, 'modes': 1}  # defaults
            
            for setting in settings_part.split(','):
                setting = setting.strip()
                if '=' in setting:
                    key, value = setting.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'fft':
                        var_config['fft'] = bool(int(value))
                    elif key == 'modes':
                        var_config['modes'] = int(value)
            
            var_settings[var_name] = var_config
            
    except Exception as e:
        print(f"Error parsing variable settings: {e}")
        return {}
    
    return var_settings


def parse_ref_times(ref_times_str: str) -> Dict[str, int]:
    """Parse reference times string"""
    reference_times = {}
    
    if not ref_times_str:
        return reference_times
    
    try:
        for item in ref_times_str.split(','):
            item = item.strip()
            if ':' in item:
                var, time_str = item.split(':', 1)
                reference_times[var.strip()] = int(time_str.strip())
    except Exception as e:
        print(f"Error parsing reference times: {e}")
        return {}
    
    return reference_times


def main():
    parser = argparse.ArgumentParser(description='Multi-Variable Time Series Analyzer')
    
    # Required arguments
    parser.add_argument('fn', type=str, help='Data directory or file path')
    parser.add_argument('-v', '--variables', type=str, required=True,
                       help='Comma-separated list of variables (e.g., "Phirth,wExB")')
    
    # Time parameters
    parser.add_argument('-n1', type=int, default=0, help='First time step')
    parser.add_argument('-n', type=int, default=1, help='Number of time steps')
    parser.add_argument('-dn', type=int, default=1, help='Time step increment')
    parser.add_argument('-s', type=int, default=0, help='Species number')
    
    # Spatial parameters
    parser.add_argument('--min_r', type=float, default=0.7, help='Minimum radial coordinate')
    parser.add_argument('--max_r', type=float, default=1.1, help='Maximum radial coordinate')
    parser.add_argument('--min_angle', type=float, default=-30.0, help='Minimum angle in degrees')
    parser.add_argument('--max_angle', type=float, default=30.0, help='Maximum angle in degrees')
    parser.add_argument('--r_center', type=float, default=0.9, help='Center radius for analysis')
    parser.add_argument('--r_width', type=float, default=0.05, help='Radial width for averaging')
    
    # Processing parameters
    parser.add_argument('--var_settings', type=str, default="",
                       help='Variable settings: "var1:fft=1,modes=3;var2:fft=0"')
    parser.add_argument('--ref_times', type=str, default="",
                       help='Reference times: "var1:time1,var2:time2"')
    
    # Visualization parameters
    parser.add_argument('--normalize', action='store_true', help='Normalize variables')
    parser.add_argument('--dual_y', action='store_true', help='Use dual y-axis (2 variables only)')
    parser.add_argument('--save', type=str, default=None, help='Save path')
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    
    args = parser.parse_args()
    
    # Parse inputs
    var_list = [v.strip() for v in args.variables.split(',')]
    var_settings = parse_var_settings(args.var_settings)
    reference_times = parse_ref_times(args.ref_times)
    
    print(f"Variables: {var_list}")
    print(f"Settings: {var_settings}")
    print(f"Reference times: {reference_times}")
    
    # Create analyzer
    dirname = os.path.dirname(args.fn) if os.path.dirname(args.fn) else '.'
    analyzer = MultiVariableAnalyzer(dirname, args.s, args.min_r, args.max_r, 
                                   args.min_angle, args.max_angle)
    
    # Analyze
    time_arr, var_data = analyzer.analyze_multiple_variables(
        var_list, var_settings, reference_times, args.n1, args.n, args.dn,
        args.r_center, args.r_width
    )
    
    # Plot
    analyzer.plot_multiple_variables(
        time_arr, var_data, var_list, args.r_center, args.r_width,
        args.normalize, args.dual_y, args.save, args.title
    )


if __name__ == "__main__":
    main()
    
# Example usage:
# python comp_data_time.py /data/sim1 -v "Phirth,wExB,Pres2D" \
#   --var_settings "Phirth:fft=1,modes=3;wExB:fft=0,modes=1;Pres2D:fft=1,modes=5" \
#   --ref_times "Phirth:0,wExB:50" \
#   --dual_y --save comparison.png
  