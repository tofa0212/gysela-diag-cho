import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import mylib
from typing import Optional, List, Tuple


class RadialProfileAnalyzer:
    """Analyzer for radial profiles of various quantities"""
    
    def __init__(self, dirname: str, ref_dirname: Optional[str] = None,
                 r_min: float = 0.0, r_max: float = 1.0):
        """
        Initialize analyzer
        
        Args:
            dirname: Directory containing simulation data
            ref_dirname: Reference directory for baseline comparison
            r_min: Minimum normalized radius
            r_max: Maximum normalized radius
        """
        self.dirname = os.path.abspath(dirname)
        self.ref_dirname = os.path.abspath(ref_dirname) if ref_dirname else None
        self.r_min = r_min
        self.r_max = r_max
        
        self._load_grid_data()
        
        if self.ref_dirname:
            print(f"Reference directory: {self.ref_dirname}")
    
    def _load_grid_data(self):
        """Load radial grid and normalization"""
        rg = mylib.read_data(self.dirname, 'rg')
        R0 = mylib.read_data(self.dirname, 'R0')
        rhostar = mylib.read_data(self.dirname, 'rhostar')
        
        self.R0 = R0 *rhostar
        self.rg = rg *rhostar
        
        # Apply additional radial mask
        self.xind = (self.rg >= self.r_min) & (self.rg <= self.r_max)
        self.rg = self.rg[self.xind]
        
        print(f"Grid loaded: {len(self.rg)} radial points")
    
    def calculate_temperature(self, timestep: int, use_ref_dir: bool = False) -> np.ndarray:
        """Calculate ion temperature at given timestep"""
        dirname = self.ref_dirname if (use_ref_dir and self.ref_dirname) else self.dirname
        P, n, _ = mylib.read_data(
            dirname, 'stress_FSavg', 'dens_FSavg', 'time_diag', t1=timestep
        )
        Ti = (P / n)[self.xind]
        return Ti
    
    def calculate_density(self, timestep: int, use_ref_dir: bool = False) -> np.ndarray:
        """Calculate density at given timestep"""
        dirname = self.ref_dirname if (use_ref_dir and self.ref_dirname) else self.dirname
        n, _ = mylib.read_data(dirname, 'dens_FSavg', 'time_diag', t1=timestep)
        return n[self.xind]
    
    def calculate_pressure(self, timestep: int, use_ref_dir: bool = False) -> np.ndarray:
        """Calculate pressure at given timestep"""
        dirname = self.ref_dirname if (use_ref_dir and self.ref_dirname) else self.dirname
        P, _ = mylib.read_data(dirname, 'stress_FSavg', 'time_diag', t1=timestep)
        return P[self.xind]
    
    def calculate_gradient_scale_length(self, timestep: int, 
                                       quantity: str = 'temperature',
                                       use_ref_dir: bool = False) -> np.ndarray:
        """
        Calculate inverse gradient scale length R/L_X
        
        Args:
            timestep: Time step
            quantity: 'temperature', 'density', or 'pressure'
            use_ref_dir: Use reference directory if True
        """
        if quantity == 'temperature':
            profile = self.calculate_temperature(timestep, use_ref_dir)
        elif quantity == 'density':
            profile = self.calculate_density(timestep, use_ref_dir)
        elif quantity == 'pressure':
            profile = self.calculate_pressure(timestep, use_ref_dir)
        else:
            raise ValueError(f"Unknown quantity: {quantity}")
        
        grad = np.gradient(profile, self.rg)
        R_over_L = -grad / profile * self.R0
        return R_over_L
    
    def get_profile_data(self, timesteps: List[int], 
                        data_type: str = 'temperature',
                        reference_time: Optional[int] = None,
                        compare_with_ref_dir: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get radial profile data for multiple timesteps
        
        Args:
            timesteps: List of timesteps
            data_type: Type of data ('temperature', 'density', 'pressure', 
                      'temperature_gradient', 'density_gradient', 'pressure_gradient')
            reference_time: Reference timestep for subtraction (optional)
            compare_with_ref_dir: If True and ref_dirname exists, subtract 
                                 ref_dir profile at same timestep
            
        Returns:
            (profile_2d, time_array, ref_profile)
        """
        n_times = len(timesteps)
        profile_2d = np.zeros([len(self.rg), n_times])
        time_array = np.zeros(n_times)
        
        # Calculate functions mapping
        calc_funcs = {
            'temperature': self.calculate_temperature,
            'density': self.calculate_density,
            'pressure': self.calculate_pressure,
            'temperature_gradient': lambda t, use_ref=False: self.calculate_gradient_scale_length(t, 'temperature', use_ref),
            'density_gradient': lambda t, use_ref=False: self.calculate_gradient_scale_length(t, 'density', use_ref),
            'pressure_gradient': lambda t, use_ref=False: self.calculate_gradient_scale_length(t, 'pressure', use_ref),
        }
        
        if data_type not in calc_funcs:
            raise ValueError(f"Unknown data_type: {data_type}. Available: {list(calc_funcs.keys())}")
        
        calc_func = calc_funcs[data_type]
        
        # Get reference profile from reference time (single reference for all)
        ref_profile = None
        if reference_time is not None:
            ref_profile = calc_func(reference_time)
        
        # Calculate profiles
        for i, timestep in enumerate(timesteps):
            try:
                profile = calc_func(timestep)
                tdiag = mylib.read_data(self.dirname, 'time_diag', t1=timestep)
                
                # If comparing with ref_dir, subtract ref_dir profile at same timestep
                if compare_with_ref_dir and self.ref_dirname:
                    ref_dir_profile = calc_func(timestep, use_ref_dir=True)
                    profile = profile - ref_dir_profile
                
                profile_2d[:, i] = profile
                time_array[i] = tdiag[0]
            except Exception as e:
                print(f"Warning: Failed to process timestep {timestep}: {e}")
                profile_2d[:, i] = np.nan
                time_array[i] = np.nan
        
        return profile_2d, time_array, ref_profile
    
    def plot_profiles(self, timesteps: List[int],
                     data_type: str = 'temperature',
                     reference_time: Optional[int] = None,
                     compare_with_ref_dir: bool = False,
                     normalize_edge: bool = False,
                     colors: Optional[List[str]] = None,
                     font_size: int = 12,
                     save_path: Optional[str] = None):
        """
        Plot radial profiles for multiple timesteps
        
        Args:
            timesteps: List of timesteps
            data_type: Type of data to plot
            reference_time: Reference timestep for subtraction (from same directory)
            compare_with_ref_dir: If True, subtract ref_dir profile at each timestep
            normalize_edge: If True, subtract edge value for each profile
            colors: List of colors (optional)
            font_size: Base font size
            save_path: Path to save figure
        """
        # Get data
        profile_2d, time_array, ref_profile = self.get_profile_data(
            timesteps, data_type, reference_time, compare_with_ref_dir
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))
        
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))
        
        # Plot each timestep
        for i, (timestep, time_val, color) in enumerate(zip(timesteps, time_array, colors)):
            profile = profile_2d[:, i].copy()
            
            # Apply reference subtraction (from reference_time in same directory)
            if ref_profile is not None and not compare_with_ref_dir:
                profile = profile - ref_profile
            
            # Apply edge normalization
            if normalize_edge:
                profile = profile - profile[0]
            
            # Skip if all NaN
            if np.all(np.isnan(profile)):
                continue
            
            label = f't={mylib.sci_note(time_val)}'
            ax.plot(self.rg, profile, label=label, color=color, linewidth=2)
        
        # Labels and formatting
        ylabel_map = {
            'temperature': r'$T_i$',
            'density': r'$n$',
            'pressure': r'$P$',
            'temperature_gradient': r'$R/L_{T_i}$',
            'density_gradient': r'$R/L_n$',
            'pressure_gradient': r'$R/L_P$',
        }
        
        ylabel = ylabel_map.get(data_type, data_type)
        
        # Modify ylabel based on operations
        if compare_with_ref_dir and self.ref_dirname:
            ylabel = f'Δ{ylabel} (vs ref_dir)'
        elif reference_time is not None:
            ref_time_val = mylib.read_data(self.dirname, 'time_diag', t1=reference_time)[0]
            ylabel = f'{ylabel} - {ylabel}(t={mylib.sci_note(ref_time_val)})'
        
        if normalize_edge:
            ylabel = f'{ylabel} (edge-normalized)'
        
        ax.set_xlabel('r/a', size=font_size+2)
        ax.set_ylabel(f'$\Delta T_i$', size=font_size+2)
        # ax.set_ylabel(ylabel, size=font_size+2)
        ax.set_xlim(self.r_min, self.r_max)
        ax.tick_params(labelsize=font_size)
        ax.legend(fontsize=font_size-2, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at zero if comparing
        if compare_with_ref_dir or reference_time is not None:
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        title = f'Radial Profile: {data_type.replace("_", " ").title()}'
        # ax.set_title(title, size=font_size+4)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot radial profiles of various quantities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic parameters
    parser.add_argument('fn', type=str, help='Data directory path')
    
    # Reference directory
    parser.add_argument('--ref_dir', type=str, default=None,
                       help='Reference directory for baseline comparison')
    parser.add_argument('--compare_ref_dir', action='store_true',
                       help='Subtract ref_dir profile at each timestep (requires --ref_dir)')
    
    # Time parameters
    parser.add_argument('-t', '--timesteps', nargs='+', type=int, default=None,
                       help='List of timesteps to plot')
    parser.add_argument('-n1', '--start_time', type=int, default=0,
                       help='Starting timestep')
    parser.add_argument('-n', '--num_steps', type=int, default=1,
                       help='Number of timesteps')
    parser.add_argument('-dn', '--time_increment', type=int, default=10,
                       help='Time step increment')
    parser.add_argument('-nf', '--ref_time', type=int, default=None,
                       help='Reference timestep for subtraction (from same directory)')
    
    # Data type
    parser.add_argument('-d', '--data_type', type=str, default='temperature',
                       choices=['temperature', 'density', 'pressure',
                               'temperature_gradient', 'density_gradient', 'pressure_gradient'],
                       help='Type of data to plot')
    
    # Spatial parameters
    parser.add_argument('--r_min', type=float, default=0.0,
                       help='Minimum normalized radius')
    parser.add_argument('--r_max', type=float, default=1.0,
                       help='Maximum normalized radius')
    
    # Plot options
    parser.add_argument('--normalize_edge', action='store_true',
                       help='Subtract edge value from each profile')
    parser.add_argument('--font_size', type=int, default=12,
                       help='Base font size')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plot to specified path')
    
    args = parser.parse_args()
    
    # Generate timesteps
    if args.timesteps is None:
        timesteps = list(range(args.start_time, 
                              args.start_time + args.num_steps * args.time_increment,
                              args.time_increment))
    else:
        timesteps = args.timesteps
    
    # Validate ref_dir usage
    if args.compare_ref_dir and not args.ref_dir:
        parser.error("--compare_ref_dir requires --ref_dir")
    
    # Set reference time (only used if not comparing with ref_dir)
    # if args.ref_time is None and not args.compare_ref_dir:
    #     args.ref_time = timesteps[0]
    
    print(f"Plotting {args.data_type} profiles")
    print(f"Timesteps: {timesteps}")
    if args.compare_ref_dir:
        print(f"Comparing with reference directory at each timestep")
    elif args.ref_time is not None:
        print(f"Reference time (same directory): {args.ref_time}")
    
    # Initialize analyzer
    analyzer = RadialProfileAnalyzer(
        args.fn,
        ref_dirname=args.ref_dir,
        r_min=args.r_min,
        r_max=args.r_max
    )
    
    # Plot
    analyzer.plot_profiles(
        timesteps,
        data_type=args.data_type,
        reference_time=args.ref_time if not args.compare_ref_dir else None,
        compare_with_ref_dir=args.compare_ref_dir,
        normalize_edge=args.normalize_edge,
        font_size=args.font_size,
        save_path=args.save
    )