import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import mylib
from typing import List, Tuple, Optional


class TemperatureProfileAnalyzer:
    """Analyzer for ion temperature profiles and related quantities"""
    
    def __init__(self, dirname: str, ref_dirname: Optional[str] = None,
                 r_min: float = 0.0, r_max: float = 1.8,
                 min_angle: Optional[float] = None, 
                 max_angle: Optional[float] = None,
                 ref_time: int = 1501):
        """
        Initialize analyzer
        
        Args:
            dirname: Directory containing simulation data
            ref_dirname: Reference directory for baseline comparison
            r_min: Minimum normalized radius
            r_max: Maximum normalized radius
        """
        self.dirname = os.path.abspath(dirname)
        self.ref_dirname = ref_dirname
        self.ref_time = ref_time
        self.r_min = r_min
        self.r_max = r_max        
        self.min_angle = min_angle
        self.max_angle = max_angle
        
        self._load_grid_data()
        
        if ref_dirname:
            self._load_reference_data()
    
    def _load_grid_data(self):
        """Load radial grid and normalization parameters"""
        self.rg = mylib.read_data(self.dirname, 'rg')
        self.rhostar = mylib.read_data(self.dirname, 'rhostar')
        self.rg *= self.rhostar
        
        # Apply radial mask
        self.xind = np.where((self.rg >= self.r_min) & (self.rg <= self.r_max))[0]
        self.rg = self.rg[self.xind]
        
        # Normalization
        self.R0, self.Ts0 = mylib.read_data(self.dirname, 'R0', 'Ts0')
        self.R0 *= self.rhostar
        
        print(f"Grid loaded: {len(self.rg)} radial points")

        # Load ballooning angle data
        if self.min_angle is not None:
            self.bal_ang = mylib.esti_bal_angle(self.dirname)
            self.bal_ang = self.bal_ang[:, self.xind]

            # Setup angle analysis
            self._setup_angle_analysis()

        def _setup_angle_analysis(self):
            """Setup angle indices for analysis"""
            angles_rad = self.bal_ang[:, 0]  # Use angles from the first radial point

            # Convert degree range to radians
            min_angle_rad = np.deg2rad(self.min_angle)
            max_angle_rad = np.deg2rad(self.max_angle)

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

    
    def _load_reference_data(self):
        """Load reference temperature profile"""
        P_ref, n_ref, _ = mylib.read_data(
            self.ref_dirname, 'stress_FSavg', 'dens_FSavg', 'time_diag', t1=self.ref_time
        )
        self.T_ref = P_ref / n_ref
    
    def _load_2D_reference_data(self):
        """Load 2D reference temperature with anisotropy"""
        Ppar_2D, Pperp_2D, n_2D = mylib.read_data(
                self.ref_dirname, "PparGC_rtheta", "PperpGC_rtheta", "densGC_rtheta", t1=self.ref_time
        )
        self.Tpar_2D_ref  = Ppar_2D  / n_2D
        self.Tperp_2D_ref = Pperp_2D / n_2D
    
    def calc_temperature_profile(self, timesteps: List[int]) -> Tuple[np.ndarray, float]:
        """
        Calculate time-averaged ion temperature profile
        
        Args:
            timesteps: List of timesteps to average over
            
        Returns:
            (Ti_avg, tdiag_last): Averaged temperature and last diagnostic time
        """
        Ti_avg = None
        tdiag = None
    
        for timestep in timesteps:
            P, n, tdiag = mylib.read_data(
                self.dirname, 'stress_FSavg', 'dens_FSavg', 'time_diag', t1=timestep
            )
            Ti = P / n
            
            if Ti_avg is None:
                Ti_avg = Ti
            else:
                Ti_avg += Ti
        
        Ti_avg /= len(timesteps)
        Ti_avg = Ti_avg[self.xind]
        
        return Ti_avg, tdiag
    
    def calc_2D_temperature_profile(self, timesteps: List[int]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate time-averaged 2D ion temperature profiles, considering anisotropy

        Args:
            timesteps: List of timesteps to average over

        Return:
            (Ti_par_avg, Ti_perp_avg, tdiag_last): Averaged temperature and last diagnostic time
        """

        Ti_par_avg  = None
        Ti_perp_avg = None
        tdiag       = None

        for timestep in timesteps:
            Ppar, Pperp, n, tdiag = mylib.read_data(
                    self.dirname, "PparGC_rtheta", "PperpGC_rtheta", "densGC_rtheta", 
                    "time_diag", t1 = timestep
            )
            Ti_par  = Ppar  / n
            Ti_perp = Pperp / n

            if Ti_par_avg is None:
                Ti_par_avg = Ti_par
            else:
                Ti_par_avg += Ti_par

            if Ti_perp_avg is None:
                Ti_perp_avg = Ti_perp
            else:
                Ti_perp_avg += Ti_perp

        Ti_par_avg /= len(timesteps)
        Ti_par_avg = np.mean(Ti_par_avg[self.angle_indices, self.xind], axis = 0)
        
        Ti_perp_avg /= len(timesteps)
        Ti_perp_avg = np.mean(Ti_perp_avg[self.angle_indices, self.xind], axis = 0)

        return Ti_par_avg, Ti_perp_avg, tdiag 
        

    def calc_gradient_scale_length(self, Ti: np.ndarray) -> np.ndarray:
        """
        Calculate inverse gradient scale length R/L_T
        
        Args:
            Ti: Temperature profile
            
        Returns:
            R/L_T profile
        """
        grad_Ti = np.gradient(Ti, self.rg)
        RLT = -grad_Ti / Ti * self.R0
        return RLT
    
    def calc_heat_flux_profile(self, timesteps: List[int]) -> np.ndarray:
        """
        Calculate time-averaged heat flux profile
        
        Args:
            timesteps: List of timesteps to average over
            
        Returns:
            Averaged heat flux profile
        """
        Q_avg = None
        
        for timestep in timesteps:
            Q_par, Q_perp, _ = mylib.read_data(
                self.dirname, 'QGC_par_vE_rtheta', 'QGC_perp_vE_rtheta', 
                'time_diag', t1=timestep
            )
            Q_total = Q_par + Q_perp
            Q_theta_avg = np.mean(Q_total, axis=0)
            
            if Q_avg is None:
                Q_avg = Q_theta_avg
            else:
                Q_avg += Q_theta_avg
        
        Q_avg /= len(timesteps)
        Q_avg = Q_avg[self.xind]
        
        return Q_avg
    
    def calc_diffusivity(self, timesteps: List[int], epsilon: float = 1e-12) -> np.ndarray:
        """
        Calculate effective diffusivity χ = Q / (-∇T)
        
        Args:
            timesteps: List of timesteps to average over
            epsilon: Small value to prevent division by zero
            
        Returns:
            Diffusivity profile
        """
        Ti_avg, _ = self.calc_temperature_profile(timesteps)
        Q_avg = self.calc_heat_flux_profile(timesteps)
        
        grad_Ti = -np.gradient(Ti_avg, self.rg)
        grad_Ti_safe = np.where(np.abs(grad_Ti) < epsilon, epsilon, grad_Ti)
        
        chi = Q_avg / grad_Ti_safe
        return chi


class MultiCaseComparison:
    """Compare multiple simulation cases"""
    
    def __init__(self, dirnames: List[str], labels: List[str],
                 ref_dirname: Optional[str] = None,
                 r_min: float = 0.0, r_max: float = 1.8,
                 min_angle: Optional[float] = None,
                 max_angle: Optional[float] = None, 
                 ref_time: int = 1501):
        """
        Initialize multi-case comparison
        
        Args:
            dirnames: List of directories to compare
            labels: Labels for each case
            ref_dirname: Reference directory for baseline
            r_min: Minimum normalized radius
            r_max: Maximum normalized radius
            min_angle: Minimum angle
            max_angle: Maximum angle
        """
        self.analyzers = []
        self.labels = labels
        
        for dirname in dirnames:
            analyzer = TemperatureProfileAnalyzer(
                dirname, ref_dirname=ref_dirname, r_min=r_min, r_max=r_max,
                ref_time=ref_time
            )
            self.analyzers.append(analyzer)
        
        # Use first analyzer's grid
        self.rg = self.analyzers[0].rg
        self.has_reference = ref_dirname is not None
    
    def plot_temperature_profiles(self, timesteps: List[int],
                                  plot_type: str = 'absolute',
                                  colors: Optional[List[str]] = None,
                                  font_size: int = 12,
                                  save_path: Optional[str] = None):
        """
        Plot temperature profiles for all cases
        
        Args:
            timesteps: List of timesteps to average over
            plot_type: 'absolute', 'difference', or 'gradient'
            colors: List of colors for each case
            font_size: Base font size
            save_path: Path to save figure
        """
        if colors is None:
            colors = ['r', 'g', 'b', 'k', 'm', 'c', 'y']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for analyzer, label, color in zip(self.analyzers, self.labels, colors):
            Ti_avg, tdiag = analyzer.calc_temperature_profile(timesteps)
            
            if plot_type == 'absolute':
                y_data = Ti_avg
                ylabel = r'$T_i$'
                title = f'Ion Temperature Profile (t={tdiag[0]:.2e})'
            
            elif plot_type == 'difference':
                if not self.has_reference:
                    raise ValueError("Reference data required for 'difference' plot")
                T_ref_slice = analyzer.T_ref[analyzer.xind]
                y_data = Ti_avg - T_ref_slice
                ylabel = r'$\Delta T_i$'
                title = f'Temperature Difference (t={tdiag[0]:.2e})'
            
            elif plot_type == 'gradient':
                y_data = analyzer.calc_gradient_scale_length(Ti_avg)
                ylabel = r'$R/L_{T_i}$'
                title = f'Inverse Gradient Scale Length (t={tdiag[0]:.2e})'
            
            else:
                raise ValueError(f"Unknown plot_type: {plot_type}")
            
            ax.plot(self.rg, y_data, label=label, color=color, linewidth=2)
        
        # Add reference line for difference plot
        if plot_type == 'difference':
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('r/a', size=font_size+2)
        ax.set_ylabel(ylabel, size=font_size+2)
        # ax.set_title(title, size=font_size+4)
        ax.set_xlim(left=0, right=self.rg[-1])
        ax.set_ylim(-0.15, 0.4)
        ax.legend(fontsize=font_size-2, loc='upper left')
        ax.tick_params(labelsize=font_size)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_diffusivity(self, timesteps: List[int],
                        colors: Optional[List[str]] = None,
                        font_size: int = 12,
                        save_path: Optional[str] = None):
        """
        Plot effective diffusivity for all cases
        
        Args:
            timesteps: List of timesteps to average over
            colors: List of colors for each case
            font_size: Base font size
            save_path: Path to save figure
        """
        if colors is None:
            colors = ['r', 'g', 'b', 'k', 'm', 'c', 'y']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for analyzer, label, color in zip(self.analyzers, self.labels, colors):
            chi = analyzer.calc_diffusivity(timesteps)
            ax.plot(self.rg, chi, label=label, color=color, linewidth=2)
        
        ax.set_xlabel('r/a', size=font_size+2)
        ax.set_ylabel(r'$\chi$ [normalized]', size=font_size+2)
        ax.set_title('Effective Diffusivity', size=font_size+4)
        ax.set_xlim(left=0)
        ax.legend(fontsize=font_size-2)
        ax.tick_params(labelsize=font_size)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare ion temperature profiles across multiple cases',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic parameters
    parser.add_argument('directories', nargs='+', 
                       help='List of data directories to compare')
    parser.add_argument('-l', '--labels', nargs='+', default=None,
                       help='Labels for each case')
    
    # Time parameters
    parser.add_argument('-t', '--timesteps', nargs='+', type=int, default=None,
                       help='List of timesteps to average over')
    parser.add_argument('-n', '--start_time', type=int, default=1501,
                       help='Starting timestep')
    parser.add_argument('-dn', '--num_steps', type=int, default=1,
                       help='Number of timesteps to average')
    
    # Reference case
    parser.add_argument('--ref_dir', type=str, default=None,
                       help='Reference directory for baseline comparison')
    parser.add_argument('--ref_time', type=int, default=1501,
                        help='Reference timestep for baseline comparison')
    
    # Spatial parameters
    parser.add_argument('--r_min', type=float, default=0.0,
                       help='Minimum normalized radius')
    parser.add_argument('--r_max', type=float, default=1.8,
                       help='Maximum normalized radius')
    
    # Plot options
    parser.add_argument('--plot_type', type=str, default='difference',
                       choices=['absolute', 'difference', 'gradient'],
                       help='Type of temperature plot')
    parser.add_argument('--plot_diffusivity', action='store_true',
                       help='Also plot diffusivity')
    parser.add_argument('--colors', nargs='+', default=['r','g','b','k','m','c','y'],
                       help='Colors for each case')
    
    # Visualization parameters
    parser.add_argument('--font_size', type=int, default=12,
                       help='Base font size')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plots to specified directory')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    
    # Generate labels if not provided
    if args.labels is None:
        args.labels = [f'Case {i+1}' for i in range(len(args.directories))]
    
    if len(args.labels) != len(args.directories):
        raise ValueError("Number of labels must match number of directories")
    
    # Generate timesteps
    if args.timesteps is None:
        timesteps = list(range(args.start_time, args.start_time + args.num_steps))
    else:
        timesteps = args.timesteps
    
    print(f"Comparing {len(args.directories)} cases")
    print(f"Timesteps: {timesteps[0]} to {timesteps[-1]} ({len(timesteps)} steps)")
    
    # Initialize comparison
    comparison = MultiCaseComparison(
        args.directories,
        args.labels,
        ref_dirname=args.ref_dir,
        r_min=args.r_min,
        r_max=args.r_max,
        ref_time=args.ref_time
    )
    
    # Plot temperature profiles
    save_path = None
    if args.save:
        save_path = os.path.join(args.save, f'temperature_{args.plot_type}.png')
    
    comparison.plot_temperature_profiles(
        timesteps,
        plot_type=args.plot_type,
        colors=args.colors,
        font_size=args.font_size,
        save_path=save_path
    )
    
    # Plot diffusivity if requested
    if args.plot_diffusivity:
        save_path = None
        if args.save:
            save_path = os.path.join(args.save, 'diffusivity.png')
        
        comparison.plot_diffusivity(
            timesteps,
            colors=args.colors,
            font_size=args.font_size,
            save_path=save_path
        )
