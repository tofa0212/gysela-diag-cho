import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import mylib
from typing import Optional, Literal


class Reynolds_stress_analyzer:
    """Analyzer for Reynolds stress components in plasma turbulence"""
    
    def __init__(self, dirname: str, spnum: int = 0,
                 r_min: float = 0.0, r_max: float = 1.15):
        """
        Initialize analyzer
        
        Args:
            dirname: Directory containing simulation data
            spnum: Species number (0=ions, 1=electrons)
            r_min: Minimum normalized radius
            r_max: Maximum normalized radius
        """
        self.dirname = os.path.abspath(dirname)
        self.spnum = spnum
        self.r_min = r_min
        self.r_max = r_max
        
        self._load_common_data()
        
    def _load_common_data(self):
        """Load grid and geometry data"""
        # Use mylib function to load normalized grid
        grid_data = mylib.load_normalized_grid(
            self.dirname,
            spnum=self.spnum,
            r_min=self.r_min,
            r_max=self.r_max,
            return_mask=True
        )

        self.rg = grid_data['rg']
        self.R0 = grid_data['R0']
        self.rhostar = grid_data['rhostar']
        self.rmask = grid_data['mask']

        # Additional parameters
        self.Ts0 = mylib.read_data(self.dirname, 'Ts0', spnum=self.spnum)

        # Use mylib function to load geometry data
        geom_data = mylib.load_geometry_data(
            self.dirname,
            spnum=self.spnum,
            radial_mask=self.rmask,
            components=['thetag', 'B', 'jacob_space']
        )

        self.thg = geom_data['thetag']
        self.B = geom_data['B']
        self.jacob_space = geom_data['jacob_space']

        print(f"Grid loaded: {len(self.rg)} radial points, {len(self.thg)} poloidal points")
    
    def _get_angle_indices(self, min_angle: float, max_angle: float) -> np.ndarray:
        """
        Get angle indices for specified range

        Args:
            min_angle: Minimum angle in degrees
            max_angle: Maximum angle in degrees

        Returns:
            Array of indices
        """
        return mylib.get_angle_indices(self.thg, min_angle, max_angle,
                                       in_degrees=True)
    
    def _reduce_3d_to_2d(self, data_3d: np.ndarray, 
                        reduction: Literal['mean', 'rms'] = 'mean') -> np.ndarray:
        """
        Reduce 3D data (phi, theta, r) to 2D (theta, r) by averaging over phi
        
        Args:
            data_3d: 3D array [n_phi, n_theta, n_r]
            reduction: 'mean' or 'rms'
        
        Returns:
            2D array [n_theta, n_r]
        """
        if reduction == 'mean':
            return np.mean(data_3d, axis=0)
        elif reduction == 'rms':
            return np.sqrt(np.mean(data_3d**2, axis=0))
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def calc_RS_elec(self, timestep: int,
                    reduction: Literal['mean', 'rms'] = 'mean') -> np.ndarray:
        """
        Calculate electrostatic (ExB) component of Reynolds stress

        Args:
            timestep: Time step index
            reduction: How to reduce phi dimension ('mean' or 'rms')

        Returns:
            2D array [n_theta, n_r] of RS_elec
        """
        # Load 3D potential
        Phi3D = mylib.read_data(self.dirname, 'Phi_3D', t1=timestep-1501, spnum=self.spnum)
        Phi3D = Phi3D[:, :, self.rmask]  # Apply radial mask

        # Calculate ExB velocities using mylib function
        dr_vExB, dth_vExB = mylib.calc_exb_velocity_2d(
            Phi3D, self.rg, self.thg, self.B, self.jacob_space
        )

        # Calculate Reynolds stress using mylib function
        RS_3d = mylib.calc_reynolds_stress_from_velocities(
            dr_vExB, dth_vExB, self.rg, self.thg
        )

        # Reduce to 2D
        return self._reduce_3d_to_2d(RS_3d, reduction)
    
    def calc_RS_diag(self, timestep: int) -> np.ndarray:
        """
        Calculate diamagnetic component of Reynolds stress

        Args:
            timestep: Time step index

        Returns:
            2D array [n_theta, n_r] of RS_diag
        """
        # Load 2D pressure and density data
        P_parallel, P_perp = mylib.read_data(
            self.dirname, 'PparGC_rtheta', 'PperpGC_rtheta',
            t1=timestep, spnum=self.spnum
        )
        dens_2D = mylib.read_data(
            self.dirname, 'densGC_rtheta', t1=timestep, spnum=self.spnum
        )

        # Apply radial mask
        P_perp = P_perp[:, self.rmask]
        dens_2D = dens_2D[:, self.rmask]

        # Calculate diamagnetic velocities using mylib function
        dr_vdiag, dth_vdiag = mylib.calc_diamagnetic_velocity_2d(
            P_perp, dens_2D, self.rg, self.thg, self.B, self.jacob_space
        )

        # Calculate Reynolds stress using mylib function
        RS_diag = mylib.calc_reynolds_stress_from_velocities(
            dr_vdiag, dth_vdiag, self.rg, self.thg
        )

        return RS_diag
    
    def calc_RS_mix(self, timestep: int,
                   reduction: Literal['mean', 'rms'] = 'mean') -> np.ndarray:
        """
        Calculate mixed (ExB × diamagnetic) component of Reynolds stress

        Args:
            timestep: Time step index
            reduction: How to reduce phi dimension

        Returns:
            2D array [n_theta, n_r] of RS_mix
        """
        # Load 3D potential
        Phi3D = mylib.read_data(self.dirname, 'Phi_3D', t1=timestep-1501, spnum=self.spnum)
        Phi3D = Phi3D[:, :, self.rmask]

        # Load 2D pressure and density
        P_perp = mylib.read_data(
            self.dirname, 'PperpGC_rtheta', t1=timestep, spnum=self.spnum
        )[:, self.rmask]

        dens_2D = mylib.read_data(
            self.dirname, 'densGC_rtheta', t1=timestep, spnum=self.spnum
        )[:, self.rmask]

        # Calculate diamagnetic velocities (2D) using mylib function
        dr_vdiag, dth_vdiag = mylib.calc_diamagnetic_velocity_2d(
            P_perp, dens_2D, self.rg, self.thg, self.B, self.jacob_space
        )

        # Calculate ExB velocities (3D) using mylib function
        dr_vExB, dth_vExB = mylib.calc_exb_velocity_2d(
            Phi3D, self.rg, self.thg, self.B, self.jacob_space
        )

        # Broadcast diamagnetic velocities to 3D for mixed term
        dr_vdiag_3d = dr_vdiag[np.newaxis, :, :]
        dth_vdiag_3d = dth_vdiag[np.newaxis, :, :]

        # Calculate mixed Reynolds stress using mylib function
        # Note: Using ExB velocity gradients with diamagnetic velocity
        RS_3d = mylib.calc_reynolds_stress_from_velocities(
            dr_vdiag_3d, dth_vExB, self.rg, self.thg
        ) + mylib.calc_reynolds_stress_from_velocities(
            dr_vExB, dth_vdiag_3d, self.rg, self.thg
        )

        return self._reduce_3d_to_2d(RS_3d, reduction)
    
    def plot_RS_component(self, RS_data: np.ndarray, 
                         component_name: str,
                         timestep: int,
                         vmin: Optional[float] = None,
                         vmax: Optional[float] = None,
                         colormap: str = 'RdBu_r',
                         font_size: int = 12,
                         ax: Optional[plt.Axes] = None) -> tuple:
        """
        Plot single RS component as 2D contour
        
        Args:
            RS_data: 2D array [n_theta, n_r]
            component_name: Name for title
            timestep: Time step for title
            vmin, vmax: Color scale limits
            colormap: Colormap name
            font_size: Base font size
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            (fig, ax) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        # Create meshgrid
        theta_deg = np.rad2deg(self.thg)
        R, Theta = np.meshgrid(self.rg, theta_deg)

        # Auto-scale color limits using mylib function
        vmin, vmax = mylib.auto_color_limits(RS_data, mode='symmetric',
                                             vmin=vmin, vmax=vmax)

        # Plot
        contour = ax.pcolormesh(R, Theta, RS_data,
                               cmap=colormap, vmin=vmin, vmax=vmax,
                               shading='gouraud')

        # Add colorbar using mylib function
        mylib.add_colorbar(contour, ax, label=f'RS [{component_name}]',
                          font_size=font_size)
        
        ax.set_xlabel('r/a', size=font_size+2)
        ax.set_ylabel('θ [deg]', size=font_size+2)
        ax.set_title(f'Reynolds Stress: {component_name} (t={timestep})', 
                    size=font_size+4)
        ax.tick_params(labelsize=font_size)
        
        return fig, ax
    
    def plot_RS_all(self, timestep: int, 
                   reduction: Literal['mean', 'rms'] = 'mean',
                   colormap: str = 'RdBu_r',
                   font_size: int = 12,
                   save_path: Optional[str] = None):
        """
        Plot all three RS components in a single figure
        
        Args:
            timestep: Time step index
            reduction: How to reduce phi dimension
            colormap: Colormap name
            font_size: Base font size
            save_path: Path to save figure (optional)
        """
        # Calculate all components
        print(f"Calculating RS components for t={timestep}...")
        RS_elec = self.calc_RS_elec(timestep, reduction)
        RS_diag = self.calc_RS_diag(timestep)
        RS_mix = self.calc_RS_mix(timestep, reduction)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Determine common color scale using mylib function
        all_data = np.concatenate([
            RS_elec.ravel(), RS_diag.ravel(), RS_mix.ravel()
        ])
        vmin, vmax = mylib.auto_color_limits(all_data, mode='symmetric')

        # Plot each component with common scale
        self.plot_RS_component(RS_elec, 'ExB', timestep,
                              vmin=vmin, vmax=vmax,
                              colormap=colormap, font_size=font_size,
                              ax=axes[0])
        self.plot_RS_component(RS_diag, 'Diamagnetic', timestep,
                              vmin=vmin, vmax=vmax,
                              colormap=colormap, font_size=font_size,
                              ax=axes[1])
        self.plot_RS_component(RS_mix, 'Mixed', timestep,
                              vmin=vmin, vmax=vmax,
                              colormap=colormap, font_size=font_size,
                              ax=axes[2])
        
        fig.suptitle(f'Reynolds Stress Components (t={timestep}, {reduction} reduction)', 
                    size=font_size+6, y=1.02)
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        return fig, axes
    
    def plot_RS_radial_profile(self, timestep: int,
                               min_angle: Optional[float] = None,
                               max_angle: Optional[float] = None,
                               reduction: Literal['mean', 'rms'] = 'mean',
                               font_size: int = 12,
                               save_path: Optional[str] = None):
        """
        Plot radial profiles of RS components averaged over theta range
        
        Args:
            timestep: Time step index
            min_angle: Minimum angle in degrees (None = full theta average)
            max_angle: Maximum angle in degrees (None = full theta average)
            reduction: How to reduce phi dimension
            font_size: Base font size
            save_path: Path to save figure (optional)
        """
        # Calculate all components
        RS_elec = self.calc_RS_elec(timestep, reduction)
        RS_diag = self.calc_RS_diag(timestep)
        RS_mix = self.calc_RS_mix(timestep, reduction)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if min_angle is None or max_angle is None:
            # Full theta average
            RS_elec_prof = np.mean(RS_elec, axis=0)
            RS_diag_prof = np.mean(RS_diag, axis=0)
            RS_mix_prof = np.mean(RS_mix, axis=0)
            title_suffix = "(θ-averaged)"
        else:
            # Average over specified theta range
            angle_indices = self._get_angle_indices(min_angle, max_angle)
            
            if len(angle_indices) == 0:
                print(f"Warning: No angles found in range [{min_angle}°, {max_angle}°]")
                return None, None
            
            RS_elec_prof = np.mean(RS_elec[angle_indices, :], axis=0)
            RS_diag_prof = np.mean(RS_diag[angle_indices, :], axis=0)
            RS_mix_prof = np.mean(RS_mix[angle_indices, :], axis=0)
            RS_total_prof = RS_elec_prof + RS_diag_prof + RS_mix_prof
            title_suffix = f"(θ ∈ [{min_angle:.0f}°, {max_angle:.0f}°])"
        
        # Plot
        ax.plot(self.rg, RS_elec_prof, 'o-', label='ExB', linewidth=2, markersize=4)
        ax.plot(self.rg, RS_diag_prof, 's-', label='Diamagnetic', linewidth=2, markersize=4)
        ax.plot(self.rg, RS_mix_prof, '^-', label='Mixed', linewidth=2, markersize=4)
        ax.plot(self.rg, RS_total_prof, 'k--', label='Total', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('r/a', size=font_size+2)
        ax.set_ylabel('Reynolds Stress', size=font_size+2)
        ax.set_title(f'Radial Profile {title_suffix} (t={timestep})', 
                    size=font_size+4)
        ax.legend(fontsize=font_size)
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
        description='Analyze Reynolds stress components in plasma turbulence',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic parameters (aligned with other code)
    parser.add_argument('fn', type=str, help='Data directory path')
    parser.add_argument('-s', '--spnum', type=int, default=0, 
                       help='Species number (0=ions)')
    
    # Time parameters (aligned with other code)
    parser.add_argument('-t', '--timesteps', nargs='+', type=int, default=None,
                       help='List of timesteps to plot')
    
    # Spatial parameters (aligned with other code)
    parser.add_argument('--r_min', type=float, default=0.7,
                       help='Minimum radial coordinate for analysis')
    parser.add_argument('--r_max', type=float, default=1.2,
                       help='Maximum radial coordinate for analysis')
    
    # Angular range for radial profile averaging
    parser.add_argument('--min_angle', type=float, default=None,
                       help='Minimum angle for radial profile theta averaging (degrees)')
    parser.add_argument('--max_angle', type=float, default=None,
                       help='Maximum angle for radial profile theta averaging (degrees)')
    
    # Analysis options
    parser.add_argument('--reduction', type=str, default='mean',
                       choices=['mean', 'rms'],
                       help='Reduction method for toroidal dimension')
    
    # Plot options
    parser.add_argument('--plot_contour', action='store_true',
                       help='Plot 2D contour maps')
    parser.add_argument('--plot_profile', action='store_true',
                       help='Plot radial profiles')
    
    # Visualization parameters (aligned with other code)
    parser.add_argument('--colormap', type=str, default='RdBu_r',
                       help='Colormap for the plot')
    parser.add_argument('--font_size', type=int, default=12,
                       help='Base font size')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plots to specified directory')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    
    # Initialize analyzer
    analyzer = Reynolds_stress_analyzer(
        args.fn, 
        spnum=args.spnum,
        r_min=args.r_min,
        r_max=args.r_max
    )
    
    # Determine timesteps
    if args.timesteps is None:
        timesteps = [0]  # Default
    else:
        timesteps = args.timesteps
    
    # Determine what to plot (default: both if neither specified)
    plot_contour = args.plot_contour
    plot_profile = args.plot_profile
    if not plot_contour and not plot_profile:
        plot_contour = True
        plot_profile = True
    
    # Process time steps
    for timestep in timesteps:
        print(f"\n{'='*60}")
        print(f"Processing time step {timestep}")
        print('='*60)
        
        # Plot 2D contours
        if plot_contour:
            save_path = None
            if args.save:
                save_path = os.path.join(args.save, f'RS_contour_t{timestep}.png')
            
            analyzer.plot_RS_all(
                timestep, 
                reduction=args.reduction,
                colormap=args.colormap,
                font_size=args.font_size,
                save_path=save_path
            )
            # Plot radial profiles
            analyzer.plot_RS_radial_profile(
                timestep, 
                min_angle=args.min_angle,                
                max_angle=args.max_angle,
                reduction=args.reduction,
                font_size=args.font_size,
                save_path=save_path)
            

        
        # Plot radial profiles
        if plot_profile:
            save_path = None
            if args.save:
                angle_str = ""