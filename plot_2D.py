import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import mylib
from typing import List, Optional, Tuple
from scipy.ndimage import uniform_filter1d
import matplotlib.patches as patches

# Import data processing module
try:
    import GYS_dataprocess
except ImportError:
    print("Warning: GYS_dataprocess module not found. Using basic data loading only.")
    GYS_dataprocess = None


class PlasmaData2DPlotter:
    """Class for 2D visualization of plasma simulation data"""
    
    def __init__(self, dirname: str, spnum: int = 0, 
                 coord_system: str = 'psi_theta',
                 r_min: float = 0.7, r_max: float = 1.2):
        self.dirname = dirname
        self.spnum = spnum
        self.coord_system = coord_system
        self.r_min = r_min
        self.r_max = r_max
        
        # Load geometry data
        self._load_geometry()
        
    def _load_geometry(self):
        """Load geometric coordinates and setup grids"""
        try:
            # Load normalization constants
            self.R0, self.rhostar = mylib.read_data(self.dirname, 'R0', 'rhostar', spnum=self.spnum)
            
            # Load coordinate arrays
            self.R, self.Z = mylib.read_data(self.dirname, 'R', 'Z', spnum=self.spnum)
            self.rg, self.thetag = mylib.read_data(self.dirname, 'rg', 'thetag', spnum=self.spnum)
            self.psi = mylib.read_data(self.dirname, 'psi', spnum=self.spnum)
            
            # Normalize coordinates
            self.R /= self.R0
            self.Z /= self.R0
            self.rg *= self.rhostar
            
            # Setup coordinate meshgrids
            if self.coord_system == 'psi_theta':
                self.X, self.Y = np.meshgrid(self.rg, self.thetag/np.pi) # self.psi/self.psi[-1]
                self.x_label = r'$r/a$' #r'$\rho = \psi/\psi_{\mathrm{edge}}$'
                self.y_label = r'$\theta/\pi$'
            else:  # R-Z coordinates
                self.X, self.Y = self.R, self.Z
                self.x_label = r'$R/R_0$'
                self.y_label = r'$Z/R_0$'
            
            # Setup radial range indices
            self.x1, self.x2 = 0, -1
            self.y1 = np.argmin(np.abs(self.rg - self.r_min))
            self.y2 = np.argmin(np.abs(self.rg - self.r_max))
            
            # Reference radial positions for visualization
            self.reference_radii = [0.4, 0.6, 0.9, 1.15]
            
            print(f"Geometry loaded: {len(self.rg)} radial points, {len(self.thetag)} poloidal points")
            print(f"Radial analysis range: r ∈ [{self.rg[self.y1]:.2f}, {self.rg[self.y2]:.2f}]")
            
        except Exception as e:
            print(f"Error loading geometry: {e}")
            raise
    
    def load_data(self, dtype: str, timestep: int) -> np.ndarray:
        """Load data for specified type and timestep"""
        try:
            if GYS_dataprocess is not None:
                return GYS_dataprocess.process_data(dtype, self.dirname, timestep, self.spnum, mylib)
            else:
                return mylib.read_data(self.dirname, dtype, t1=timestep, spnum=self.spnum)
        except Exception as e:
            print(f"Error loading {dtype} data at timestep {timestep}: {e}")
            raise
    
    def apply_perturbation_analysis(self, data: np.ndarray, mode: str, dtype: str, 
                                   timestep: int, fft_modes: int = 19, 
                                   reference_time: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """Apply different types of perturbation analysis"""
        
        if mode == 'y':  # FFT filtering for perturbations
            return self._apply_fft_filtering(data, fft_modes)
            
        elif mode == 'r':  # Reference subtraction
            return self._apply_reference_subtraction(data, dtype, timestep, reference_time)
            
        elif mode == 'd':  # Density normalization
            return self._apply_density_normalization(data, timestep)
            
        elif mode == 'p':  # Pressure normalization
            return self._apply_pressure_normalization(data, timestep)
            
        else:  # No perturbation analysis
            cbar_max_lim = np.max((data)[self.x1:self.x2, self.y1:self.y2])
            cbar_min_lim = np.min((data)[self.x1:self.x2, self.y1:self.y2])
            return data, cbar_min_lim, cbar_max_lim
    
    def _apply_fft_filtering(self, data: np.ndarray, fft_modes: int) -> Tuple[np.ndarray, float]:
        """Apply FFT filtering to extract perturbations"""
        try:
            bal_ang = mylib.esti_bal_angle(self.dirname)
            databal = mylib.interp2_bal(bal_ang, data, self.thetag)
            
            # Apply FFT filtering
            data_fft = np.fft.fft(databal, axis=0)
            for j in range(0, fft_modes):
                data_fft[j, :] = 0
                data_fft[-j, :] = 0
            
            pdata_bal = np.fft.ifft(data_fft, axis=0)
            pdata = np.real(mylib.inv_interp2_bal(bal_ang, pdata_bal, self.thetag))
            
            cbar_lim = np.max(np.abs(pdata)[self.x1:self.x2, self.y1:self.y2])
            return pdata, -cbar_lim, cbar_lim
            
        except Exception as e:
            print(f"Error in FFT filtering: {e}")
            return data, -np.max(np.abs(data)), np.max(np.abs(data))
    
    def _apply_reference_subtraction(self, data: np.ndarray, dtype: str, timestep: int,
                                    reference_time: Optional[int]) -> Tuple[np.ndarray, float]:
        """Apply reference subtraction"""
        try:
            if reference_time is not None:
                ref_data = self.load_data(dtype, reference_time)
            else:
                # Use n0 component for Phi data, or load default reference
                if dtype == 'Phirth' and GYS_dataprocess is not None:
                    ref_data = GYS_dataprocess.process_data('Phirth_n0', self.dirname, timestep, self.spnum, mylib)
                else:
                    # Default reference (could be made configurable)
                    ref_data = self.load_data(dtype, 1501)
            
            pdata = data - ref_data
            cbar_lim = np.max(np.abs(pdata)[self.x1:self.x2, self.y1:self.y2])
            return pdata, -cbar_lim, cbar_lim
            
        except Exception as e:
            print(f"Error in reference subtraction: {e}")
            return data, np.max(np.abs(data))
    
    def _apply_density_normalization(self, data: np.ndarray, timestep: int) -> Tuple[np.ndarray, float]:
        """Normalize by density"""
        try:
            dens_data = mylib.read_data(self.dirname, 'densGC_rtheta', t1=timestep, spnum=self.spnum)
            pdata = data / dens_data #* 2.5 / 0.65  # Normalization factors could be parameters
            cbar_max_lim = np.max((pdata)[self.x1:self.x2, self.y1:self.y2])
            cbar_min_lim = np.min((pdata)[self.x1:self.x2, self.y1:self.y2])
            return pdata, cbar_min_lim, cbar_max_lim
            
        except Exception as e:
            print(f"Error in density normalization: {e}")
            return data, np.max(np.abs(data))
    
    def _apply_pressure_normalization(self, data: np.ndarray, timestep: int) -> Tuple[np.ndarray, float]:
        """Normalize by pressure"""
        try:
            Ppar_data, Pperp_data = mylib.read_data(self.dirname, 'PparGC_rtheta', 'PperpGC_rtheta', 
                                                   t1=timestep, spnum=self.spnum)
            pressure = Ppar_data/2 + Pperp_data
            pdata = data / pressure * 2.5
            cbar_max_lim = np.max(np.abs(pdata)[self.x1:self.x2, self.y1:self.y2])
            cbar_min_lim = np.min(np.abs(pdata)[self.x1:self.x2, self.y1:self.y2])
            return pdata, cbar_min_lim, cbar_max_lim
            
        except Exception as e:
            print(f"Error in pressure normalization: {e}")
            return data, np.min(np.abs(data)), np.max(np.abs(data))
    
    def plot_2d_data(self, dtype: str, timestep: int, perturbation_mode: str = 'n',
                    fft_modes: int = 19, reference_time: Optional[int] = None,
                    colormap: str = 'bwr', save_path: Optional[str] = None,
                    show_reference_lines: bool = True, font_size: int = 12):
        """Create 2D plot of plasma data"""
        
        # Load data
        data = self.load_data(dtype, timestep)
        
        # Get time information
        try:
            time_diag = mylib.read_data(self.dirname, 'time_diag', t1=timestep, spnum=self.spnum)
            time_value = time_diag[0]
        except:
            time_value = timestep
        
        # Apply perturbation analysis
        processed_data, cbar_min_lim, cbar_max_lim = self.apply_perturbation_analysis(
            data, perturbation_mode, dtype, timestep, fft_modes, reference_time
        )
        
        # Create plot
        _, ax = mylib.init_plot_params(font_size=font_size)
        
        # Plot data
        mesh = plt.pcolormesh(
            self.X[self.x1:self.x2, self.y1:self.y2], 
            self.Y[self.x1:self.x2, self.y1:self.y2], 
            processed_data[self.x1:self.x2, self.y1:self.y2], 
            cmap=colormap, vmin=cbar_min_lim, vmax=cbar_max_lim, shading='gouraud'
            # vmin = 0.2, vmax = 1.8, shading='gouraud'
        )
        
        # Add colorbar
        cbar = plt.colorbar(mesh)
        cbar.ax.tick_params(labelsize=font_size-2)
        
        # Set labels and title
        plt.xlabel(self.x_label, size=font_size+8)
        plt.ylabel(self.y_label, size=font_size+8)
        
        # Create title with perturbation info
        pert_info = self._get_perturbation_info(perturbation_mode, fft_modes, reference_time)
        title = f"{dtype}{pert_info}, t = {time_value:.2e}"
        plt.title(title, size=font_size+12)
        
        # Add reference lines
        if show_reference_lines:
            self._add_reference_lines(ax)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
        
        plt.show()
    
    def _get_perturbation_info(self, mode: str, fft_modes: int, reference_time: Optional[int]) -> str:
        """Generate perturbation info string for title"""
        if mode == 'y':
            return f" (FFT filtered, modes={fft_modes})"
        elif mode == 'r':
            ref_str = f"t={reference_time}" if reference_time else "n0"
            return f" (ref. subtracted, {ref_str})"
        elif mode == 'd':
            return " (density normalized)"
        elif mode == 'p':
            return " (pressure normalized)"
        else:
            return ""
    
    def _add_reference_lines(self, ax):
        """Add reference lines to the plot"""
        if self.coord_system == 'psi_theta':
            # Add r/a = 1 line in psi-theta coordinates
            plt.axvline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        else:
            # Add reference flux surfaces in R-Z coordinates
            for rad in self.reference_radii:
                try:
                    ir = np.argmin(np.abs(self.rg - rad))
                    plt.plot(self.R[self.x1:self.x2, ir], self.Z[self.x1:self.x2, ir], 
                            color='black', ls='--', lw=1.0, alpha=0.7)
                except:
                    continue
            ir = np.argmin(np.abs(self.rg - 1.0))
            plt.plot(self.R[self.x1:self.x2, ir], self.Z[self.x1:self.x2, ir], 
                    color='black', ls='-', lw=1.5, alpha=0.7)
    
    def plot_equilibrium(self, dtype: str, levels: int = 50, 
                        remove_mean: bool = True, save_path: Optional[str] = None):
        """Plot equilibrium quantities with contours"""
        
        # Load equilibrium data
        data = mylib.read_data(self.dirname, dtype, spnum=self.spnum)
        data /= (self.R0**2)  # Normalize
        
        if remove_mean:
            mean_data = np.mean(data, axis=0, keepdims=True)
            data = data - mean_data
        
        _, ax = mylib.init_plot_params(font_size=20)
        
        # Create filled contour plot
        contour_fill = plt.contourf(
            self.R[self.x1:self.x2, self.y1:self.y2], 
            self.Z[self.x1:self.x2, self.y1:self.y2], 
            data[self.x1:self.x2, self.y1:self.y2], 
            levels=levels, cmap='bwr'
        )
        
        plt.colorbar(contour_fill)
        
        # Add contour lines
        contour_lines = plt.contour(
            self.R[self.x1:self.x2, self.y1:self.y2], 
            self.Z[self.x1:self.x2, self.y1:self.y2], 
            data[self.x1:self.x2, self.y1:self.y2], 
            levels=20, colors='black', linewidths=0.5
        )
        
        plt.title(f"{dtype} (Equilibrium)", size=24)
        plt.xlabel('R', size=16)
        plt.ylabel('Z', size=16)
        
        # Add reference flux surfaces
        for rad in self.reference_radii:
            try:
                ir = np.argmin(np.abs(self.rg - rad))
                plt.plot(self.R[self.x1:self.x2, ir], self.Z[self.x1:self.x2, ir], 
                        color='red', ls='--', lw=1.0, alpha=0.8)
            except:
                continue
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Equilibrium plot saved as {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Enhanced 2D Plasma Data Plotter')
    
    # Required arguments
    parser.add_argument('fn', type=str, help='Data directory path')
    
    # Data parameters
    parser.add_argument('-d', '--dtype', type=str, default='Phirth',
                       help='Data type to plot (default: Phirth)')
    parser.add_argument('-s', '--spnum', type=int, default=0,
                       help='Species number (default: 0)')
    
    # Time parameters
    parser.add_argument('-t', '--timesteps', nargs='+', type=int, default=None,
                       help='List of timesteps to plot')
    parser.add_argument('--equilibrium', action='store_true',
                       help='Plot equilibrium instead of time evolution')
    
    # Analysis parameters
    parser.add_argument('-p', '--perturbation', type=str, default='n',
                       choices=['n', 'y', 'r', 'd', 'p'],
                       help='Perturbation analysis mode: n(one), y(fft), r(ref), d(dens), p(press)')
    parser.add_argument('--fft_modes', type=int, default=1,
                       help='Number of modes to filter in FFT analysis (default: 1)')
    parser.add_argument('--ref_time', type=int, default=None,
                       help='Reference timestep for subtraction (default: auto)')
    
    # Spatial parameters
    parser.add_argument('--coord_system', type=str, default='psi_theta',
                       choices=['psi_theta', 'rz'],
                       help='Coordinate system for plotting (default: psi_theta)')
    parser.add_argument('--r_min', type=float, default=0.7,
                       help='Minimum radial coordinate for analysis (default: 0.7)')
    parser.add_argument('--r_max', type=float, default=1.2,
                       help='Maximum radial coordinate for analysis (default: 1.2)')
    
    # Visualization parameters
    parser.add_argument('--colormap', type=str, default='bwr',
                       help='Colormap for the plot (default: bwr)')
    parser.add_argument('--font_size', type=int, default=12,
                       help='Base font size (default: 12)')
    parser.add_argument('--no_ref_lines', action='store_true',
                       help='Disable reference lines on plot')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plots to specified directory')
    
    args = parser.parse_args()
    
    # Get directory name
    dirname = os.path.dirname(args.fn) if os.path.dirname(args.fn) else '.'
    
    # Initialize plotter
    plotter = PlasmaData2DPlotter(
        dirname, args.spnum, args.coord_system, args.r_min, args.r_max
    )
    
    if args.equilibrium:
        # Plot equilibrium
        save_path = os.path.join(args.save, f"{args.dtype}_equilibrium.png") if args.save else None
        plotter.plot_equilibrium(args.dtype, save_path=save_path)
        
    elif args.timesteps:
        # Plot time evolution
        for i, timestep in enumerate(args.timesteps):
            save_path = None
            if args.save:
                save_path = os.path.join(args.save, f"{args.dtype}_t{timestep:05d}.png")
            
            print(f"Plotting timestep {timestep} ({i+1}/{len(args.timesteps)})")
            plotter.plot_2d_data(
                args.dtype, timestep, args.perturbation, args.fft_modes, 
                args.ref_time, args.colormap, save_path, 
                not args.no_ref_lines, args.font_size
            )
    else:
        print("Please specify timesteps with -t or use --equilibrium for equilibrium plots")


if __name__ == "__main__":
    main()