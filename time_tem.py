import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mylib
import matplotlib.ticker as ticker
from typing import Tuple, Optional


class TemperatureFluctuationAnalyzer:
    """Analyzer for temperature fluctuations in plasma simulation data"""
    
    def __init__(self, dirname: str, min_r: float = 0.2, max_r: float = 1.0):
        self.dirname = dirname
        self.min_r = min_r
        self.max_r = max_r
        
        # Load geometry data
        self._load_geometry()
        
    def _load_geometry(self):
        """Load radial grid and normalization constants"""
        self.rg = mylib.read_data(self.dirname, 'rg')
        self.R0, self.rhostar = mylib.read_data(self.dirname, 'R0', 'rhostar')
        
        # Normalize radial coordinate
        self.rg *= self.rhostar
        self.R0 *= self.rhostar
        
        # Select radial range
        self.xind = np.where((self.rg < self.max_r) & (self.rg > self.min_r))[0]
        self.rg = self.rg[self.xind]
        
        print(f"Radial range: {self.min_r:.2f} < r < {self.max_r:.2f}")
        print(f"Number of radial points: {len(self.rg)}")
        
    def load_temperature_data(self, n1: int, n: int, dn: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load and calculate temperature data for time series"""
        
        print(f"Loading temperature data from t={n1} to t={n1+(n-1)*dn} (step={dn})")
        
        rg_len = len(self.rg)
        temperature_2d = np.zeros([rg_len, n])
        time_array = np.zeros(n)
        
        for ni in range(n):
            itime = n1 + ni * dn
            try:
                pressure, density, time_diag = mylib.read_data(
                    self.dirname, 'stress_FSavg', 'dens_FSavg', 'time_diag', t1=itime
                )
                
                # Calculate temperature Ti = P/n
                temperature = pressure / density
                temperature_2d[:, ni] = temperature[self.xind]
                time_array[ni] = time_diag
                
            except Exception as e:
                print(f"Warning: Error loading data at time step {itime}: {e}")
                temperature_2d[:, ni] = np.nan
                time_array[ni] = itime  # Fallback
                
        return temperature_2d, time_array
    
    def remove_temporal_trend(self, temperature_2d: np.ndarray, 
                            window_size: int = 201) -> np.ndarray:
        """Remove temporal trend using moving average
        
        Args:
            temperature_2d: 2D array of temperature [radius, time]
            window_size: Size of moving average window
            
        Returns:
            Detrended temperature fluctuations
        """
        
        print(f"Removing temporal trend with window size: {window_size}")
        
        rg_len, n_time = temperature_2d.shape
        fluctuations = np.zeros_like(temperature_2d)
        
        # Apply moving average detrending for each radial position
        for ri in range(rg_len):
            time_series = temperature_2d[ri, :]
            
            # Handle NaN values
            if np.all(np.isnan(time_series)):
                fluctuations[ri, :] = np.nan
                continue
                
            # Pad data for edge handling
            pad_width = window_size // 2
            padded_data = np.pad(time_series, pad_width=pad_width, mode='reflect')
            
            # Calculate moving average
            window = np.ones(window_size) / window_size
            moving_avg = np.convolve(padded_data, window, mode='valid')
            
            # Extract fluctuations
            fluctuations[ri, :] = time_series - moving_avg
            
        return fluctuations
    
    def create_contour_plot(self, temperature_fluctuations: np.ndarray, 
                          time_array: np.ndarray,
                          colormap: str = 'hot',
                          percentile_limits: Tuple[float, float] = (1.0, 99.0),
                          window_size: int = 201,
                          save_path: Optional[str] = None,
                          plot_title: Optional[str] = None) -> None:
        """Create contour plot of temperature fluctuations"""
        
        fig, ax = mylib.init_plot_params(figsize=(5,8), font_size=14)
        
        # Create meshgrid
        X, Y = np.meshgrid(self.rg, time_array[window_size//2: -window_size//2 + 1])
        
        # Calculate colorbar limits using percentiles
        temp_valid = temperature_fluctuations[:, window_size//2: -window_size//2 + 1]
        valid_data = temp_valid[~np.isnan(temp_valid)]
        # print(np.shape(valid_data))
        # print(np.shape(temperature_fluctuations))
        # valid_data = valid_data[window_size//2: -window_size//2 + 1, :]
        if len(valid_data) > 0:
            if percentile_limits[0] < 0:
                vmin = -np.percentile(valid_data, -percentile_limits[0])
            else:
                vmin = np.percentile(valid_data, percentile_limits[0])
            vmax = np.percentile(valid_data, percentile_limits[1])
        else:
            vmin, vmax = -1, 1
            
        # vmin = -0.01
        # vmax = 0.01
            
        # Create contour plot
        mesh = ax.pcolormesh(X, Y, np.transpose(temp_valid),
                           cmap=colormap, vmin=vmin, vmax=vmax, shading='gouraud')
        
        # Add colorbar
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.ax.tick_params(labelsize=14)
        # cbar.set_label('Temperature Fluctuation', size=16)
        
        # Format axes
        ax.set_xlabel(r'r/a', size=20)
        ax.set_ylabel(r'Time [$\omega_c^{-1}$]', size=20)
        ax.tick_params(axis='both', labelsize=14)
        
        # Format y-axis in scientific notation
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_size(14)
        
        # Set title
        # if plot_title:
        #     ax.set_title(plot_title, size=18)
        # else:
        #     ax.set_title('Temperature Fluctuations', size=18)
        
        # Adjust layout
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
        
        plt.show()
    
    def analyze_fluctuations(self, n1: int, n: int, dn: int,
                           window_size: int = 201,
                           colormap: str = 'hot',
                           percentile_limits: Tuple[float, float] = (1.0, 99.0),
                           save_path: Optional[str] = None,
                           plot_title: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Complete analysis pipeline for temperature fluctuations"""
        
        # Load data
        temperature_2d, time_array = self.load_temperature_data(n1, n, dn)
        
        # Remove temporal trend
        fluctuations = self.remove_temporal_trend(temperature_2d, window_size)
        
        # Create visualization
        self.create_contour_plot(fluctuations, time_array, colormap, 
                               percentile_limits, window_size, save_path, plot_title)
        
        return fluctuations, time_array, self.rg


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Temperature Fluctuation Analyzer')
    
    # Required arguments
    parser.add_argument('fn', type=str, help='Data directory path')
    
    # Time parameters
    parser.add_argument('-n1', type=int, default=0, 
                       help='First time step (default: 0)')
    parser.add_argument('-n', type=int, default=1, 
                       help='Number of time steps (default: 1)')
    parser.add_argument('-dn', type=int, default=1, 
                       help='Time step increment (default: 1)')
    
    # Spatial parameters
    parser.add_argument('--min_r', type=float, default=0.2,
                       help='Minimum radial coordinate (default: 0.2)')
    parser.add_argument('--max_r', type=float, default=1.0,
                       help='Maximum radial coordinate (default: 1.0)')
    
    # Processing parameters
    parser.add_argument('--window_size', type=int, default=201,
                       help='Moving average window size (default: 201)')
    
    # Visualization parameters
    parser.add_argument('--colormap', type=str, default='hot',
                       help='Colormap for contour plot (default: hot)')
    parser.add_argument('--percentile_low', type=float, default=1.0,
                       help='Lower percentile for colorbar (default: 1.0)')
    parser.add_argument('--percentile_high', type=float, default=99.0,
                       help='Upper percentile for colorbar (default: 99.0)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plot to specified path')
    parser.add_argument('--title', type=str, default=None,
                       help='Custom plot title')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.window_size < 1:
        raise ValueError("Window size must be positive")
    if args.min_r >= args.max_r:
        raise ValueError("min_r must be less than max_r")
    
    # Create analyzer
    dirname = os.path.dirname(args.fn) if os.path.dirname(args.fn) else '.'
    analyzer = TemperatureFluctuationAnalyzer(dirname, args.min_r, args.max_r)
    
    # Run analysis
    fluctuations, time_array, radial_grid = analyzer.analyze_fluctuations(
        args.n1, args.n, args.dn, args.window_size, args.colormap,
        (args.percentile_low, args.percentile_high), args.save, args.title
    )
    
    # Print summary statistics
    valid_data = fluctuations[~np.isnan(fluctuations)]
    if len(valid_data) > 0:
        print(f"\nFluctuation statistics:")
        print(f"  Mean: {np.mean(valid_data):.6f}")
        print(f"  RMS: {np.sqrt(np.mean(valid_data**2)):.6f}")
        print(f"  Min: {np.min(valid_data):.6f}")
        print(f"  Max: {np.max(valid_data):.6f}")


if __name__ == '__main__':
    main()