import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import mylib
import matplotlib.ticker as ticker
from tqdm import tqdm
from scipy.signal import welch
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, asdict
import warnings
import json
from pathlib import Path

"""
Phi Analysis Pipeline - GYSELA Turbulence Analysis Tool

QUICK START:
    # Command line
    python phi_analysis.py /path/to/data -n1 1500 -n 800

    # Python script
    from phi_analysis import PhiAnalysisPipeline, AnalysisConfig
    config = AnalysisConfig(dirname='/path/to/data', n_start=1500, n_steps=800)
    pipeline = PhiAnalysisPipeline(config)
    result = pipeline.run_full_analysis()
    pipeline.plot_all()

For detailed usage guide, see: phi_analysis_usage.md
Or run: python phi_analysis.py --help
"""

@dataclass
class AnalysisConfig:
    """Configuration for phi analysis"""
    # Directory and time range
    dirname: str
    n_start: int = 0
    n_steps: int = 100
    dn: int = 1
    
    # Spatial ranges
    angle_min_deg: float = -30.0
    angle_max_deg: float = 30.0
    r_min: float = 0.4
    r_max: float = 0.6
    
    # Analysis options
    remove_n0_mode: bool = True
    compute_slope: bool = True
    slope_freq_min: float = 3e-4
    slope_freq_max: float = 2e-3
    
    # Welch parameters
    welch_nperseg: Optional[int] = None  # None = auto
    
    # Output options
    save_figures: bool = False
    save_data: bool = False
    output_dir: Optional[str] = None
    figure_format: str = 'png'
    figure_dpi: int = 300
    
    # Plot options
    colormap: str = 'hot'
    contour_percentile: float = 99.0
    font_size: int = 14
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.r_min >= self.r_max:
            raise ValueError("r_min must be less than r_max")
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class AnalysisResult:
    """Container for analysis results"""
    # Grid data
    rg: np.ndarray
    time_arr: np.ndarray
    
    # 2D data
    Phi2D: np.ndarray  # [n_radial, n_time]
    
    # Spectrum data
    frequencies: Optional[np.ndarray] = None
    psd: Optional[np.ndarray] = None
    r_center: Optional[float] = None
    
    # Slope analysis
    slope: Optional[float] = None
    slope_intercept: Optional[float] = None
    slope_freqs: Optional[np.ndarray] = None
    
    # Metadata
    config: Optional[AnalysisConfig] = None
    
    def save_to_npz(self, filepath: str):
        """Save all numerical data to npz file"""
        save_dict = {
            'rg': self.rg,
            'time_arr': self.time_arr,
            'Phi2D': self.Phi2D,
        }
        
        if self.frequencies is not None:
            save_dict.update({
                'frequencies': self.frequencies,
                'psd': self.psd,
                'r_center': np.array([self.r_center]),
            })
        
        if self.slope is not None:
            save_dict.update({
                'slope': np.array([self.slope]),
                'slope_intercept': np.array([self.slope_intercept]),
            })
        
        np.savez_compressed(filepath, **save_dict)
        print(f"Data saved to: {filepath}")


class PhiAnalysisPipeline:
    """Pipeline for phi fluctuation analysis"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.result = None
        
        # Load grid data
        self._load_grid_data()
    
    def _load_grid_data(self):
        """Load spatial grid and prepare indices"""
        # Load radial grid
        self.rg_orig = mylib.read_data(self.config.dirname, 'rg')
        R0, inv_rho = mylib.read_data(self.config.dirname, 'R0', 'rhostar')
        
        rg_max = 1.0 / inv_rho
        self.xind = np.squeeze(np.where(
            (self.rg_orig <= rg_max) & (self.rg_orig > 0. * rg_max)
        ))
        self.rg = self.rg_orig[self.xind] / rg_max
        
        # Load ballooning angle
        bal_ang_orig = mylib.esti_bal_angle(self.config.dirname)
        self.bal_ang = bal_ang_orig[:, self.xind]
        self.angles_rad = bal_ang_orig[:, 0]
        
        # Get angle indices
        self.angle_indices = self._get_angle_indices()
        
        # Get radial indices for spectrum
        self.radial_indices = self._get_radial_indices()
        
        print(f"Grid loaded:")
        print(f"  Radial points: {len(self.rg)} (r/a ∈ [{self.rg[0]:.3f}, {self.rg[-1]:.3f}])")
        print(f"  Angle points in range: {len(self.angle_indices)}")
        print(f"  Radial points for spectrum: {len(self.radial_indices)}")
    
    def _get_angle_indices(self) -> np.ndarray:
        """Get angle indices within specified range"""
        min_rad = np.deg2rad(self.config.angle_min_deg) % (2 * np.pi)
        max_rad = np.deg2rad(self.config.angle_max_deg) % (2 * np.pi)
        
        angles_norm = self.angles_rad % (2 * np.pi)
        
        if min_rad <= max_rad:
            mask = (angles_norm >= min_rad) & (angles_norm <= max_rad)
        else:
            mask = (angles_norm >= min_rad) | (angles_norm <= max_rad)
        
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            raise ValueError(
                f"No angles in range [{self.config.angle_min_deg}°, "
                f"{self.config.angle_max_deg}°]"
            )
        
        return indices
    
    def _get_radial_indices(self) -> np.ndarray:
        """Get radial indices within specified range"""
        mask = (self.rg >= self.config.r_min) & (self.rg <= self.config.r_max)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            warnings.warn(
                f"No radial points in range [r/a={self.config.r_min:.3f}, "
                f"{self.config.r_max:.3f}]. Using middle point."
            )
            indices = np.array([len(self.rg) // 2])
        
        return indices
    
    def process_timeseries(self) -> AnalysisResult:
        """Process phi data over time"""
        n = self.config.n_steps
        Phi2D = np.zeros([len(self.rg), n])
        time_arr = np.zeros(n)
        
        print(f"\nProcessing time series:")
        print(f"  Steps: {self.config.n_start} to "
              f"{self.config.n_start + (n-1)*self.config.dn} "
              f"(increment: {self.config.dn})")
        
        for ni in tqdm(range(n), desc="Time steps", unit="step"):
            itime = self.config.n_start + ni * self.config.dn
            
            try:
                # Read data
                if self.config.remove_n0_mode:
                    phirth_orig, phirth_n0, tdiag = mylib.read_data(
                        self.config.dirname, 'Phirth', 'Phirth_n0', 
                        'time_diag', t1=itime
                    )
                    pdata = (phirth_orig - phirth_n0)[:, self.xind]
                else:
                    phirth_orig, tdiag = mylib.read_data(
                        self.config.dirname, 'Phirth', 'time_diag', t1=itime
                    )
                    pdata = phirth_orig[:, self.xind]
                
                # Average over angle range
                anal_data = pdata[self.angle_indices, :]
                phi_profile = np.mean(np.abs(anal_data), axis=0)
                
                Phi2D[:, ni] = phi_profile
                time_arr[ni] = tdiag[0]
                
            except Exception as e:
                warnings.warn(f"Error at time {itime}: {e}")
                Phi2D[:, ni] = np.nan
                time_arr[ni] = np.nan
        
        # Create result object
        self.result = AnalysisResult(
            rg=self.rg,
            time_arr=time_arr,
            Phi2D=Phi2D,
            config=self.config
        )
        
        return self.result
    
    def compute_spectrum(self) -> AnalysisResult:
        """Compute frequency spectrum"""
        if self.result is None:
            raise RuntimeError("Must run process_timeseries() first")
        
        # Remove NaN times
        valid_idx = ~np.isnan(self.result.time_arr)
        
        if np.sum(valid_idx) < 2:
            raise ValueError("Need at least 2 valid time steps")
        
        time_valid = self.result.time_arr[valid_idx]
        Phi2D_valid = self.result.Phi2D[:, valid_idx]
        
        # Calculate sampling parameters
        dt = np.mean(np.diff(time_valid))
        if dt <= 0:
            raise ValueError(f"Non-positive time step: dt={dt}")
        
        fs = 1.0 / dt
        
        # Average over radial range
        signal = np.mean(Phi2D_valid[self.radial_indices, :], axis=0)
        r_center = np.mean(self.rg[self.radial_indices])
        
        print(f"\nSpectrum computation:")
        print(f"  dt = {dt:.2e}, fs = {fs:.2e} [ω_c]")
        print(f"  Radial range: r/a ∈ [{self.rg[self.radial_indices[0]]:.3f}, "
              f"{self.rg[self.radial_indices[-1]]:.3f}] (center: {r_center:.3f})")
        print(f"  Time series length: {len(signal)}")
        
        # Welch method
        nperseg = self.config.welch_nperseg
        if nperseg is None:
            nperseg = min(len(signal), 256 if len(signal) >= 256 else len(signal))
        
        frequencies, psd = welch(signal, fs=fs, nperseg=nperseg, scaling='density')
        
        # Update result
        self.result.frequencies = frequencies
        self.result.psd = psd
        self.result.r_center = r_center
        
        return self.result
    
    def analyze_slope(self) -> AnalysisResult:
        """Analyze power law slope"""
        if self.result.frequencies is None:
            raise RuntimeError("Must run compute_spectrum() first")
        
        freq_min = self.config.slope_freq_min
        freq_max = self.config.slope_freq_max
        
        # Find frequency range
        mask = ((self.result.frequencies >= freq_min) & 
                (self.result.frequencies <= freq_max))
        indices = np.where(mask)[0]
        
        if len(indices) < 2:
            warnings.warn("Not enough points for slope fitting")
            return self.result
        
        freqs = self.result.frequencies[indices]
        psd = self.result.psd[indices]
        
        # Filter positive values
        valid = psd > 0
        if np.sum(valid) < 2:
            warnings.warn("Not enough positive PSD values")
            return self.result
        
        freqs = freqs[valid]
        psd = psd[valid]
        
        # Log-log fit
        slope, intercept = np.polyfit(np.log(freqs), np.log(psd), 1)
        
        print(f"\nSlope analysis:")
        print(f"  Frequency range: [{freq_min:.1e}, {freq_max:.1e}] [ω_c]")
        print(f"  Data points: {len(freqs)}")
        print(f"  Slope: {slope:.2f}")
        
        # Update result
        self.result.slope = slope
        self.result.slope_intercept = intercept
        self.result.slope_freqs = freqs
        
        return self.result
    
    def run_full_analysis(self) -> AnalysisResult:
        """Run complete analysis pipeline"""
        self.process_timeseries()
        self.compute_spectrum()
        
        if self.config.compute_slope:
            self.analyze_slope()
        
        return self.result
    
    def plot_contour(self, ax=None, show=True):
        """Plot contour of Phi2D"""
        if self.result is None:
            raise RuntimeError("No results to plot")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        X, Y = np.meshgrid(self.result.rg, self.result.time_arr)
        valid = ~np.isnan(self.result.Phi2D)
        
        if np.any(valid):
            vmax = np.percentile(
                self.result.Phi2D[valid], 
                self.config.contour_percentile
            )
            
            contour = ax.pcolormesh(
                X, Y, self.result.Phi2D.T,
                cmap=self.config.colormap,
                vmin=0, vmax=vmax,
                shading='gouraud'
            )
            
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label(
                r'$|\Phi|$ (angle-averaged)', 
                size=self.config.font_size
            )
            cbar.ax.tick_params(labelsize=self.config.font_size-2)
        
        # Format axes
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_size(self.config.font_size-2)
        ax.tick_params(axis='both', labelsize=self.config.font_size-2)
        
        ax.set_xlabel(r'$r/a$', size=self.config.font_size+2)
        ax.set_ylabel(r'Time [$\omega_c^{-1}$]', size=self.config.font_size+2)
        ax.set_title(
            f'Phi Fluctuation ({self.config.angle_min_deg:.0f}° to '
            f'{self.config.angle_max_deg:.0f}°)',
            size=self.config.font_size+4
        )
        
        fig.tight_layout()
        
        if self.config.save_figures:
            filepath = Path(self.config.output_dir) / f'phi_contour.{self.config.figure_format}'
            fig.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
            print(f"Contour plot saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig, ax
    
    def plot_spectrum(self, ax=None, show=True):
        """Plot frequency spectrum"""
        if self.result.frequencies is None:
            raise RuntimeError("No spectrum to plot")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        ax.loglog(
            self.result.frequencies, 
            self.result.psd,
            'b-', linewidth=1.5,
            label='Spectrum'
        )
        
        # Add slope fit if available
        if self.result.slope is not None:
            freqs = self.result.slope_freqs
            psd_fit = np.exp(self.result.slope_intercept) * (freqs ** self.result.slope)
            ax.loglog(
                freqs, psd_fit,
                'r--', linewidth=2,
                label=f'$\\omega^{{{self.result.slope:.2f}}}$'
            )
            ax.legend(fontsize=self.config.font_size-2)
        
        ax.set_xlabel(r'Frequency [$\omega_c$]', size=self.config.font_size+2)
        ax.set_ylabel(r'PSD [$|\Phi|^2 / \omega_c$]', size=self.config.font_size+2)
        ax.set_title(
            f'Frequency Spectrum at r/a = {self.result.r_center:.3f}\n'
            f'(Angle: {self.config.angle_min_deg:.0f}° to '
            f'{self.config.angle_max_deg:.0f}°)',
            size=self.config.font_size
        )
        
        ax.grid(True, which="both", ls="-", alpha=0.3)
        ax.tick_params(axis='both', labelsize=self.config.font_size-2)
        
        fig.tight_layout()
        
        if self.config.save_figures:
            filepath = Path(self.config.output_dir) / f'spectrum.{self.config.figure_format}'
            fig.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
            print(f"Spectrum plot saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig, ax
    
    def plot_all(self, show=True):
        """Create all plots in a single figure"""
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = plt.subplot(1, 2, 1)
        self.plot_contour(ax=ax1, show=False)
        
        ax2 = plt.subplot(1, 2, 2)
        self.plot_spectrum(ax=ax2, show=False)
        
        fig.tight_layout()
        
        if self.config.save_figures:
            filepath = Path(self.config.output_dir) / f'combined.{self.config.figure_format}'
            fig.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
            print(f"Combined plot saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig


def batch_analysis(config_files: List[str], output_dir: str = './batch_results'):
    """Run analysis for multiple configurations"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for config_file in config_files:
        print(f"\n{'='*60}")
        print(f"Processing: {config_file}")
        print('='*60)
        
        config = AnalysisConfig.load_from_file(config_file)
        config.output_dir = str(Path(output_dir) / Path(config_file).stem)
        config.save_figures = True
        config.save_data = True
        
        try:
            pipeline = PhiAnalysisPipeline(config)
            result = pipeline.run_full_analysis()
            pipeline.plot_all(show=False)
            
            # Save data
            data_file = Path(config.output_dir) / 'result.npz'
            result.save_to_npz(str(data_file))
            
            results[config_file] = result
            
        except Exception as e:
            print(f"Error processing {config_file}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phi fluctuation analysis pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('dirname', type=str, help='Data directory')
    parser.add_argument('-n1', type=int, default=0, help='Start time step')
    parser.add_argument('-n', type=int, default=100, help='Number of steps')
    parser.add_argument('-dn', type=int, default=1, help='Step increment')
    
    parser.add_argument('--min_angle', type=float, default=-30.0, help='Min angle [deg]')
    parser.add_argument('--max_angle', type=float, default=30.0, help='Max angle [deg]')
    parser.add_argument('--min_r', type=float, default=0.4, help='Min r/a')
    parser.add_argument('--max_r', type=float, default=0.6, help='Max r/a')
    
    parser.add_argument('--no_slope', action='store_true', help='Skip slope analysis')
    parser.add_argument('--save', action='store_true', help='Save figures and data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    parser.add_argument('--config', type=str, help='Load config from JSON file')
    parser.add_argument('--save_config', type=str, help='Save config to JSON file')
    
    parser.add_argument('--batch', nargs='+', help='Run batch analysis with config files')
    
    args = parser.parse_args()
    
    # Batch mode
    if args.batch:
        batch_analysis(args.batch, args.output_dir)
    
    # Single analysis mode
    else:
        if args.config:
            config = AnalysisConfig.load_from_file(args.config)
        else:
            config = AnalysisConfig(
                dirname=args.dirname,
                n_start=args.n1,
                n_steps=args.n,
                dn=args.dn,
                angle_min_deg=args.min_angle,
                angle_max_deg=args.max_angle,
                r_min=args.min_r,
                r_max=args.max_r,
                compute_slope=not args.no_slope,
                save_figures=args.save,
                save_data=args.save,
                output_dir=args.output_dir if args.save else None
            )
        
        if args.save_config:
            config.save_to_file(args.save_config)
            print(f"Configuration saved to: {args.save_config}")
        
        # Run analysis
        pipeline = PhiAnalysisPipeline(config)
        result = pipeline.run_full_analysis()
        pipeline.plot_all()
        
        if config.save_data:
            result.save_to_npz(str(Path(config.output_dir) / 'result.npz'))