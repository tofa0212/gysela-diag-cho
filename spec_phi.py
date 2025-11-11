"""
Spectrogram Analysis Tool
Analyzes and visualizes spectrograms of phi field data across multiple modes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Tuple, List
import argparse
import os
from pathlib import Path
from scipy import signal
from tqdm import tqdm
from multiprocessing import Pool
import functools
import mylib

try:
    from GYS_dataprocess import process_data
except ImportError:
    print("Warning: GYS_dataprocess module not found. Using basic data loading only.")
    
    def process_data(dtype: str, dirname: str, t1: int, spnum: int, mylib):
        """Fallback function when GYS_dataprocess is not available"""
        return mylib.read_data(dirname, dtype, t1=t1, spnum=spnum)

def process_fft_modes(args):
    """Process FFT for a single mode index."""
    j, phi3d_selected, bal_ang = args
    databal = mylib.interp2_bal(bal_ang, phi3d_selected[j, :, :])
    data_fft_j = np.fft.fft(databal, axis=0)
    return data_fft_j


class SpectrogramAnalyzer:
    """Analyzer for creating and managing spectrograms with dynamic colorbar updates."""
    
    def __init__(self, dirn: str, min_r: float = 0.7, 
                 max_r: float = 1.1):
        """
        Initialize the analyzer.
        
        Args:
            dirn: Directory containing data files
            r1: Starting radial position
            r2: Ending radial position
        """
        self.dirn = Path(dirn)
        self.min_r = min_r
        self.max_r = max_r
        
        self._load_common_data()
        
    def _load_common_data(self):
        """Load common data required for analysis."""
        self.rg = mylib.read_data(self.dirn, 'rg')
        self.thetag, self.phig = mylib.read_data(self.dirn, 'thetag', 'phig')
        self.R0, self.inv_rho = mylib.read_data(self.dirn, 'R0', 'rhostar')
        
        # Normalize radial coordinate
        rg_max = 1. / self.inv_rho
        self.rg *= self.inv_rho
        
        # Select radial range
        self.xind = np.squeeze(np.where((self.rg <= self.max_r) & (self.rg >= self.min_r)))
        self.R0 = self.R0 / rg_max
        self.rg = self.rg[self.xind]
        self.bal_ang = mylib.esti_bal_angle(str(self.dirn))

        
    @staticmethod
    def make_odd(n: int) -> int:
        """Ensure a number is odd."""
        return n if n % 2 else n + 1

    
    def load_phi_data(self, n: int, n1: int, dn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load phi field data over time.
        
        Args:
            n: Number of time steps to load
            n1: First time index
            dn: Time step interval
            
        Returns:
            Phi: Complex array of shape (nmode, n)
            time_arr: Time values
        """
        
        Phi3d_time = None
        time_arr = np.zeros(n)
        
        for ni in tqdm(range(n), desc="Processing time steps", unit="step"):
            itime = n1 + ni * dn
            phi3d, tdiag = mylib.read_data(
                str(self.dirn), 'Phi_3D', 'time_diag', t1=itime
            )

            phi3d_selected = phi3d[:, :, self.xind]
            # Shift to ballooning representation
            data_fft =np.zeros_like(phi3d_selected, dtype=np.complex128)
            
            for j in range(np.shape(phi3d)[0]):
                databal = mylib.interp2_bal(
                    self.bal_ang[:, self.xind], 
                    phi3d_selected[j, :, :]
                )
                data_fft[j, :, :] = np.fft.fft(databal, axis=0)
                
            data_fft = np.fft.fft(data_fft, axis=0)
            
            data_2D = np.sum(self.rg* data_fft[:, :, :], axis=2)
            
            if Phi3d_time is None:
                Phi3d_time = np.zeros((n, data_2D.shape[0], data_2D.shape[1]), dtype=np.complex128)
            
            Phi3d_time[ni, :, :] = np.fft.fftshift(data_2D)
            time_arr[ni] = tdiag[0]
        
        nky, nkx = data_2D.shape
        kx = np.fft.fftshift(np.fft.fftfreq(nkx)*nkx )
        ky = np.fft.fftshift(np.fft.fftfreq(nky)*nky/(self.phig[-1]/(2.0*np.pi)) )
        
        return Phi3d_time, time_arr, kx, ky
    
    def load_pres_data(self, n: int, n1: int, dn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load phi field data over time.
        
        Args:
            n: Number of time steps to load
            n1: First time index
            dn: Time step interval
            
        Returns:
            Pressure: Complex array of shape (nmode, n)
            time_arr: Time values
        """
        
        Pres3d_time = None
        time_arr = np.zeros(n)
        
        for ni in tqdm(range(n), desc="Processing time steps", unit="step"):
            itime = n1 + ni * dn
            pres3d, tdiag = mylib.read_data(
                str(self.dirn), 'Pperp_GC_3D', 'time_diag', t1=itime
            )

            pres3d_selected = pres3d[:, :, self.xind]
            # Shift to ballooning representation
            data_fft =np.zeros_like(pres3d_selected, dtype=np.complex128)
            
            for j in range(np.shape(pres3d)[0]):
                databal = mylib.interp2_bal(
                    self.bal_ang[:, self.xind], 
                    pres3d_selected[j, :, :]
                )
                data_fft[j, :, :] = np.fft.fft(databal, axis=0)
                
            data_fft = np.fft.fft(data_fft, axis=0)
            
            data_2D = np.sum(self.rg* data_fft[:, :, :], axis=2)
            
            if Pres3d_time is None:
                Pres3d_time = np.zeros((n, data_2D.shape[0], data_2D.shape[1]), dtype=np.complex128)
            
            Pres3d_time[ni, :, :] = np.fft.fftshift(data_2D)
            time_arr[ni] = tdiag[0]
        
        nky, nkx = data_2D.shape
        kx = np.fft.fftshift(np.fft.fftfreq(nkx)*nkx )
        ky = np.fft.fftshift(np.fft.fftfreq(nky)*nky/(self.phig[-1]/(2.0*np.pi)) )
        
        return Pres3d_time, time_arr, kx, ky
    
    def load_data(self, dtype: str, n: int, n1: int, dn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load phi field data over time.
        
        Args:
            n: Number of time steps to load
            n1: First time index
            dn: Time step interval
            
        Returns:
            Pressure: Complex array of shape (nmode, n)
            time_arr: Time values
        """
        
        Pres3d_time = None
        time_arr = np.zeros(n)
        
        for ni in tqdm(range(n), desc="Processing time steps", unit="step"):
            itime = n1 + ni * dn
            tdiag = mylib.read_data(str(self.dirn), 'time_diag', t1=itime)
            pres3d = process_data(dtype,  str(self.dirn), itime, 0, mylib)
            pres3d_selected = pres3d[:, :, self.xind]
            # Shift to ballooning representation
            data_fft =np.zeros_like(pres3d_selected, dtype=np.complex128)
            
            for j in range(np.shape(pres3d)[0]):
                databal = mylib.interp2_bal(
                    self.bal_ang[:, self.xind], 
                    pres3d_selected[j, :, :]
                )
                data_fft[j, :, :] = np.fft.fft(databal, axis=0)
                
            data_fft = np.fft.fft(data_fft, axis=0)
            # data_fft[:, 0, :] = 0.0  # Remove DC component
            
            data_2D = np.sum(self.rg* data_fft[:, :, :], axis=2)
            
            if Pres3d_time is None:
                Pres3d_time = np.zeros((n, data_2D.shape[0], data_2D.shape[1]), dtype=np.complex128)
            
            Pres3d_time[ni, :, :] = np.fft.fftshift(data_2D)
            time_arr[ni] = tdiag[0]
        
        nky, nkx = data_2D.shape
        kx = np.fft.fftshift(np.fft.fftfreq(nkx)*nkx )
        ky = np.fft.fftshift(np.fft.fftfreq(nky)*nky/(self.phig[-1]/(2.0*np.pi)) )
        
        return Pres3d_time, time_arr, kx, ky
    
    
    def create_colorbar_callback(self, fig, ax, im, cbar, Pxx, bins, freqs):
        """
        Create callback function for dynamic colorbar updates on zoom/pan.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axis
            im: Image object
            cbar: Colorbar object
            Pxx: Power spectral density data
            bins: Time bins
            freqs: Frequency bins
        """
        def update_colorbar(event):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Find indices within current view
            idx_t = np.where((bins >= xlim[0]) & (bins <= xlim[1]))[0]
            idx_f = np.where(
                (freqs >= ylim[0]/self.freq_factor) & 
                (freqs <= ylim[1]/self.freq_factor)
            )[0]
            
            if idx_t.size > 0 and idx_f.size > 0:
                # Extract subset of data
                Pxx_subset = Pxx[
                    np.min(idx_f):np.max(idx_f)+1, 
                    np.min(idx_t):np.max(idx_t)+1
                ]
                
                # Update colorbar limits based on percentiles
                vmin = np.percentile(Pxx_subset[Pxx_subset > 0], 1)
                vmax = np.percentile(Pxx_subset, 99)
                
                im.set_norm(colors.LogNorm(vmin=vmin, vmax=vmax))
                cbar.update_normal(im)
                fig.canvas.draw_idle()
                
                print(f"Colorbar range: [{vmin:.2e}, {vmax:.2e}]")
        
        ax.callbacks.connect('xlim_changed', update_colorbar)
        ax.callbacks.connect('ylim_changed', update_colorbar)
    
    def create_spectrogram(
        self, 
        Phi: np.ndarray, 
        time_arr: np.ndarray
    ) -> None:
        """
        Create and display spectrograms for all modes.
        
        Args:
            Phi: Complex phi field data of shape (nmode, n)
            time_arr: Time values
        """
        # Calculate time step and window size
        dt = np.mean(np.diff(time_arr))
        win_size = self.make_odd(time_arr.size // 8)
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            self.nmode, 1, 
            figsize=(10, self.nmode * 3.5)
        )
        if self.nmode == 1:
            axes = [axes]
        
        # Create spectrogram for each mode
        for i, ax in enumerate(axes):
            Phi_mag = np.abs(Phi[i, :])
            
            # Compute spectrogram
            Pxx, freqs, bins, _ = ax.specgram(
                Phi_mag, 
                NFFT=win_size, 
                Fs=1/dt, 
                noverlap=win_size * 9 // 10
            )
            
            # Remove default specgram image
            ax.images[-1].remove()
            
            # Create custom imshow with log normalization
            extent = [
                time_arr.min(), time_arr.max(), 
                freqs.min() * self.freq_factor, 
                freqs.max() * self.freq_factor
            ]
            
            im = ax.imshow(
                Pxx,
                extent=extent,
                aspect='auto',
                origin='lower',
                interpolation='nearest',
                cmap='jet',
                norm=colors.LogNorm(
                    vmin=np.percentile(Pxx[Pxx > 0], 1), 
                    vmax=np.percentile(Pxx, 99)
                )
            )
            
            # Configure axis
            ax.tick_params(axis='both', labelsize=12)
            ax.set_ylabel(r'Frequency [$\omega_i$]', fontsize=14)
            ax.set_xlabel(r'Time [$\omega_c^{-1}$]', fontsize=14)
            # ax.set_title(f'Mode {i + self.n0}', fontsize=14, pad=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Power', fontsize=14)
            
            # Set initial y-limits
            ax.set_ylim(0.5, 5.0)
            
            # Add dynamic colorbar callback
            self.create_colorbar_callback(fig, ax, im, cbar, Pxx, bins, freqs)
        
        # Adjust layout
        fig.tight_layout()
        plt.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.10)
        
        plt.show()

    def plot_freq_spectrum(
        self,
        Phi: np.ndarray,
        time_arr: np.ndarray,
        kx: np.ndarray,
        ky: np.ndarray,
        method: str = 'peak',
        freq_range: Tuple[float, float] = None,
        save_path: str = None
        ):
        """
        Plot 2D frequency spectrum showing representative frequency for each (kx, ky) mode.
        
        Args:
            Phi: Complex array of shape (n_time, nky, nkx)
            time_arr: Time values
            kx: Wavenumber array in x direction
            ky: Wavenumber array in y direction
            method: Method to compute representative frequency
                    'peak': Frequency with maximum power
                    'weighted': Power-weighted mean frequency
                    'centroid': Spectral centroid
            freq_range: (freq_min, freq_max) to consider, None for full range
            save_path: Path to save the figure
        """
        
        n_time, nky, nkx = Phi.shape
        dt = time_arr[1] - time_arr[0]
        fs = 1.0 / dt
        
        # Initialize array to store representative frequencies
        freq_map = np.zeros((nky, nkx))
        power_map = np.zeros((nky, nkx))
        
        print("Computing frequency spectrum for each mode...")
        
        # Compute FFT for each (kx, ky) mode
        for iky in range(nky):
            for ikx in range(nkx):
                # Get time series for this mode
                mode_timeseries = Phi[:, iky, ikx]
                
                # Compute power spectral density
                freqs, psd = signal.periodogram(
                    mode_timeseries,
                    fs=fs,
                    window='hann',
                    scaling='density'
                )
                
                # Apply frequency range filter if specified
                if freq_range is not None:
                    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                    freqs_filtered = freqs[freq_mask]
                    psd_filtered = psd[freq_mask]
                else:
                    freqs_filtered = freqs
                    psd_filtered = psd
                
                # Skip if no valid frequencies
                if len(freqs_filtered) == 0:
                    continue
                
                # Compute representative frequency based on method
                if method == 'peak':
                    # Frequency with maximum power
                    peak_idx = np.argmax(np.abs(psd_filtered))
                    freq_map[iky, ikx] = freqs_filtered[peak_idx]
                    power_map[iky, ikx] = np.abs(psd_filtered[peak_idx])
                    
                elif method == 'weighted':
                    # Power-weighted mean frequency
                    total_power = np.sum(np.abs(psd_filtered))
                    if total_power > 0:
                        freq_map[iky, ikx] = np.sum(
                            freqs_filtered * np.abs(psd_filtered)
                        ) / total_power
                        power_map[iky, ikx] = total_power
                        
                elif method == 'centroid':
                    # Spectral centroid
                    total_power = np.sum(np.abs(psd_filtered))
                    if total_power > 0:
                        freq_map[iky, ikx] = np.sum(
                            freqs_filtered * np.abs(psd_filtered)
                        ) / total_power
                        power_map[iky, ikx] = total_power
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Frequency map
        im1 = axes[0].pcolormesh(
            kx, ky, freq_map,
            shading='auto',
            cmap='bwr'
        )
        axes[0].set_xlabel('kx (mode number)', fontsize=12)
        axes[0].set_ylabel('ky (mode number)', fontsize=12)
        axes[0].set_title(f'Representative Frequency Map ({method})', fontsize=14)
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Frequency [a.u.]', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Power map (log scale)
        power_map_plot = np.log10(power_map + 1e-20)  # Avoid log(0)
        im2 = axes[1].pcolormesh(
            kx, ky, power_map_plot,
            shading='auto',
            cmap='viridis'
        )
        axes[1].set_xlabel('kx (mode number)', fontsize=12)
        axes[1].set_ylabel('ky (mode number)', fontsize=12)
        axes[1].set_title('Power Spectrum (log scale)', fontsize=14)
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('log₁₀(Power)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return freq_map, power_map


    def plot_freq_spectrum_with_dispersion(
        self,
        Phi: np.ndarray,
        time_arr: np.ndarray,
        kx: np.ndarray,
        ky: np.ndarray,
        dispersion_func=None,
        method: str = 'peak',
        save_path: str = None
    ):
        """
        Plot frequency spectrum with optional theoretical dispersion relation overlay.
        
        Args:
            Phi: Complex array of shape (n_time, nky, nkx)
            time_arr: Time values
            kx: Wavenumber array in x direction
            ky: Wavenumber array in y direction
            dispersion_func: Function that takes (kx, ky) and returns theoretical frequency
                            e.g., lambda kx, ky: np.sqrt(kx**2 + ky**2)
            method: Method to compute representative frequency
            save_path: Path to save the figure
        """
        # Get frequency and power maps
        freq_map, power_map = self.plot_freq_spectrum(
            Phi, time_arr, kx, ky, method=method, save_path=None
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot frequency map
        im = ax.pcolormesh(
            kx, ky, freq_map,
            shading='auto',
            cmap='jet',
            alpha=0.8
        )
        
        # Overlay theoretical dispersion if provided
        if dispersion_func is not None:
            KX, KY = np.meshgrid(kx, ky)
            freq_theory = dispersion_func(KX, KY)
            
            # Plot contours of theoretical dispersion
            contours = ax.contour(
                kx, ky, freq_theory,
                levels=10,
                colors='white',
                linewidths=1.5,
                alpha=0.7
            )
            ax.clabel(contours, inline=True, fontsize=8)
        
        ax.set_xlabel('kx (mode number)', fontsize=12)
        ax.set_ylabel('ky (mode number)', fontsize=12)
        ax.set_title(f'Frequency Spectrum with Dispersion Relation ({method})', fontsize=14)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frequency [a.u.]', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return freq_map, power_map        
        
    def create_mode_integrated_spectrogram(
        self, 
        Phi: np.ndarray, 
        time_arr: np.ndarray,
        kx_range: Tuple[float, float] = None,
        ky_range: Tuple[float, float] = None,
        weight_by_amplitude: bool = True
    ):
        """
        Create spectrogram by integrating over selected (kx, ky) modes.
        
        Args:
            Phi: Complex array of shape (n_time, nky, nkx)
            time_arr: Time values
            kx_range: Mode range in x
            ky_range: Mode range in y
            weight_by_amplitude: Weight modes by their mean amplitude
        """
        from scipy import signal
        
        # Select mode ranges
        if kx_range is not None:
            kx_mask = (self.kx >= kx_range[0]) & (self.kx <= kx_range[1])
        else:
            kx_mask = np.ones(len(self.kx), dtype=bool)
        
        if ky_range is not None:
            ky_mask = (self.ky >= ky_range[0]) & (self.ky <= ky_range[1])
        else:
            ky_mask = np.ones(len(self.ky), dtype=bool)
        
        # Extract and process selected modes
        Phi_selected = Phi[:, ky_mask, :][:, :, kx_mask]
        
        if weight_by_amplitude:
            # Weight by time-averaged amplitude
            weights = np.mean(np.abs(Phi_selected), axis=0)
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Create weighted time series (preserve phase)
            weighted_timeseries = np.sum(
                Phi_selected * weights[np.newaxis, :, :], 
                axis=(1, 2)
            )
        else:
            # Simple sum over modes
            weighted_timeseries = np.sum(Phi_selected, axis=(1, 2))
        
        # Compute spectrogram
        dt = time_arr[1] - time_arr[0]
        fs = 1.0 / dt
        
        tlen = len(time_arr)

        if tlen < 256:
            # 짧은 시계열: 가능한 한 사용
            nperseg = max(64, tlen // 2)
            noverlap = (nperseg // 10)*9
            
        elif tlen < 1024:
            # 중간 길이: 균형잡힌 해상도
            nperseg = 128
            noverlap = 124 # (nperseg // 10)*9  # 75% overlap
            
        else:
            # 긴 시계열: 더 좋은 주파수 해상도
            nperseg = 256
            noverlap = (nperseg // 100)*99  # 75% overlap

        # Use power of 2 for efficient FFT
        nperseg = 2**int(np.log2(nperseg))
        
        f, t, Sxx = signal.spectrogram(
            weighted_timeseries,
            fs=fs,
            nperseg=nperseg, # min(256, len(time_arr)//4),
            noverlap=noverlap, # min(128, len(time_arr)//8),
            window='hann',
            return_onesided=False,  # Explicitly handle complex input
            mode='complex'  # Return complex spectrogram   
        )
        
        print(f'Estimated kx range: {self.kx[kx_mask].min()} to {self.kx[kx_mask].max()}')
        print(f'Estimated ky range: {self.ky[ky_mask].min()} to {self.ky[ky_mask].max()}')

        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)

        Sxx[np.where(np.abs(Sxx) < 1e-20)] = 1e-20  # Avoid log(0)
        
        Sxx_dB = np.abs(Sxx)
        # Sxx_dB = 10 * np.log10(np.abs(Sxx))
        vmax = np.percentile(Sxx_dB, 100)
        vmin = np.percentile(Sxx_dB, 1)
        # Plot
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(
            t + time_arr[0],  # Shift to actual time
            f, 
            Sxx_dB,  # dB scale
            vmin=vmin,
            vmax=vmax,
            shading='gouraud',
            cmap='jet'
        )
        plt.ylabel('Frequency [a.u.]')
        plt.xlabel('Time')
        plt.title(f'Spectrogram (kx: {kx_range}, ky: {ky_range})')
        plt.colorbar(label='Power [dB]')
        plt.ylim([-fs/3*0, fs/4])  # Nyquist limit
    
        return f, t, Sxx
    
    def run(self, n: int, n1: int, dn: int) -> None:
        """
        Run the complete analysis pipeline.
        
        Args:
            n: Number of time steps
            n1: First time index
            dn: Time step interval
        """
        print(f"Loading data from {self.dirn}")
        print(f"Radial range: [{self.min_r}, {self.max_r}]")
        
        # Load data
        Phi, time_arr, self.kx, self.ky = self.load_phi_data(n, n1, dn)
        
        print(f"Time range: [{time_arr.min():.2f}, {time_arr.max():.2f}]")
        print(f"Mode ranges: m: [{self.kx.min():.2f}, {self.kx.max():.2f}], n: [{self.ky.min():.2f}, {self.ky.max():.2f}]")
        print(f"Creating spectrograms...")
        
        # Create frequency of each mode
        # self.plot_freq_spectrum(Phi, time_arr, self.kx, self.ky, method='peak')
        
        def itg_dispersion(kx, ky):
            """Example: Simple dispersion relation"""
            k_perp = np.sqrt(kx**2 + ky**2)
            return 0.1 * k_perp  # Your actual dispersion formula here

        freq_map, power_map = self.plot_freq_spectrum_with_dispersion(
            Phi, time_arr, self.kx, self.ky,
            dispersion_func=itg_dispersion,
            method='peak'
        )
        # Create spectrograms
        for kx_range in [(0, 20), (20, 40), (40, 60)]:
            self.create_mode_integrated_spectrogram(
                Phi, time_arr,
                kx_range=kx_range,
                ky_range=(2*kx_range[0], 2*kx_range[1])
            )
        # self.create_mode_integrated_spectrogram(Phi, time_arr)
        # self.create_spectrogram(Phi, time_arr)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize spectrograms of phi field data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'fn', 
        type=str, 
        help='Path to data directory or file'
    )
    parser.add_argument(
        '-n1', 
        type=int, 
        default=1,
        help='First time index'
    )
    parser.add_argument(
        '-n', 
        type=int, 
        default=100,
        help='Number of time steps'
    )
    parser.add_argument(
        '-dn', 
        type=int, 
        default=10,
        help='Time step interval'
    )
    parser.add_argument(
        '--min_r', 
        type=float, 
        default=0.0,
        help='Starting radial position'
    )
    parser.add_argument(
        '--max_r', 
        type=float, 
        default=1.2,
        help='Ending radial position'
    )
    
    args = parser.parse_args()
    
    # Extract directory name
    dirn = os.path.dirname(args.fn) if os.path.isfile(args.fn) else args.fn
    
    # Create analyzer and run
    analyzer = SpectrogramAnalyzer(
        dirn=dirn,
        min_r=args.min_r,
        max_r=args.max_r,
    )
    
    analyzer.run(n=args.n, n1=args.n1, dn=args.dn)


if __name__ == "__main__":
    main()