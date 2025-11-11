from pathlib import Path
import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import os
from typing import Tuple, Optional
import mylib

try:
    from GYS_dataprocess import process_data
except ImportError:
    print("Warning: GYS_dataprocess module not found. Using basic data loading only.")
    
    def process_data(dtype: str, dirname: str, t1: int, spnum: int, mylib):
        """Fallback function when GYS_dataprocess is not available"""
        return mylib.read_data(dirname, dtype, t1=t1, spnum=spnum)

class SpectrogramAnalyzer:
    """Analyzer for creating and managing spectrograms with dynamic colorbar updates."""
    
    def __init__(self, dirn: str, min_r: float = 0.7, 
                 max_r: float = 1.1):
        """
        Initialize the analyzer.
        
        Args:
            dirn: Directory containing data files
            min_r: Minimum radial position
            max_r: Maximum radial position
            mylib: Library with read_data and interp2_bal functions
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
    def _process_single_timestep(itime: int, dirn: str, dtype: str, xind: np.ndarray, 
                                 bal_ang: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process a single time step (for parallel execution).
        
        Args:
            itime: Time index
            dirn: Directory path
            dtype: Data type (e.g., 'phi', 'dens')
            xind: Radial indices to select
            bal_ang: Ballooning angles
            mylib: Library with read_data and interp2_bal functions
        
        Returns:
            data_fft_shifted: Complex array of shape (nky, nkx, n_radial)
            time: Time value
        """
        tdiag = mylib.read_data(str(dirn), 'time_diag', t1=itime)
        pres3d = process_data(dtype, str(dirn), itime, 0, mylib)
        pres3d_selected = pres3d[:, :, xind]
        
        # Shift to ballooning representation and FFT
        data_fft = np.zeros_like(pres3d_selected, dtype=np.complex128)
        
        for j in range(pres3d.shape[0]):
            databal = mylib.interp2_bal(bal_ang[:, xind], pres3d_selected[j, :, :])
            data_fft[j, :, :] = np.fft.fft(databal, axis=0)
        
        # FFT in ky direction
        data_fft = np.fft.fft(data_fft, axis=0)
        
        # Shift for centered frequencies
        data_fft_shifted = np.fft.fftshift(data_fft, axes=(0, 1))
        
        return data_fft_shifted, tdiag[0]
    
    
    def load_data_full_radial(self, dtype: str, n: int, n1: int, dn: int, 
                              output_file: str = 'fft_data_full.h5',
                              n_workers: Optional[int] = None) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load field data over time for ALL radial positions (no averaging).
        Saves to HDF5 file for memory efficiency.
        
        Args:
            dtype: Data type to load (e.g., 'phi', 'dens')
            n: Number of time steps to load
            n1: First time index
            dn: Time step interval
            output_file: Output HDF5 filename
            n_workers: Number of parallel workers (default: cpu_count())
            
        Returns:
            output_file: Path to saved HDF5 file
            time_arr: Time values
            kx: kx wavenumbers
            ky: ky wavenumbers
            
        HDF5 structure:
            - 'data': Complex array of shape (n_time, nky, nkx, n_radial)
            - 'time': Time array
            - 'kx': kx wavenumbers
            - 'ky': ky wavenumbers
            - 'radial_grid': Radial positions
            - 'radial_indices': Original radial indices (xind)
        """
        
        if n_workers is None:
            n_workers = min(cpu_count(), n)  # Don't use more workers than timesteps
        
        print(f"Processing {n} timesteps with {n_workers} workers...")
        print(f"Radial range: {self.rg.min():.3f} to {self.rg.max():.3f}")
        
        # Get dimensions from first timestep
        itime_first = n1
        pres3d_first = process_data(dtype, str(self.dirn), itime_first, 0, mylib)
        pres3d_selected = pres3d_first[:, :, self.xind]
        
        nky, ntheta, n_radial = pres3d_selected.shape
        nkx = ntheta  # After FFT in theta direction
        
        print(f"Data shape: nky={nky}, nkx={nkx}, n_radial={n_radial}, n_time={n}")
        
        # Prepare kx and ky arrays
        kx = np.fft.fftshift(np.fft.fftfreq(nkx) * nkx)
        ky = np.fft.fftshift(np.fft.fftfreq(nky) * nky / (self.phig[-1] / (2.0 * np.pi)))
        
        # Create HDF5 file
        with h5py.File(output_file, 'w') as f:
            # Create dataset with chunking for efficient I/O
            dset = f.create_dataset(
                'data',
                shape=(n, nky, nkx, n_radial),
                dtype=np.complex128,
                chunks=(1, nky, nkx, n_radial),  # Chunk by timestep
                compression='gzip',
                compression_opts=4
            )
            
            time_dset = f.create_dataset('time', shape=(n,), dtype=np.float64)
            f.create_dataset('kx', data=kx)
            f.create_dataset('ky', data=ky)
            f.create_dataset('radial_grid', data=self.rg)
            f.create_dataset('radial_indices', data=self.xind)
            
            # Store metadata
            f.attrs['n_time'] = n
            f.attrs['nky'] = nky
            f.attrs['nkx'] = nkx
            f.attrs['n_radial'] = n_radial
            f.attrs['dtype'] = dtype
            f.attrs['n1'] = n1
            f.attrs['dn'] = dn
            f.attrs['min_r'] = self.min_r
            f.attrs['max_r'] = self.max_r
            f.attrs['R0'] = self.R0
            f.attrs['inv_rho'] = self.inv_rho
            
            # Prepare timestep indices
            time_indices = [n1 + ni * dn for ni in range(n)]
            
            # Parallel processing
            process_func = partial(
                self._process_single_timestep,
                dirn=str(self.dirn),
                dtype=dtype,
                xind=self.xind,
                bal_ang=self.bal_ang
            )
            
            with Pool(n_workers) as pool:
                # Process in chunks to show progress
                chunk_size = max(1, n // 100)  # Update progress every 1%
                
                for i, (data_fft_shifted, time_val) in enumerate(
                    tqdm(
                        pool.imap(process_func, time_indices, chunksize=chunk_size),
                        total=n,
                        desc=f"Processing {dtype}",
                        unit="step"
                    )
                ):
                    dset[i, :, :, :] = data_fft_shifted
                    time_dset[i] = time_val
        
        print(f"Data saved to {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1e9:.2f} GB")
        
        # Load time array
        with h5py.File(output_file, 'r') as f:
            time_arr = f['time'][:]
        
        return output_file, time_arr, kx, ky
    
    def load_data_zonal(self, dtype: str, n: int, n1: int, dn: int, 
                              output_file: str = 'fft_data_zonal.h5',
                              n_workers: Optional[int] = None) -> Tuple[str, np.ndarray]:
        """
        Load field data over time for ALL radial positions (no averaging).
        Saves to HDF5 file for memory efficiency.
        
        Args:
            dtype: Data type to load (e.g., 'phi', 'dens')
            n: Number of time steps to load
            n1: First time index
            dn: Time step interval
            output_file: Output HDF5 filename
            n_workers: Number of parallel workers (default: cpu_count())
            
        Returns:
            output_file: Path to saved HDF5 file
            time_arr: Time values
            
        HDF5 structure:
            - 'data': Complex array of shape (n_time, nky, nkx, n_radial)
            - 'time': Time array
            - 'radial_grid': Radial positions
            - 'radial_indices': Original radial indices (xind)
        """
        
        if n_workers is None:
            n_workers = min(cpu_count(), n)  # Don't use more workers than timesteps
        
        print(f"Processing {n} timesteps with {n_workers} workers...")
        print(f"Radial range: {self.rg.min():.3f} to {self.rg.max():.3f}")
        
        # Get dimensions from first timestep
        itime_first = n1
        pres3d_first = process_data(dtype, str(self.dirn), itime_first, 0, mylib)
        pres3d_selected = pres3d_first[:, :, self.xind]
        
        nky, ntheta, n_radial = pres3d_selected.shape
        nkx = ntheta  # After FFT in theta direction
        
        print(f"Data shape: nky={nky}, nkx={nkx}, n_radial={n_radial}, n_time={n}")
        
        kx = np.fft.fftshift(np.fft.fftfreq(nkx) * nkx)
        ky = np.fft.fftshift(np.fft.fftfreq(nky) * nky / (self.phig[-1] / (2.0 * np.pi)))
        
        ikx = np.where(kx == 0)[0][0]
        iky = np.where(ky == 0)[0][0]
        
        # Create HDF5 file
        with h5py.File(output_file, 'w') as f:
            # Create dataset with chunking for efficient I/O
            dset = f.create_dataset(
                'data',
                shape=(n, n_radial),
                dtype=np.complex128,
                chunks=(1, n_radial),  # Chunk by timestep
                compression='gzip',
                compression_opts=4
            )
            
            time_dset = f.create_dataset('time', shape=(n,), dtype=np.float64)
            f.create_dataset('radial_grid', data=self.rg)
            f.create_dataset('radial_indices', data=self.xind)
            
            # Store metadata
            f.attrs['n_time'] = n
            f.attrs['nky'] = nky
            f.attrs['nkx'] = nkx
            f.attrs['n_radial'] = n_radial
            f.attrs['dtype'] = dtype
            f.attrs['n1'] = n1
            f.attrs['dn'] = dn
            f.attrs['min_r'] = self.min_r
            f.attrs['max_r'] = self.max_r
            f.attrs['R0'] = self.R0
            f.attrs['inv_rho'] = self.inv_rho
            
            # Prepare timestep indices
            time_indices = [n1 + ni * dn for ni in range(n)]
            
            # Parallel processing
            process_func = partial(
                self._process_single_timestep,
                dirn=str(self.dirn),
                dtype=dtype,
                xind=self.xind,
                bal_ang=self.bal_ang
            )
            
            with Pool(n_workers) as pool:
                # Process in chunks to show progress
                chunk_size = max(1, n // 100)  # Update progress every 1%
                
                for i, (data_fft_shifted, time_val) in enumerate(
                    tqdm(
                        pool.imap(process_func, time_indices, chunksize=chunk_size),
                        total=n,
                        desc=f"Processing {dtype}",
                        unit="step"
                    )
                ):
                    dset[i, :] = data_fft_shifted[iky, ikx, :]
                    time_dset[i] = time_val
        
        print(f"Data saved to {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1e9:.2f} GB")
        
        # Load time array
        with h5py.File(output_file, 'r') as f:
            time_arr = f['time'][:]
        
        return output_file, time_arr
    
    def load_data(self, dtype: str, n: int, n1: int, dn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load field data over time WITH radial averaging (original method).
        
        Args:
            dtype: Data type to load
            n: Number of time steps to load
            n1: First time index
            dn: Time step interval
            
        Returns:
            Pres3d_time: Complex array of shape (n, nky, nkx)
            time_arr: Time values
            kx: kx wavenumbers
            ky: ky wavenumbers
        """
        
        Pres3d_time = None
        time_arr = np.zeros(n)
        
        for ni in tqdm(range(n), desc=f"Processing {dtype}", unit="step"):
            itime = n1 + ni * dn
            tdiag = mylib.read_data(str(self.dirn), 'time_diag', t1=itime)
            pres3d = process_data(dtype, str(self.dirn), itime, 0, mylib)
            pres3d_selected = pres3d[:, :, self.xind]
            
            # Shift to ballooning representation
            data_fft = np.zeros_like(pres3d_selected, dtype=np.complex128)
            
            for j in range(np.shape(pres3d)[0]):
                databal = mylib.interp2_bal(
                    self.bal_ang[:, self.xind], 
                    pres3d_selected[j, :, :]
                )
                data_fft[j, :, :] = np.fft.fft(databal, axis=0)
                
            data_fft = np.fft.fft(data_fft, axis=0)
            
            # Radial averaging with weights
            data_2D = np.sum(self.rg * data_fft[:, :, :], axis=2)
            
            if Pres3d_time is None:
                Pres3d_time = np.zeros((n, data_2D.shape[0], data_2D.shape[1]), dtype=np.complex128)
            
            Pres3d_time[ni, :, :] = np.fft.fftshift(data_2D)
            time_arr[ni] = tdiag[0]
        
        nky, nkx = data_2D.shape
        kx = np.fft.fftshift(np.fft.fftfreq(nkx) * nkx)
        ky = np.fft.fftshift(np.fft.fftfreq(nky) * nky / (self.phig[-1] / (2.0 * np.pi)))
        
        return Pres3d_time, time_arr, kx, ky
    
    @staticmethod
    def load_fft_data_from_file(filename: str, time_slice: Optional[slice] = None, 
                                radial_slice: Optional[slice] = None):
        """
        Load FFT data from saved HDF5 file.
        
        Args:
            filename: HDF5 file path
            time_slice: Slice object for time dimension (e.g., slice(0, 100))
            radial_slice: Slice object for radial dimension (e.g., slice(10, 20))
            
        Returns:
            data: Complex array (possibly sliced)
            time: Time array
            kx, ky: Wavenumber arrays
            radial_grid: Radial positions
            metadata: Dictionary with additional info
        """
        with h5py.File(filename, 'r') as f:
            # Get slices
            if time_slice is None:
                time_slice = slice(None)
            if radial_slice is None:
                radial_slice = slice(None)
            
            # Load data
            data = f['data'][time_slice, :, :, radial_slice]
            time = f['time'][time_slice]
            kx = f['kx'][:]
            ky = f['ky'][:]
            radial_grid = f['radial_grid'][radial_slice]
            
            # Load metadata
            metadata = {
                'dtype': f.attrs.get('dtype', 'unknown'),
                'min_r': f.attrs.get('min_r', None),
                'max_r': f.attrs.get('max_r', None),
                'R0': f.attrs.get('R0', None),
                'inv_rho': f.attrs.get('inv_rho', None),
            }
            
            # Print info
            print(f"Loaded data shape: {data.shape}")
            print(f"Data type: {metadata['dtype']}")
            print(f"Time range: {time[0]:.2f} to {time[-1]:.2f}")
            print(f"Radial range: {radial_grid[0]:.3f} to {radial_grid[-1]:.3f}")
            
        return data, time, kx, ky, radial_grid, metadata
    
    @staticmethod
    def load_zonal_data_from_file(filename: str, time_slice: Optional[slice] = None, 
                                radial_slice: Optional[slice] = None):
        """
        Load zonal data from saved HDF5 file.

        Args:
            filename: HDF5 file path
            time_slice: Slice object for time dimension (e.g., slice(0, 100))
            radial_slice: Slice object for radial dimension (e.g., slice(10, 20))
            
        Returns:
            data: Complex array (possibly sliced)
            time: Time array
            radial_grid: Radial positions
            metadata: Dictionary with additional info
        """
        with h5py.File(filename, 'r') as f:
            # Get slices
            if time_slice is None:
                time_slice = slice(None)
            if radial_slice is None:
                radial_slice = slice(None)
            
            # Load data
            data = f['data'][time_slice, radial_slice]
            time = f['time'][time_slice]
            radial_grid = f['radial_grid'][radial_slice]
            
            # Load metadata
            metadata = {
                'dtype': f.attrs.get('dtype', 'unknown'),
                'min_r': f.attrs.get('min_r', None),
                'max_r': f.attrs.get('max_r', None),
                'R0': f.attrs.get('R0', None),
                'inv_rho': f.attrs.get('inv_rho', None),
            }
            
            # Print info
            print(f"Loaded data shape: {data.shape}")
            print(f"Data type: {metadata['dtype']}")
            print(f"Time range: {time[0]:.2f} to {time[-1]:.2f}")
            print(f"Radial range: {radial_grid[0]:.3f} to {radial_grid[-1]:.3f}")
            
        return data, time, radial_grid, metadata
    
    @staticmethod
    def compute_radial_average(data: np.ndarray, radial_grid: np.ndarray) -> np.ndarray:
        """
        Compute radial average from full FFT data.
        
        Args:
            data: Complex array of shape (..., n_radial)
            radial_grid: Radial positions/weights
            
        Returns:
            averaged_data: Complex array with radial dimension averaged
        """
        return np.sum(data * radial_grid, axis=-1)