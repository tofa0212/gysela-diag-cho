import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


class HeatPulsePropagationAnalyzer:
    """Analyze heat pulse propagation in plasma"""
    
    def __init__(self, analyzer, equilibrium_time: int):
        """
        Initialize heat pulse analyzer
        
        Args:
            analyzer: RadialProfileAnalyzer instance
            equilibrium_time: Timestep before pulse injection
        """
        self.analyzer = analyzer
        self.equilibrium_time = equilibrium_time
        
        # Get equilibrium profile
        self.T_eq = analyzer.calculate_temperature(equilibrium_time)
        print(f"Equilibrium profile loaded at t={equilibrium_time}")
    
    def get_perturbation(self, timestep: int) -> np.ndarray:
        """
        Get temperature perturbation: ΔT = T(t) - T_eq
        
        Args:
            timestep: Current timestep
            
        Returns:
            Temperature perturbation profile
        """
        T_current = self.analyzer.calculate_temperature(timestep)
        return T_current - self.T_eq
    
    def track_pulse_peak(self, timesteps: list) -> tuple:
        """
        Track the radial position of pulse peak over time
        
        Args:
            timesteps: List of timesteps to analyze
            
        Returns:
            (peak_positions, peak_amplitudes, times)
        """
        peak_positions = []
        peak_amplitudes = []
        times = []
        
        for timestep in timesteps:
            dT = self.get_perturbation(timestep)
            _, tdiag = mylib.read_data(self.analyzer.dirname, 'time_diag', t1=timestep)
            
            # Find peak
            max_idx = np.argmax(np.abs(dT))
            peak_positions.append(self.analyzer.rg[max_idx])
            peak_amplitudes.append(dT[max_idx])
            times.append(tdiag[0])
        
        return np.array(peak_positions), np.array(peak_amplitudes), np.array(times)
    
    def track_pulse_front(self, timesteps: list, threshold: float = 0.1) -> tuple:
        """
        Track the leading edge of pulse (where ΔT crosses threshold)
        
        Args:
            timesteps: List of timesteps
            threshold: Threshold for pulse detection (relative to max)
            
        Returns:
            (front_positions, times)
        """
        front_positions = []
        times = []
        
        for timestep in timesteps:
            dT = self.get_perturbation(timestep)
            _, tdiag = mylib.read_data(self.analyzer.dirname, 'time_diag', t1=timestep)
            
            # Normalize by maximum
            max_dT = np.max(np.abs(dT))
            if max_dT < 1e-10:
                continue
            
            dT_norm = dT / max_dT
            
            # Find where perturbation crosses threshold (propagation front)
            # Looking inward (assuming pulse injected from edge)
            crossing_indices = np.where(dT_norm > threshold)[0]
            
            if len(crossing_indices) > 0:
                # Take innermost crossing point
                front_idx = crossing_indices[-1]
                front_positions.append(self.analyzer.rg[front_idx])
                times.append(tdiag[0])
        
        return np.array(front_positions), np.array(times)
    
    def calculate_propagation_velocity(self, timesteps: list, 
                                      method: str = 'peak') -> tuple:
        """
        Calculate pulse propagation velocity
        
        Args:
            timesteps: List of timesteps
            method: 'peak' or 'front'
            
        Returns:
            (velocities, radii, times)
        """
        if method == 'peak':
            positions, _, times = self.track_pulse_peak(timesteps)
        elif method == 'front':
            positions, times = self.track_pulse_front(timesteps)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate velocity: v = dr/dt
        velocities = np.gradient(positions, times)
        
        return velocities, positions, times
    
    def plot_perturbation_evolution(self, timesteps: list, 
                                   font_size: int = 12,
                                   save_path: str = None):
        """
        Plot perturbation profiles over time
        
        Args:
            timesteps: List of timesteps
            font_size: Font size
            save_path: Save path
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))
        
        for timestep, color in zip(timesteps, colors):
            dT = self.get_perturbation(timestep)
            _, tdiag = mylib.read_data(self.analyzer.dirname, 'time_diag', t1=timestep)
            
            ax.plot(self.analyzer.rg, dT, 
                   label=f't={mylib.sci_note(tdiag[0])}',
                   color=color, linewidth=2)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('r/a', size=font_size+2)
        ax.set_ylabel(r'$\Delta T$ (vs equilibrium)', size=font_size+2)
        ax.set_title('Heat Pulse Perturbation', size=font_size+4)
        ax.legend(fontsize=font_size-2)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, ax
    
    def plot_pulse_trajectory(self, timesteps: list, 
                            method: str = 'peak',
                            font_size: int = 12,
                            save_path: str = None):
        """
        Plot pulse position vs time (trajectory)
        
        Args:
            timesteps: List of timesteps
            method: 'peak' or 'front'
            font_size: Font size
            save_path: Save path
        """
        if method == 'peak':
            positions, amplitudes, times = self.track_pulse_peak(timesteps)
        elif method == 'front':
            positions, times = self.track_pulse_front(timesteps)
            amplitudes = None
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Position vs time
        axes[0].plot(times, positions, 'o-', linewidth=2, markersize=6)
        axes[0].set_xlabel('Time', size=font_size+2)
        axes[0].set_ylabel('Radial Position (r/a)', size=font_size+2)
        axes[0].set_title(f'Pulse Trajectory ({method})', size=font_size+4)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(labelsize=font_size)
        
        # Velocity
        if len(positions) > 1:
            velocities = np.gradient(positions, times)
            axes[1].plot(times, velocities, 's-', linewidth=2, markersize=6, color='red')
            axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
            axes[1].set_xlabel('Time', size=font_size+2)
            axes[1].set_ylabel('Propagation Velocity', size=font_size+2)
            axes[1].set_title('Pulse Velocity', size=font_size+4)
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(labelsize=font_size)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, axes


class HeatPulseSpaceTimeAnalyzer:
    """2D space-time analysis of heat pulse"""
    
    def __init__(self, analyzer, equilibrium_time: int):
        self.analyzer = analyzer
        self.equilibrium_time = equilibrium_time
        self.T_eq = analyzer.calculate_temperature(equilibrium_time)
    
    def create_spacetime_diagram(self, timesteps: list,
                                font_size: int = 12,
                                save_path: str = None):
        """
        Create 2D space-time diagram of perturbation
        
        Args:
            timesteps: List of timesteps
            font_size: Font size
            save_path: Save path
        """
        n_times = len(timesteps)
        n_radii = len(self.analyzer.rg)
        
        dT_2d = np.zeros((n_times, n_radii))
        times = np.zeros(n_times)
        
        for i, timestep in enumerate(timesteps):
            T_current = self.analyzer.calculate_temperature(timestep)
            dT_2d[i, :] = T_current - self.T_eq
            
            _, tdiag = mylib.read_data(self.analyzer.dirname, 'time_diag', t1=timestep)
            times[i] = tdiag[0]
        
        # Create meshgrid
        R, T = np.meshgrid(self.analyzer.rg, times)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Symmetric colorscale around zero
        vmax = np.max(np.abs(dT_2d))
        
        contour = ax.pcolormesh(R, T, dT_2d, 
                               cmap='RdBu_r', 
                               vmin=-vmax, vmax=vmax,
                               shading='gouraud')
        
        # Add contour lines to show propagation
        levels = np.linspace(-vmax, vmax, 11)
        cs = ax.contour(R, T, dT_2d, levels=levels, 
                       colors='black', alpha=0.3, linewidths=0.5)
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(r'$\Delta T$', size=font_size+2)
        cbar.ax.tick_params(labelsize=font_size)
        
        ax.set_xlabel('r/a', size=font_size+2)
        ax.set_ylabel('Time', size=font_size+2)
        ax.set_title('Heat Pulse Space-Time Diagram', size=font_size+4)
        ax.tick_params(labelsize=font_size)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, ax, dT_2d
    
    if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze heat pulse propagation')
    parser.add_argument('fn', type=str, help='Data directory')
    parser.add_argument('-eq', '--equilibrium_time', type=int, required=True,
                       help='Equilibrium timestep (before pulse)')
    parser.add_argument('-t', '--timesteps', nargs='+', type=int, required=True,
                       help='Timesteps to analyze')
    parser.add_argument('--method', type=str, default='peak',
                       choices=['peak', 'front'],
                       help='Tracking method')
    parser.add_argument('--spacetime', action='store_true',
                       help='Create space-time diagram')
    parser.add_argument('--save', type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize
    analyzer = RadialProfileAnalyzer(args.fn)
    pulse_analyzer = HeatPulsePropagationAnalyzer(analyzer, args.equilibrium_time)
    
    # Analyze
    pulse_analyzer.plot_perturbation_evolution(args.timesteps)
    pulse_analyzer.plot_pulse_trajectory(args.timesteps, method=args.method)
    
    if args.spacetime:
        st_analyzer = HeatPulseSpaceTimeAnalyzer(analyzer, args.equilibrium_time)
        st_analyzer.create_spacetime_diagram(args.timesteps)