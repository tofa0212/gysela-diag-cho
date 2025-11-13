# GYSELA Diagnostic Analysis Tools

Python toolkit for analyzing GYSELA gyrokinetic turbulence simulation data.

## 📋 Overview

This repository provides analysis tools for GYSELA (GYrokinetic SEmi-LAgrangian) simulation outputs, including:

- **Reynolds stress analysis** - Turbulent momentum transport decomposition
- **Temperature profile analysis** - Multi-case temperature comparison and gradient calculations
- **Spectral analysis** - Fourier decomposition and frequency spectra
- **2D plasma visualization** - Poloidal cross-section plots
- **Heat pulse tracking** - Pulse propagation and diffusivity analysis

## 🚀 Quick Start

### Basic Usage

**Reynolds Stress Analysis:**
```python
from Calc_RS import Reynolds_stress_analyzer

# Initialize analyzer
analyzer = Reynolds_stress_analyzer(
    dirname='/path/to/simulation/data',
    spnum=0,          # 0=ions, 1=electrons
    r_min=0.7,        # Minimum normalized radius
    r_max=1.2         # Maximum normalized radius
)

# Plot all Reynolds stress components
analyzer.plot_RS_all(timestep=1500, reduction='mean')

# Plot radial profiles
analyzer.plot_RS_radial_profile(
    timestep=1500,
    min_angle=-30,    # Angle range in degrees
    max_angle=30
)
```

**Temperature Profile Comparison:**
```python
from comp_tem import TemperatureProfileAnalyzer

# Single case analysis
analyzer = TemperatureProfileAnalyzer(
    dirname='/path/to/data',
    r_min=0.7,
    r_max=1.2
)

# Calculate temperature and gradient scale length
Ti = analyzer.calculate_temperature(timestep=1500)
RLT = analyzer.calculate_gradient_scale_length(timestep=1500)

# Multi-case comparison
from comp_tem import MultiCaseComparison

comparison = MultiCaseComparison(
    case_dirs={
        'Case A': '/path/to/case_a',
        'Case B': '/path/to/case_b'
    },
    r_min=0.7,
    r_max=1.2
)

comparison.plot_temperature_comparison(timestep=1500)
```

**Spectral Analysis:**
```python
from Spec_phi_comp import PhiAnalysisPipeline, AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    dirname='/path/to/data',
    angle_min_deg=-30,
    angle_max_deg=30,
    radial_min=0.7,
    radial_max=1.2
)

# Run full analysis pipeline
pipeline = PhiAnalysisPipeline(config)
result = pipeline.run_full_analysis(
    time_start=0,
    time_end=100,
    time_step=1
)

# Plot results
pipeline.plot_contour_and_spectrum()
```

**2D Plasma Visualization:**
```python
from plot_2D import PlasmaViz2D

viz = PlasmaViz2D(dirname='/path/to/data', spnum=0)

# Plot 2D temperature field
viz.plot_temperature_2d(
    timestep=1500,
    component='perp',     # 'par' or 'perp'
    font_size=14
)

# Plot with flux surfaces
viz.plot_with_flux_surfaces(timestep=1500)
```

## 📁 File Structure

### Core Libraries
- **`mylib.py`** - Common utility functions for data loading, physics calculations, and plotting
- **`gysela_name.py`** - File naming conventions and data location mappings

### Analysis Modules

#### Reynolds Stress & Transport
- **`Calc_RS.py`** - Reynolds stress decomposition (ExB, diamagnetic, mixed terms)
- **`Rad_profs.py`** - Radial profile analysis and comparison

#### Temperature Analysis
- **`comp_tem.py`** - Temperature profile analysis with multi-case comparison
- **`plot_init.py`** - Initial equilibrium profile plotting

#### Spectral Analysis
- **`Spec_phi_comp.py`** - Phi fluctuation analysis (real space, production-ready)
- **`spec_phi.py`** - Spectrogram analysis (Fourier space, interactive)
- **`spec_phi_parallel.py`** - Batch processing for large datasets with HDF5 storage

#### Visualization & Utilities
- **`plot_2D.py`** - 2D poloidal plane visualization
- **`comp_data_time.py`** - Multi-variable time series analysis
- **`pulse_analyzer.py`** - Heat pulse propagation tracking
- **`check_data.py`** - Data integrity checking
- **`GYS_dataprocess.py`** - Data preprocessing utilities

## 🛠️ Installation

### Requirements
```bash
pip install numpy scipy h5py matplotlib
```

Or install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Python Version
- Python 3.7+
- Tested with Python 3.9 and 3.11

## 📊 Common Workflow Examples

### 1. Multi-Timestep Analysis
```python
from Calc_RS import Reynolds_stress_analyzer

analyzer = Reynolds_stress_analyzer('/path/to/data', r_min=0.7, r_max=1.2)

# Analyze multiple timesteps
timesteps = [1500, 2000, 2500, 3000]
for t in timesteps:
    analyzer.plot_RS_all(
        timestep=t,
        save_path=f'./output/RS_t{t}.png'
    )
```

### 2. Parameter Scan
```python
from comp_tem import MultiCaseComparison

# Compare different simulation parameters
cases = {
    'R/LT=3': '/data/rlt3',
    'R/LT=6': '/data/rlt6',
    'R/LT=9': '/data/rlt9'
}

comparison = MultiCaseComparison(cases, r_min=0.7, r_max=1.2)
comparison.plot_temperature_comparison(timestep=1500)
comparison.plot_diffusivity_comparison(timestep=1500)
```

### 3. Using Core Library Functions
```python
import mylib

# Load normalized grid
grid = mylib.load_normalized_grid(
    dirname='/path/to/data',
    r_min=0.7,
    r_max=1.2,
    return_mask=True
)
rg = grid['rg']
R0 = grid['R0']

# Load temperature profile
Ti = mylib.load_temperature_profile(
    dirname='/path/to/data',
    timestep=1500
)

# Calculate gradient scale length
RLT = mylib.calc_gradient_scale_length(Ti, rg, R0)

print(f"Peak R/LT: {RLT.max():.2f}")
```

## 📖 Key Concepts

### Coordinate System
- **r** - Normalized radial coordinate (minor radius / minor radius at separatrix)
- **θ** - Poloidal angle
- **φ** - Toroidal angle
- **ψ** - Poloidal flux coordinate

### Species Convention
- `spnum=0` - Ions
- `spnum=1` - Electrons

### Data Organization
GYSELA data is organized as:
```
simulation_directory/
├── sp0/  (ions)
│   ├── phi3D/
│   ├── moments/
│   └── profiles/
└── sp1/  (electrons)
    └── ...
```

### Common Analysis Patterns

**Angular Averaging:**
Many analyses support angle-selective averaging:
```python
# Full poloidal average
result = analyzer.analyze(min_angle=None, max_angle=None)

# Outboard midplane region only
result = analyzer.analyze(min_angle=-30, max_angle=30)
```

**Radial Selection:**
Focus analysis on specific radial regions:
```python
analyzer = SomeAnalyzer(
    dirname='/path/to/data',
    r_min=0.7,    # Core edge
    r_max=1.2     # Edge region
)
```

## 🔧 Configuration

### mylib Functions (v2.1.0)

The `mylib` module provides standardized functions used across all analysis scripts:

**Data Loading:**
- `load_normalized_grid()` - Load and normalize radial grid
- `load_temperature_profile()` - Load T = P/n profiles
- `load_2d_temperature()` - Load 2D anisotropic temperature fields
- `load_geometry_data()` - Load magnetic geometry

**Physics Calculations:**
- `calc_gradient_scale_length()` - Calculate R/L_X
- `calc_exb_velocity_2d()` - ExB drift velocity
- `calc_diamagnetic_velocity_2d()` - Diamagnetic drift
- `calc_reynolds_stress_from_velocities()` - Reynolds stress tensor

**Plotting Utilities:**
- `auto_color_limits()` - Smart color scale determination
- `add_colorbar()` - Consistent colorbar formatting
- `get_angle_indices()` - Angle range selection with wrap-around

### Environment Variables
```bash
# Optional: Set default data directory
export GYSELA_DATA_DIR=/path/to/simulations

# Optional: Set default output directory
export GYSELA_OUTPUT_DIR=./analysis_output
```

## 📝 Command-Line Usage

Most scripts can be run from command line:

```bash
# Reynolds stress analysis
python Calc_RS.py /path/to/data -t 1500 2000 2500 \
    --r_min 0.7 --r_max 1.2 \
    --plot_contour --plot_profile \
    --save ./output

# Temperature comparison
python comp_tem.py /path/to/data -t 1500 \
    --r_min 0.7 --r_max 1.2 \
    --min_angle -30 --max_angle 30 \
    --save ./output

# Spectral analysis
python Spec_phi_comp.py /path/to/data \
    --angle_min -30 --angle_max 30 \
    --radial_min 0.7 --radial_max 1.2
```

Use `--help` for full options:
```bash
python Calc_RS.py --help
```

## 🐛 Troubleshooting

**Common Issues:**

1. **File not found errors:**
   - Check that `gysela_name.py` has correct file paths
   - Verify data directory structure matches expected layout

2. **Memory errors with large datasets:**
   - Use `spec_phi_parallel.py` for batch processing
   - Process timesteps sequentially instead of loading all at once

3. **Import errors:**
   - Ensure all scripts are in the same directory or add to PYTHONPATH
   - Check that all dependencies are installed

## 🔬 Physics Background

### Reynolds Stress
Decomposition into three components:
- **ExB term**: Electrostatic turbulence contribution
- **Diamagnetic term**: Equilibrium pressure gradient effect
- **Mixed term**: Coupling between ExB and diamagnetic flows

### Gradient Scale Length
Normalized inverse gradient: **R/L_X = -R₀ × (dX/dr) / X**

Typical values:
- R/L_T ~ 3-10 (temperature gradient)
- R/L_n ~ 1-3 (density gradient)

## 📚 References

For GYSELA code and physics:
- [GYSELA official documentation]
- Relevant physics publications

## 🤝 Contributing

To add new analysis modules:
1. Use `mylib` common functions where possible
2. Follow existing class structure patterns
3. Add command-line interface with argparse
4. Include docstrings with examples

## 📄 License

[Add license information]

## ✉️ Contact

[Add contact information]

---

**Last Updated:** 2025-11-11
**Version:** 2.1.0 (mylib), 1.0.0 (analysis modules)
