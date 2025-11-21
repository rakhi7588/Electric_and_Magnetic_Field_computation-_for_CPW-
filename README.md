# Coplanar Waveguide (CPW) Field Visualization

Analytical electromagnetic field solver for **coplanar waveguide (CPW)** structures based on the formulas from **Rainee Simons' "Coplanar Waveguide Circuits, Components, and Systems"** (Equations 2.127–2.137).

This Python script computes and visualizes the **electric (E)** and **magnetic (H)** field distributions in the cross-sectional plane (y-z) of a CPW transmission line, covering both the **air region** (z ≤ 0) and the **substrate region** (0 < z ≤ h₁).

## Features

- **Analytical field calculation** using modal expansion (odd modes, quasi-TEM approximation)
- Separate field formulas for:
  - **Air side** (z ≤ 0): Ey, Ez, Hx, Hy, Hz
  - **Substrate side** (0 < z ≤ h₁): Ex, Ey, Ez, Hx, Hy, Hz
- **Streamline visualization** of electric and magnetic field lines
- Configurable CPW geometry (center strip width S, slot width W, substrate thickness h₁)
- Supports arbitrary substrate permittivity (εᵣ)

## Theory

The code implements the **quasi-TEM mode** field solutions for a CPW structure:

- **Air region**: Only transverse electric field component **Ey** is dominant; **Ex = 0**
- **Substrate region**: All six field components (Ex, Ey, Ez, Hx, Hy, Hz) are present
- Fields decay exponentially away from the slot
- Modal coefficients **Fn**, **Fn1**, **γn**, **γn1**, **rn**, **qn** govern field behavior

### Key Parameters

From the book example (page 61):

- Frequency: **3 GHz**
- Substrate permittivity: **εᵣ = 16** (e.g., alumina)
- Normalized geometry:
  - h₁/λ = 0.07
  - S/h₁ = 1.0
  - W/h₁ = 0.4

## Code Structure

### Main Functions

#### Modal Coefficients
- `compute_v_u()` – auxiliary waveguide parameters (Eq. 2.140)
- `compute_gamma_n()` – decay constant in air (γn)
- `compute_gamma_n1()` – decay constant in substrate (γn1)
- `compute_Fn()`, `compute_Fn1()` – modal coefficients (Eq. 2.138, 2.139)
- `compute_rn_qn()` – boundary condition parameters (Eq. 2.141, 2.142)

#### Electric Field Components
- **Air side**: `Ey_air()`, `Ez_air()` (Eq. 2.127, 2.128)
- **Substrate side**: `Ex_substrate()`, `Ey_substrate()`, `Ez_substrate()` (Eq. 2.132–2.134)

#### Magnetic Field Components
- **Air side**: `Hx_air()`, `Hy_air()`, `Hz_air()` (Eq. 2.129–2.131)
- **Substrate side**: `Hx_substrate()`, `Hy_substrate()`, `Hz_substrate()` (Eq. 2.135–2.137)

### Visualization

The script generates two **streamline plots**:

1. **Electric field lines** (E-field) – colored by magnitude |E|
2. **Magnetic field lines** (H-field) – colored by magnitude |H|

Both plots show:
- Air-substrate interface (solid black line at z = 0)
- Bottom of substrate (dashed gray line at z = h₁)
- Slot boundaries (vertical black lines)

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
