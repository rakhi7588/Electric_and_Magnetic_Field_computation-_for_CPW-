"""
Coplanar Waveguide Field Visualization
Based on Rainee Simons - "Coplanar Waveguide Circuits, Components, and Systems"

AIR SIDE (z ≤ 0): Ey, Ex=0, Ez + Hx, Hy, Hz
SUBSTRATE SIDE (0 ≤ z ≤ h₁): Ex, Ey, Ez + Hx, Hy, Hz
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 3e8  # Speed of light (m/s)
epsilon_0 = 8.854e-12  # Permittivity of free space (F/m)
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
eta_0 = np.sqrt(mu_0 / epsilon_0)  # Impedance of free space ≈ 377 Ω
eta = 376.7 #ohm

# =============================================================================
# CPW GEOMETRY PARAMETERS (from book example, page 61)
# =============================================================================
h1_over_lambda = 0.07  # h1/λ = 0.07 
S_over_h1 = 1.0  # S/h1 = 1
W_over_h1 = 0.4  # W/h1 = 0.4

freq = 3e9  # 3 GHz (from book example)
epsilon_r1 = 16.0  # Substrate permittivity

# Calculate wavelength
lambda_0 = c / freq  # Free space wavelength(λo and λ is same here)
h1 = h1_over_lambda * lambda_0  # Substrate thickness
S = S_over_h1 * h1  # Center strip width
W = W_over_h1 * h1  # Slot width

# Calculate guide wavelength
epsilon_eff = (epsilon_r1 + 1) / 2  # Simplified effective permittivity ; 1 is permittivity of air
lambda_g = lambda_0 / np.sqrt(epsilon_eff)  # Guide wavelength λg = λo/sqrt(eff)


# Voltage excitation
V0 = 1.0  # 1 Volt

# Calculation parameters
b = (S + 2 * W)/2  # Half width for calculation (ask this)
delta = W / b  # Normalized slot parameter

# Wavenumbers
k0 = 2 * np.pi / lambda_0 #(2pi/λo; Free-space wave number)
kg = 2 * np.pi / lambda_g #(2pi/λg; Guided wave number i.e. The wavelength of the wave inside your coplanar waveguide structure; it is always less than λo)
ks = k0 * np.sqrt(epsilon_r1)  # Wavenumber in substrate

# Intrinsic impedances
eta_s = eta_0 / np.sqrt(epsilon_r1)  # Substrate impedance (ηs = ηo/sqrt.er1)

print("="*70)
print(" COMPLETE COPLANAR WAVEGUIDE FIELD SIMULATION")
print(" Based on Rainee Simons Formulas (Equations 2.127-2.137)")
print("="*70)
print(f"Frequency: {freq/1e9:.1f} GHz")
print(f"Substrate εr: {epsilon_r1}")
print(f"Free space wavelength λ₀: {lambda_0*1000:.2f} mm")
print(f"Guide wavelength λg: {lambda_g*1000:.2f} mm")
print(f"Substrate thickness h₁: {h1*1e6:.2f} µm ({h1*1000:.3f} mm)")
print(f"Center strip width S: {S*1e6:.2f} µm")
print(f"Slot width W: {W*1e6:.2f} µm")
print(f"Calculation width b: {b*1e6:.2f} µm")
print(f"Delta δ: {delta:.4f}")
print("="*70)

# =============================================================================
# MODAL COEFFICIENTS (Equations 2.138, 2.139)
# =============================================================================

def compute_v_u(lambda_0, lambda_g, epsilon_r1):
    """
    Compute v and u parameters - Equation 2.140
    v and u are auxiliary waveguide parameters
    """
    ratio = lambda_0 / lambda_g
    
    # v parameter
    if ratio > 1:
        v = np.sqrt(ratio**2 - 1)
    else:
        v = np.sqrt(1 - ratio**2)
    
    # u parameter
    if epsilon_r1 > ratio**2:
        u = np.sqrt(epsilon_r1 - ratio**2)
    else:
        u = np.sqrt(ratio**2 - epsilon_r1)
    
    return v, u


def compute_gamma_n(n, b, lambda_g):
    """
    γn and γn1 are decay constants which are related to how each mode falls off away from the conductors.
    Decay constant for air side (z ≤ 0);
    γn = sqrt((nπ/b)² - (2π/λg)²)
    """
    term = (n * np.pi / b)**2 - (2 * np.pi / lambda_g)**2
    if term > 0:
        return np.sqrt(term)
    else:
        # Return imaginary component if term is negative
        return 1j * np.sqrt(np.abs(term))


def compute_gamma_n1(n, b, lambda_g):
    """
    Decay constant for substrate side (0 < z < h₁)
    γn1 = sqrt((nπ/b)² + (2π/λg)²)
    """
    term = (n * np.pi / b)**2 + (2 * np.pi / lambda_g)**2
    return np.sqrt(term)


def compute_Fn(n, b, lambda_0, lambda_g, epsilon_r1):
    """
    Modal coefficient Fn - Equation 2.138
    Fn = (b*γn)/(nπ) = sqrt(1 + (2bv/nλ)²)
    Returns both parts of the modal coefficient:
    1. Fn_gamma = (b*γn)/(nπ) - the gamma-based part
    2. Fn_sqrt = sqrt(1 + (2bv/nλ)²) - the square root correction factor
    
    """
    v, _ = compute_v_u(lambda_0, lambda_g, epsilon_r1)
    gamma_n = compute_gamma_n(n, b, lambda_g)
    
    gamma_n_real = np.real(gamma_n) if np.iscomplex(gamma_n) else gamma_n
    
    # Part 1: (b*γn)/(nπ)
    Fn_gamma = (b * gamma_n_real) / (n * np.pi)
    
    # Part 2: sqrt(1 + (2bv/nλ)²)
    term = 1 + ((2 * b * v) / (n * lambda_0))**2
    Fn_sqrt = np.sqrt(term)
    
    return Fn_gamma, Fn_sqrt


def compute_Fn1(n, b, lambda_0, lambda_g, epsilon_r1):
    """
    Modal coefficient Fn1 - Equation 2.139
    Fn1 = (b*γn1)/(nπ) = sqrt(1 - (2bu/nλ)²)
    Returns both parts separately for flexibility:
    Returns: (Fn1_gamma, Fn1_sqrt)
    """
    _, u = compute_v_u(lambda_0, lambda_g, epsilon_r1)
    gamma_n1 = compute_gamma_n1(n, b, lambda_g)
    
    Fn1_gamma = (b * gamma_n1) / (n * np.pi)
    
    term_arg = 1 - ((2 * b * u) / (n * lambda_0))**2
    if term_arg > 0:
        Fn1_sqrt = np.sqrt(term_arg)
    else:
        Fn1_sqrt = np.sqrt(np.abs(term_arg))
    
    return Fn1_gamma, Fn1_sqrt
    
    
def compute_rn_qn(n, b, h1, lambda_0, lambda_g, epsilon_r1):
    """
    Compute rn and qn for boundary conditions - Equations 2.141, 2.142
    
    rn = γn1*h1 + tanh⁻¹(Fn1/(εr*Fn))
    qn = γn1*h1 + coth⁻¹(Fn/Fn1)
    
    These are used in substrate field boundary conditions
    Parameters:
    - n: mode index
    - b: full width
    - h1: substrate thickness
    - lambda_0: free-space wavelength
    - lambda_g: guide wavelength
    - epsilon_r: substrate permittivity
    
    Returns:
    - rn, qn: complex values used in field calculations
    
    """
     # Compute modal coefficients
    Fn_gamma, Fn_sqrt = compute_Fn(n, b, lambda_0, lambda_g, epsilon_r1)
    Fn = Fn_gamma  # Or full depending on your use case
    
    Fn1_gamma, Fn1_sqrt = compute_Fn1(n, b, lambda_0, lambda_g, epsilon_r1)
    Fn1 = Fn1_gamma  # Or full
    
    # Compute gamma_n1
    gamma_n1 = compute_gamma_n1(n, b, lambda_g)

    # Safeguard against division by zero or near-zero
    if abs(Fn) < 1e-12:
        Fn = 1e-12
    if abs(Fn1) < 1e-12:
        Fn1 = 1e-12

    # Compute ratio for rn
    ratio_rn = Fn1 / (epsilon_r1 * Fn)
    
    # Compute rn = gamma_n1*h1 + arctanh(ratio_rn)
    if abs(ratio_rn) < 1:
        rn = gamma_n1 * h1 + np.arctanh(ratio_rn)
    else:
        rn = gamma_n1 * h1  # fallback
    
    # Compute ratio for qn
    ratio_qn = Fn / Fn1
    
    # Compute coth inverse: 0.5 * ln((x+1)/(x-1)), valid if |x|>1
    if abs(ratio_qn) > 1:
        qn = gamma_n1 * h1 + 0.5 * np.log((ratio_qn + 1) / (ratio_qn -1))
    else:
        qn = gamma_n1 * h1  # fallback
    
    return rn, qn


# =============================================================================
# ELECTRIC FIELD COMPONENTS - AIR SIDE (z ≤ 0)
# Per book: Only Ey is non-zero, Ex = 0, Ez ≈ 0 (quasi-TEM)
# =============================================================================

def Ey_air(y, z, n_max=7):
    """
    Ey component in air (z ≤ 0) - Equation 2.127
    This is the ONLY non-zero transverse electric field in air
    This product of sine functions and normalizations makes sure that the field components "fit" the shape of the conductor and the slots correctly.
    The field is strongest where the slot is open (where it can "see" the voltage)
    The field is zero (or minimal) at the metal walls, where there should be no tangential electric field
    spatial part describes that how the electric field varies along the vertical direction (y-axis) across the cross-section of the coplanar waveguide rectangle.​For n=1 (the fundamental mode), the sine function starts at zero at y=0, rises to a peak, and returns to zero at y=b
    For larger n, there are more peaks and zero crossings—higher order modes.
    """
    Ey = 0.0
    for n in range(1, n_max, 6):  # odd n only
        gamma_n = compute_gamma_n(n, b, lambda_g)

        numerator = np.sin((n * np.pi )/ 2) * np.sin((n * np.pi * delta )/ 2)
        denominator = (n * np.pi * delta) / 2 if abs(delta) > 1e-12 else 1.0
        bracket_term = numerator / denominator

        spatial = np.sin((n * np.pi * y) / b)
        decay = np.exp(-np.real(gamma_n) * abs(z))

        Ey += (2 * V0 / b) * bracket_term * spatial * decay

    return np.real(Ey)


def Ex_air(y, z, n_max=7):
    """
    Ex component in air (z ≤ 0) - ZERO per book page 59
    "On the air side of the slot the Ey and Ez components of the electric field 
    and Hx, Hy, and Hz component of the magnetic field exist."
    Ex is NOT listed, hence Ex = 0 in air
    """
    return 0.0

def Ez_air(y, z, n_max=7):
    """
    Ez component in air (z ≤ 0) - Equation 2.128
    
    Ez = -(2V0/b) * Σ (1/Fn) * [sin(nπ/2)/(nπδ/2)] * sin(nπδ/2) * cos(nπy/b) * e^(-γn|z|)
    
    Note: First sine is sin(nπ/2) (mode index only)
          Second sine is sin(nπδ/2) (slot width)
    """
    Ez = 0.0
    for n in range(1, n_max, 6):  # odd n only
        gamma_n = compute_gamma_n(n, b, lambda_g)
        Fn_gamma, Fn_sqrt = compute_Fn(n, b, lambda_0, lambda_g, epsilon_r1)
        Fn = Fn_gamma  # Use gamma part
        
        # Slot factor with unique variable names
        slot_numerator = np.sin(n * np.pi * delta / 2) * np.sin(n * np.pi * delta / 2)
        slot_denominator = n * np.pi * delta / 2 if abs(delta) > 1e-12 else 1.0
        slot_factor = slot_numerator / slot_denominator
        
        # Spatial and decay terms
        y_term = np.cos(n * np.pi * y / b)
        z_term = np.exp(-np.real(gamma_n) * abs(z))
        
        # Sum contribution
        Ez += -(2 * V0 / b) * (1 / Fn) * slot_factor * y_term * z_term

    return np.real(Ez)
    
        

# =============================================================================
# MAGNETIC FIELD COMPONENTS - AIR SIDE (z ≤ 0)
# Equations 2.129, 2.130, 2.131
# =============================================================================

def Hx_air(y, z, n_max=7):
    """  Hx component in air (z ≤ 0) - Equation 2.129
    Hx represents how the magnetic field loops around the slot. These loops are strongest near the slot itself.
    The sine and other terms ensure the field meets boundary conditions—it's shaped to fit the width and boundaries of the region
    Exponential Decay: The magnetic field fades away as you move vertically farther from the slot (into air or substrate)
    
    Hx = -j * (2V0)/(η*b) * (λ/λg)² * (2b/λ) * Σ [1 - (λg/λ)²] / (n*Fn) 
         * [sin(nπδ/2)/(nπδ/2)] * sin(nπδ/2) * sin(nπy/b) * e^(-γn|z|)
    
    Note: This is a complex-valued field component due to -j factor"""
    Hx = 0.0 + 0.0j  # Initialize as complex
    
    # Wavelength ratio terms
    lambda_ratio = lambda_0 / lambda_g
    lambda_ratio_sq = lambda_ratio**2
    lambda_g_ratio_sq = (lambda_g / lambda_0)**2
    
    # Prefactor: -j * (2V0)/(η*b) * (λ/λg)² * (2b/λ)
    prefactor = -1j * (2 * V0) / (eta * b) * lambda_ratio_sq * (2 * b / lambda_0)
    
    for n in range(1, n_max, 6):  # odd n only
        gamma_n = compute_gamma_n(n, b, lambda_g)
        Fn_gamma, Fn_sqrt = compute_Fn(n, b, lambda_0, lambda_g, epsilon_r1)
        Fn = Fn_gamma  # Use gamma part
        
        # Avoid division by zero
        if abs(Fn) < 1e-12:
            continue
        
        # Modal coefficient term: [1 - (λg/λ)²] / (n*Fn)
        modal_factor = (1 - lambda_g_ratio_sq) / (n * Fn)
        
        # Slot terms: [sin(nπδ/2)/(nπδ/2)] * sin(nπδ/2)
        sin_term = np.sin(n * np.pi * delta / 2)
        denom_term = n * np.pi * delta / 2 if abs(delta) > 1e-12 else 1.0
        magnetic_slot_factor = (sin_term / denom_term) * sin_term
        
        # Spatial variation: sin(nπy/b)
        y_spatial = np.sin(n * np.pi * y / b)
        
        # Exponential decay: e^(-γn|z|)
        z_decay = np.exp(-np.real(gamma_n) * abs(z))
        
        # Sum contribution
        Hx += prefactor * modal_factor * magnetic_slot_factor * y_spatial * z_decay
    
    return Hx  # Return complex value

def Hy_air(y, z, n_max=7):
    """ Hy component in air (z ≤ 0) - Equation 2.130
    
    Hy = (2V0)/(η*b) * (λ/λg) * Σ (1/Fn) 
         * [sin(nπδ/2)/(nπδ/2)] * sin(nπδ/2) * cos(nπy/b) * e^(-γn|z|)
    
    Note: This is a real-valued field component (no j factor)"""
    Hy = 0.0
    
    # Wavelength ratio term
    lambda_ratio = lambda_0 / lambda_g
    
    # Prefactor: (2V0)/(η*b) * (λ/λg)
    prefactor = (2 * V0) / (eta * b) * lambda_ratio
    
    for n in range(1, n_max, 6):  # odd n only
        gamma_n = compute_gamma_n(n, b, lambda_g)
        Fn_gamma, Fn_sqrt = compute_Fn(n, b, lambda_0, lambda_g, epsilon_r1)
        Fn = Fn_gamma  # Use gamma part
        
        # Avoid division by zero
        if abs(Fn) < 1e-12:
            continue
        
        # Modal coefficient term: 1/Fn
        modal_factor = 1 / Fn
        
        # Slot terms: [sin(nπδ/2)/(nπδ/2)] * sin(nπδ/2)
        sin_term = np.sin(n * np.pi * delta / 2)
        denom_term = n * np.pi * delta / 2 if abs(delta) > 1e-12 else 1.0
        hy_slot_factor = (sin_term / denom_term) * sin_term
        # Spatial variation: cos(nπy/b)
        y_spatial = np.cos(n * np.pi * y / b)
        
        # Exponential decay: e^(-γn|z|)
        z_decay = np.exp(-np.real(gamma_n) * abs(z))
        
        # Sum contribution
        Hy += prefactor * modal_factor * hy_slot_factor * y_spatial * z_decay
    
    return np.real(Hy)  # Return real value

def Hz_air(y, z, n_max=7):
    """Hz component in air (z ≤ 0) - Equation 2.131
    Hz = (2V0)/(η*b) * (λ/λg) * Σ 
         [sin(nπδ/2)/(nπδ/2)] * sin(nπδ/2) * cos(nπy/b) * e^(-γn|z|)
    """
    Hz = 0.0
    
    # Wavelength ratio term
    lambda_ratio = lambda_0 / lambda_g
    
    # Prefactor: (2V0)/(η*b) * (λ/λg)
    prefactor = (2 * V0) / (eta * b) * lambda_ratio
    
    for n in range(1, n_max, 6):  # odd n
        gamma_n = compute_gamma_n(n, b, lambda_g)
        
        sin_term = np.sin(n * np.pi * delta / 2)
        denom_term = n * np.pi * delta / 2 if abs(delta) > 1e-12 else 1.0
        slot_factor = (sin_term / denom_term) * sin_term
        
        y_term = np.cos(n * np.pi * y / b)
        z_term = np.exp(-np.real(gamma_n) * abs(z))
        
        Hz += prefactor * slot_factor * y_term * z_term
    
    return Hz

# =============================================================================
# ELECTRIC FIELD COMPONENTS - SUBSTRATE SIDE (0 ≤ z ≤ h₁)
# Equations 2.132, 2.133, 2.134
# =============================================================================

def Ex_substrate(y, z, h1, n_max=7):
    """Ex component in substrate (0 < z ≤ h₁) - Equation 2.132
       
    Parameters:
    - y, z: spatial coordinates
    - h1: substrate thickness
    - n_max: maximum odd mode number
    
    Returns:
    - complex value of Ex
    """
    Ex = 0.0 + 0.0j
    prefactor = 1j * 2 * V0 / lambda_g
    
    for n in range(1, n_max, 6):
        sin_n_pi_delta_2 = np.sin(n * np.pi * delta / 2)
        bracket_slot_term = (sin_n_pi_delta_2 / (n * np.pi * delta / 2)) * sin_n_pi_delta_2
        
        # Spatial cos term
        spatial_term = np.cos(n * np.pi * y / b)  # corrected here
        
        # Normalization denominator term
        denom = n * (1 + (2 * b / (n * lambda_g))**2)
        if abs(denom) < 1e-12:
            continue
        
           # Modal terms from boundary conditions
        gamma_n1 = compute_gamma_n1(n, b, lambda_g)
        rn, qn = compute_rn_qn(n, b, h1, lambda_0, lambda_g, epsilon_r1)
        
        # Hyperbolic sine and hyperbolic trig functions
        hyperbolic_term = (1 / np.tanh(qn) - np.tanh(rn)) * np.sinh(gamma_n1 * z)
        
        Ex += prefactor * (2 / denom) * bracket_slot_term * spatial_term * hyperbolic_term

    return Ex
        

def Ey_substrate(y, z, h1, n_max=7):
    """
    Ey component in substrate (0 <= z <= h1) - Equation matching Rainee Simons 2.133
    The hyperbolic part (terms involving sinh and cosh) in the CPW field formulas for the substrate region controls how the electromagnetic fields vary and decay vertically (with z) inside the substrate 
    Parameters:
    - y, z: spatial coordinates
    - h1: substrate thickness
    - n_max: maximum odd modes to sum
    
    Returns:
    - complex value of Ey
    """
    Ey = 0.0 + 0.0j
    prefactor = 2 * V0 / b

    for n in range(1, n_max, 6):  # odd n only
        # Slot term as bracketed product [sin(nπδ/2)/(nπδ/2)* sin(nπδ/2)]
        sin_n_pi_delta_2 = np.sin(n * np.pi * delta / 2)
        bracket_slot_term = (sin_n_pi_delta_2 / (n * np.pi * delta / 2)) * sin_n_pi_delta_2
        
        # Spatial sine term
        spatial_term = np.sin(n * np.pi * y / b)
        
        # Pre-factor in denominator: 1 + (2b/(nλg))^2
        denom_factor = 1 + ((2 * b )/ (lambda_g))**2
        
        # Boundary modal related functions and terms
        gamma_n1 = compute_gamma_n1(n, b, lambda_g)
        rn, qn = compute_rn_qn(n, b, h1, lambda_0, lambda_g, epsilon_r1)
        
        # Hyperbolic cosine and sine terms weighted by modal terms
        hyperbolic_term = (
            np.cosh(gamma_n1 * z)
            - ((np.tanh(rn) + ((2 * b / (n * lambda_g))**2 * (1 / np.tanh(qn)))) / denom_factor)
            * np.sinh(gamma_n1 * z)
        )
        
        # Contribution of this term in the sum
        Ey += prefactor * bracket_slot_term * spatial_term * hyperbolic_term
    
    return Ey
    

def Ez_substrate(y, z,h1, n_max=7):
    """Ez component in substrate (0 < z ≤ h₁) - Equation 2.134
    
    Parameters:
    - y, z: spatial coordinates
    - h1: substrate thickness
    - n_max: maximum odd modes to sum

    Returns:
    - complex value of Ez
    """
    Ez = 0.0 + 0.0j
    prefactor = -2 * V0 / b

    for n in range(1, n_max, 6):  # odd n only
        # Slot term as bracketed product [sin(nπδ/2)/(nπδ/2) * sin(nπδ/2)]
        sin_n_pi_delta_2 = np.sin(n * np.pi * delta / 2)
        bracket_slot_term = (sin_n_pi_delta_2 / (n * np.pi * delta / 2)) * sin_n_pi_delta_2

        # Modal coefficient Fn1
        Fn1_gamma, Fn1_sqrt = compute_Fn1(n, b, lambda_0, lambda_g, epsilon_r1)
        Fn1 = Fn1_gamma  # Use gamma component

        # Avoid division by zero
        if abs(Fn1) < 1e-12:
            continue

        # After slot term, apply 1/Fn1
        slot_with_modal = bracket_slot_term * (1 / Fn1)

        # Spatial cosine term
        spatial_term = np.cos(n * np.pi * y / b)
        
        # Boundary modal functions
        gamma_n1 = compute_gamma_n1(n, b, lambda_g)
        rn, qn = compute_rn_qn(n, b, h1, lambda_0, lambda_g, epsilon_r1)

        # Hyperbolic term: sinh(γn1 z) - tanh(rn) * cosh(γn1 z)
        hyperbolic_term = np.sinh(gamma_n1 * z) - np.tanh(rn) * np.cosh(gamma_n1 * z)

        # Sum contribution
        Ez += prefactor * slot_with_modal * spatial_term * hyperbolic_term

    return Ez


# =============================================================================
# MAGNETIC FIELD COMPONENTS - SUBSTRATE SIDE (0 ≤ z ≤ h₁)
# Equations 2.135, 2.136, 2.137
# =============================================================================

def Hx_substrate(y, z, n_max=7):
    """Hx component in substrate (0 < z ≤ h₁) - Equation 2.135
    Parameters:
    - y, z: spatial coordinates
    - h1: substrate thickness
    - n_max: maximum odd modes to sum

    Returns:
    - complex value of Hx
    """
    Hx = 0.0 + 0.0j
    # Leading prefactor before summation (matches book)
    prefactor = 1j * (2 * V0) / (eta * b) * (lambda_0 / lambda_g)**2 * (2 * b / lambda_0)
    
    for n in range(1, n_max, 6):  # odd n only
        # Modal coefficient Fn (gamma-based, matches air/definitions)
        Fn_gamma, _ = compute_Fn(n, b, lambda_0, lambda_g, epsilon_r1)
        Fn = Fn_gamma
        if abs(Fn) < 1e-12:
            continue

        # Modal coefficient Fn1
        Fn1_gamma, _ = compute_Fn1(n, b, lambda_0, lambda_g, epsilon_r1)
        Fn1 = Fn1_gamma

        # Slot term (matches book's sine product)
        sin_n_pi_delta_2 = np.sin(n * np.pi * delta / 2)
        bracket_slot_term = (sin_n_pi_delta_2 / (n * np.pi * delta / 2)) * sin_n_pi_delta_2

        # Spatial term (y)
        spatial_term = np.sin(n * np.pi * y / b)
        
        # Modal terms and boundary conditions
        gamma_n1 = compute_gamma_n1(n, b, lambda_g)
        rn, qn = compute_rn_qn(n, b, h1, lambda_0, lambda_g, epsilon_r1)

        # Hyperbolic terms
        denom = 1 + (2 * b / (n * lambda_g))**2
        lambda_ratio = lambda_g / lambda_0
        er = epsilon_r1

        hyperbolic_cosh = (
            (Fn1**2 * (1 / np.tanh(qn)) - er * lambda_ratio * np.tanh(rn)) / denom
        ) * np.cosh(gamma_n1 * z)
        hyperbolic_sinh = (1 - er * lambda_ratio**2) * np.sinh(gamma_n1 * z)

        hyperbolic_term = hyperbolic_cosh - hyperbolic_sinh

        # Modal coefficient in sum 1/(nFn)
        modal_factor = (1 / (n * Fn))
        # Sum contribution
        Hx += prefactor * modal_factor * bracket_slot_term * spatial_term * hyperbolic_term

    return Hx

def Hy_substrate(y, z, h1, n_max=7):
    """Hy component in substrate (0 < z ≤ h₁) - Equation 2.136
    
    Parameters:
    - y, z: spatial coordinates
    - h1: substrate thickness
    - n_max: maximum odd modes to sum

    Returns:
    - complex value of Hy
    """
    Hy = 0.0 + 0.0j
    prefactor = -2 * V0 / (eta * b) * (lambda_0 / lambda_g)

    for n in range(1, n_max, 6):  # odd n only
        # Modal coefficient Fn1
        Fn1_gamma, _ = compute_Fn1(n, b, lambda_0, lambda_g, epsilon_r1)
        Fn1 = Fn1_gamma
        if abs(Fn1) < 1e-12:
            continue

        # Slot term [sin(nπδ/2)/(nπδ/2) * sin(nπδ/2)]
        sin_n_pi_delta_2 = np.sin(n * np.pi * delta / 2)
        bracket_slot_term = (sin_n_pi_delta_2 / (n * np.pi * delta / 2)) * sin_n_pi_delta_2

        # Spatial cosine term
        spatial_term = np.cos(n * np.pi * y / b)

        # Modal/boundary parameters
        gamma_n1 = compute_gamma_n1(n, b, lambda_g)
        rn, qn = compute_rn_qn(n, b, h1, lambda_0, lambda_g, epsilon_r1)
        er = epsilon_r1
        num_factor = (2 * b / (n * lambda_0)) ** 2  # Note: lambda_0 (not lambda_g)
        denom = 1 + (2 * b / (n * lambda_g)) ** 2

        # Hyperbolic part as per book
        hyperbolic_cosh = (
            (Fn1 ** 2 * (1 / np.tanh(qn)) + er * num_factor * np.tanh(rn)) / denom
        ) * np.cosh(gamma_n1 * z)
        hyperbolic_sinh = np.sinh(gamma_n1 * z)
        hyperbolic_term = hyperbolic_cosh - hyperbolic_sinh

        # Summation (multiply by 1/Fn1 after slot factor as in book)
        Hy += prefactor * (1 / Fn1) * bracket_slot_term * spatial_term * hyperbolic_term

    return Hy


def Hz_substrate(y, z, h1, n_max=7):
    """Hz component in substrate (0 < z ≤ h₁) - Equation 2.137
     Hz component in substrate (0 <= z <= h1) - Equation 2.137

    Parameters:
    - y, z: spatial coordinates
    - h1: substrate thickness
    - n_max: maximum odd modes to sum

    Returns:
    - real value of Hz
    """
    Hz = 0.0
    prefactor = (2 * V0) / (eta * b) * (lambda_0 / lambda_g)

    for n in range(1, n_max, 6):  # odd n only
        # Slot term [sin(nπδ/2)/(nπδ/2) * sin(nπδ/2)]
        sin_n_pi_delta_2 = np.sin(n * np.pi * delta / 2)
        bracket_slot_term = (sin_n_pi_delta_2 / (n * np.pi * delta / 2)) * sin_n_pi_delta_2

        # Spatial sine term (y-dependence)
        spatial_term = np.sin(n * np.pi * y / b)

        # Modal/boundary parameters
        gamma_n1 = compute_gamma_n1(n, b, lambda_g)
        rn, qn = compute_rn_qn(n, b, h1, lambda_0, lambda_g, epsilon_r1)

        # Hyperbolic term: cosh(gamma_n1 z) - coth(qn) * sinh(gamma_n1 z)
        hyperbolic_term = np.cosh(gamma_n1 * z) - (1 / np.tanh(qn)) * np.sinh(gamma_n1 * z)

        # Sum contribution (note: no 1/Fn1 factor for Hz)
        Hz += prefactor * bracket_slot_term * spatial_term * hyperbolic_term

    return np.real(Hz)

# =============================================================================
# COMPUTATION GRID
# =============================================================================
n_max = 7  # or 5 or 7 -- always an integer, not a float

n_ys = 100           # Number of points along y (horizontal)
n_zs = 120           # Number of points along z (vertical)
z_air = -0.2 * h1    # Extent into air above slot (upwards of slot: z < 0)
z_sub = 1.1 * h1     # Slightly below bottom of substrate

y_arr = np.linspace(0, b, n_ys)      # y grid from left to right
z_arr = np.linspace(z_air, z_sub, n_zs)  # z grid from air (z < 0), through slot, to bottom of substrate

Y, Z = np.meshgrid(y_arr, z_arr) #Creates 2D arrays for every combination of y and z coordinate point on your calculation grid.Y[i, j], Z[i, j] refer to the y and z position of each point in your evaluation area
E_y = np.zeros_like(Y, dtype=float)#Initializes a 2D array (same shape as the grid) to store the y-component of the electric field at every grid poin
E_z = np.zeros_like(Z, dtype=float)#Initializes a 2D array for the z-component of the electric field at every grid point
H_y = np.zeros_like(Y, dtype=float) #Initializes an array for the y-component of the magnetic field.
H_z = np.zeros_like(Z, dtype=float) #Initializes an array for the z-component of the magnetic field.

# ================= Grid loop for field computation ===========================
#This nested loop iterates over every point (i,j) in your 2D spatial grid, where i indexes vertical positions (along z) and j indexes horizontal positions (along y).
#For each spatial point, it classifies the point as being:In the air region (z≤0), In the substrate region (0<z≤h1 ),Or outside the substrate (z>h1),
#and then calculates all the field components (electric and magnetic) as per the corresponding formulas for that region and stores these values in pre-allocated arrays


for i in range(n_zs): #loops vertically through the z-direction grid points.
    for j in range(n_ys): # loops horizontally through the y-direction grid points.
        y = y_arr[j]#fetch the actual spatial coordinates at each grid point.
        z = z_arr[i]#fetch the actual spatial coordinates at each grid point.
        if z <= 0:  # Air region
            E_y[i, j] = Ey_air(y, z, n_max)#to compute electric field components at (y,z).
            E_z[i, j] = Ez_air(y, z, n_max)#to compute electric field components at (y,z).
            H_y[i, j] = np.real(Hy_air(y, z, n_max))#to compute magnetic field components and store their real parts (since fields may be complex).
            H_z[i, j] = np.real(Hz_air(y, z, n_max))#to compute magnetic field components and store their real parts (since fields may be complex).
        elif 0 < z <= h1:  # Substrate region
            E_y[i, j] = np.real(Ey_substrate(y, z, h1, n_max))
            E_z[i, j] = np.real(Ez_substrate(y, z, h1, n_max))
            H_y[i, j] = np.real(Hy_substrate(y, z, h1, n_max))#
            H_z[i, j] = np.real(Hz_substrate(y, z, h1, n_max))#
        else:  # Outside substrate (could set to 0 or low value) #Set fields to zero since typically this region is outside the physical CPW and considered negligible
            E_y[i, j] = 0.0
            E_z[i, j] = 0.0
            H_y[i, j] = 0.0
            H_z[i, j] = 0.0

# ======================= Plotting electric field streamlines =============================
plt.figure(figsize=(8, 6))
# Stream plot for E-field lines
plt.streamplot(y_arr*1e3, z_arr*1e3, E_y, E_z, color=np.sqrt(E_y**2 + E_z**2), 
               linewidth=1, cmap='plasma', density=2, arrowstyle='->')

#y_arr*1e3, z_arr*1e3:X (y) and Y (z) coordinates converted from meters to millimeters for axis labeling in mm.
# color=np.sqrt(E_y**2 + E_z**2): Color code the streamlines according to the magnitude of the electric field ∣E∣, using brightness variations.
#linewidth=1: Set thickness of the streamlines
#cmap='plasma': Use the 'plasma' color map for the lines colored by magnitude.
#density=2: Controls the density of streamlines (higher is denser).
#arrowstyle='->': Arrows indicate direction of the electric field vector
#E_y, E_z,: The two vector components of the electric field at each grid point.


# Draw slot and substrate. # Draws black and grey horizontal lines representing the substrate surfaces:At z=0 (interface between air and substrate) and At the bottom of the substrate, z=h1

plt.axhline(0, color='k', lw=2)  # Top air-substrate interface
plt.axhline(h1*1e3, color='grey', lw=2, linestyle="--")  # Bottom of substrate

# Slot edges at y = (b-S/2-W) and y = (b+S/2+W) 
#Draws vertical black lines marking the boundaries of the slot in the y direction.
slot_left = b/2 - (S/2 + W)
slot_right = b/2 + (S/2 + W)
plt.axvline(slot_left*1e3, color='k', lw=3, linestyle='-', alpha=0.7)
plt.axvline(slot_right*1e3, color='k', lw=3, linestyle='-', alpha=0.7)

plt.title("Coplanar Waveguide Cross-Section Electric Field Lines (y-z plane)")
plt.xlabel("y [mm]")
plt.ylabel("z [mm] (0= interface, upwards=air, down=substrate)")
plt.xlim([0, b*1e3]) #Sets x-axis and y-axis limits for the plot to focus on the relevant region.
plt.ylim([z_air*1e3, (h1+0.05*h1)*1e3])
plt.colorbar(label="|E| (a.u.)") #Shows a side colorbar indicating the electric field magnitude scale in arbitrary units.
plt.tight_layout()
plt.show()

"""Output plot shows Electric Field Lines
 Single-Sided Slot: The field lines are calculated and displayed only from one slot, covering 
 the region between the ground and the signal conductor, rather than both sides of the typical CPW structure.

Field Pattern: The electric field lines in your plot spread outward from the slot into both the air region above (z < 0) and
the substrate (z > 0).
     
Boundary Geometry: The substrate boundary (dashed line at the top) and slot region are indicated

Direction and Line Shape: The field lines curve and return toward the edge of the slot and the metal.
Magnitude: The color intensity in your plot reflects the electric field strength
    """
    
# --- Plotting magnetic field streamlines ---
plt.figure(figsize=(8, 6))#Creates a figure window 8 inches wide by 6 inches tall for the plot.
plt.streamplot(y_arr*1e3, z_arr*1e3, H_y, H_z, color=np.sqrt(H_y**2 + H_z**2), 
               linewidth=1, cmap='cool', density=2, arrowstyle='->')




plt.axhline(0, color='k', lw=2)  # Air-Substrate interface
plt.axhline(h1*1e3, color='grey', lw=2, linestyle="--")  # Bottom of substrate

# Draw Slot edges (same approach as for E-field)
slot_left = b/2 - (S/2 + W)
slot_right = b/2 + (S/2 + W)
plt.axvline(slot_left*1e3, color='k', lw=3, linestyle='-', alpha=0.7)
plt.axvline(slot_right*1e3, color='k', lw=3, linestyle='-', alpha=0.7)

plt.title("Coplanar Waveguide Cross-Section Magnetic Field Lines (y-z plane)")
plt.xlabel("y [mm]")
plt.ylabel("z [mm] (0=interface, upwards=air, down=substrate)")
plt.xlim([0, b*1e3])
plt.ylim([z_air*1e3, (h1+0.05*h1)*1e3])
plt.colorbar(label="|H| (a.u.)")
plt.tight_layout()
plt.show()


"""Output plot shows Magnetic Field Lines
y (horizontal, in mm): This represents the physical width across the waveguide, including ground, slots, and signal regions.
z<0 (Negative z): This region is air—the space above the substrate, above the CPW structure.
0<z≤h1(Positive z): This region is the substrate—the dielectric layer beneath the CPW metal.
The solid black line at z=0 represents the air-substrate interface (top of the substrate, where the metallization sits).
The dashed gray line at z=h1(top of the plot) represents the bottom of the substrate.


The field lines arch and loop above and below the CPW surface, demonstrating how magnetic field circulates around the conductors and within the substrate.
This pattern is characteristic of transverse magnetic fields in a coplanar waveguide structure
    """