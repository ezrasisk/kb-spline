import numpy as np
from scipy.interpolate import BSpline, splrep, splev
import matplotlib.pyplot as plt

# Prototype implementation for DAG-HJB enhancement with B-spline moments.
# Run this script to compute and plot the approximated value function.

# Parameters
gamma = 1.0  # Risk aversion
sigma = 0.1  # Volatility
H = 1.0  # Total horizon
num_h_steps = 20  # Time steps
dh = H / num_h_steps
m = 5  # Global classes
k = 3  # B-spline order
fixed_delta = 0.05  # Fixed delta for ask/bid (for simplicity; replace with optimize if desired)

# Knots for S in [0,1]
t_S = np.r_[np.zeros(k+1), np.linspace(0.2, 0.8, 3), np.ones(k+1)]
num_basis_S = len(t_S) - k - 1

# Discrete q values
q_vals = np.array([-2, -1, 0, 1, 2])
num_q = len(q_vals)

# B-spline basis function (handles scalar or array input)
def b_spline_basis_S(j, S):
    spl = BSpline(t_S, np.eye(num_basis_S)[j], k)
    return spl(S)

# Compute univariate moments using numerical integration (trapz on fine grid)
S_fine = np.linspace(0, 1, 1000)
moments_S = {}
for j in range(num_basis_S):
    B_fine = b_spline_basis_S(j, S_fine)
    moments_S[j] = {
        0: np.trapz(B_fine, S_fine),
        1: np.trapz(S_fine * B_fine, S_fine),
        2: np.trapz(S_fine**2 * B_fine, S_fine)
    }

# Value function coefficients: c[num_q, num_basis_S]
c = np.zeros((num_q, num_basis_S))

# Grid for projection
S_grid = np.linspace(0, 1, 50)
basis_vals_S = np.zeros((len(S_grid), num_basis_S))
for j in range(num_basis_S):
    basis_vals_S[:, j] = b_spline_basis_S(j, S_grid)

# Initialize with terminal condition (X fixed at 0 for toy)
for iq, q in enumerate(q_vals):
    terminal = -np.exp(-gamma * q * S_grid)
    c[iq, :] = np.linalg.lstsq(basis_vals_S, terminal, rcond=None)[0]

# Evaluate V at shifted S (clamped to [0,1])
def V_at_shift(c_iq, S_shift):
    S_shift_clamped = np.clip(S_shift, 0, 1)
    basis_shift = np.array([b_spline_basis_S(j, S_shift_clamped) for j in range(num_basis_S)])
    return np.dot(basis_shift, c_iq)

# Path-specific Hamiltonian for one class
def H_i(delta_a, delta_b, V_curr, iq, S):
    lambda_a = np.exp(-delta_a)
    lambda_b = np.exp(-delta_b)
    V_a = V_at_shift(c[iq], S + delta_a)
    V_b = V_at_shift(c[iq], S - delta_b)
    return lambda_a * (V_a - V_curr) + lambda_b * (V_b - V_curr)

# Summed Hamiltonian for active subset (n classes, identical for toy)
def H_sum(S, iq, n, V_curr):
    return n * H_i(fixed_delta, fixed_delta, V_curr, iq, S)

# Backward time-stepping loop
np.random.seed(42)  # For reproducibility
for step in range(num_h_steps):
    c_new = np.copy(c)
    for iq in range(num_q):
        V_grid = basis_vals_S @ c[iq, :]

        # Diffusion term
        curr_h = H - step * dh
        spl = splrep(S_grid, V_grid, s=0.01)  # Slight smoothing
        d2V_dS2 = splev(S_grid, spl, der=2)
        diffusion = 0.5 * sigma**2 * curr_h * d2V_dS2

        # Hamiltonian term with random active subset size n
        n = np.random.randint(1, m + 1)
        H_grid = np.array([H_sum(s, iq, n, v) for s, v in zip(S_grid, V_grid)])

        # Update rule (forward Euler in reverse time)
        rhs = V_grid - dh * (diffusion + H_grid)
        c_new[iq, :] = np.linalg.lstsq(basis_vals_S, rhs, rcond=None)[0]

    # Apply moment-based correction for stability
    curv_adjust = dh * np.array([moments_S[j][2] for j in range(num_basis_S)])
    for iq in range(num_q):
        c_new[iq, :] += curv_adjust * 0.01

    c = c_new

# Example output
iq_0 = np.where(q_vals == 0)[0][0]
V_approx_mid = np.dot([b_spline_basis_S(j, 0.5) for j in range(num_basis_S)], c[iq_0, :])
print(f"Approximated V at h=0, S=0.5, q=0: {V_approx_mid}")

print("Sample moments for basis 0:", moments_S[0])

# Plot V(S) for q=0 at h=0
V_plot = basis_vals_S @ c[iq_0, :]
plt.plot(S_grid, V_plot)
plt.title("Approximated Value Function V(S) for q=0 at h=0")
plt.xlabel("S")
plt.ylabel("V")
plt.grid(True)
plt.show()
