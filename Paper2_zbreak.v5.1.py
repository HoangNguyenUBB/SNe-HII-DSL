"""
Create three flat-wCDM contour sets in the same frame:
    1) SNe with z <  Z_BREAK_1
    2) SNe with z >= Z_BREAK_1 & z < Z_BREAK_2
    3) SNe with z >= Z_BREAK_2

Pantheon+ and DES are combined with block-diagonal covariance.
Each redshift subset profiles its own Pantheon+ and DES H0 nuisance
parameters, using only the datasets actually present in that subset.

Outputs
-------
Figure_wCDM_split_z.pdf
Figure_wCDM_split_z.png

Adapted from the user's July 2026 Figure 2 script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import block_diag, cholesky, solve_triangular
import time
from scipy.optimize import lsq_linear
from matplotlib.ticker import FuncFormatter

# ============================================================
# User settings
# ============================================================

AGE_CORRECTION_LIST = np.array([0,1])
    
Z_BREAK_LIST = np.array([2.3, 0.33]) #, 0.5, 0.7, 1.])

FIXING_H0_LIST = np.array([0,1])
USE_WCDM_AS_FIDUCIAL = 1

NUMBER_OF_CONTOURS_PLOT = 2
   
OMEGA_M_MIN, OMEGA_M_MAX = 0., 0.6
W_MIN, W_MAX = -1.5, 0.0
RESOLUTION_OM, RESOLUTION_W = 0.05, 0.05
N_OMEGA, N_W = int((OMEGA_M_MAX - OMEGA_M_MIN)/RESOLUTION_OM) + 1, int((W_MAX - W_MIN)/RESOLUTION_W) + 1

numeric_label = ["(I)", "(II)", "(III)", "(IV)", "(V)", "(VI)", "(VII)", "(VIII)"]
# numeric_label = ["(I)", "(II)", "(III)", "(IV)]
# numeric_label = ["(V)", "(VI)", "(VII)", "(VIII)"]
numeric_count = 0

for FIXING_H0 in FIXING_H0_LIST:
    for Z_BREAK in Z_BREAK_LIST:
        for AGE_CORRECTION in AGE_CORRECTION_LIST:
            print("===================================================================================")
            print("Post-ABC |" if AGE_CORRECTION else "Pre-ABC |", "Fixed H0 params" if FIXING_H0 else "Float H0 params")
                        
            PLOT_HIGH_Z_GRAPH = 1 if Z_BREAK < 2.3 else 0
                
            C_LIGHT = 299792.458
            AGE_DELTA = 0.183
            AGE_KAPPA = 2.2
            
            if AGE_CORRECTION == 0:
            # Using Kolb (global fit pre-ABC)
                H0PAN_FID_KOLB = 70.64
                H0DES_FID_KOLB = 65.66
            # Using wCDM (global fit pre-ABC, common w and OM)
                # H0PAN_FID_WCDM = 72.70
                # H0DES_FID_WCDM = 69.13
                H0PAN_FID_WCDM = 72.43
                H0DES_FID_WCDM = 69.00
            else:
            # Using Kolb (global fit post-ABC)
                H0PAN_FID_KOLB = 72.56
                H0DES_FID_KOLB = 68.79
            # Using wCDM (global fit post-ABC, common w and OM)
                # H0PAN_FID_WCDM = 72.78
                # H0DES_FID_WCDM = 69.20
                H0PAN_FID_WCDM = 72.53
                H0DES_FID_WCDM = 69.10
            
            if USE_WCDM_AS_FIDUCIAL:
                H0PAN_FID = H0PAN_FID_WCDM
                H0DES_FID = H0DES_FID_WCDM
            else:
                H0PAN_FID = H0PAN_FID_KOLB
                H0DES_FID = H0DES_FID_KOLB
            
            if FIXING_H0:
                H0_HALFWIDTH = 0.0
            else:
                H0_HALFWIDTH = 100.0
                
            H0_BOUNDS = {
                0: (max(1e-6, H0PAN_FID - H0_HALFWIDTH-1.e-6), H0PAN_FID + H0_HALFWIDTH+1.e-6),   # Pantheon+: a_Pan <= H0Pan <= b_Pan
                1: (max(1e-6, H0DES_FID - H0_HALFWIDTH-1.e-6), H0DES_FID + H0_HALFWIDTH+1.e-6),   # DES:       a_DES <= H0DES <= b_DES
            }
            
            LOW_Z_PLOT_COLOR = "limegreen"
            HIGH_Z_PLOT_COLOR  = "dodgerblue"
            
            if Z_BREAK >= 2.3:
                LOW_Z_PLOT_COLOR  = "blue" if AGE_CORRECTION else "darkred"
            
            # ============================================================
            # Utilities
            # ============================================================
            
            def age_correction(mu, z, enabled):
                mu_corrected = np.asarray(mu, dtype=float).copy()
                if enabled:
                    mu_corrected -= AGE_DELTA * (1.0 - np.exp(-AGE_KAPPA * z))
                return mu_corrected
            
            
            def load_data():
                # ---------------- Pantheon+ ----------------
                pan = pd.read_csv(
                    "Pantheon+SH0ES.dat",
                    sep=r"\s+",
                    comment="#",
                    encoding="utf-8",
                )
            
                z_pan = pan["zHD"].to_numpy(float)
                zhel_pan = pan["zHEL"].to_numpy(float)
                mu_pan = age_correction(
                    pan["MU_SH0ES"].to_numpy(float), z_pan, AGE_CORRECTION
                )
            
                cov_pan = np.load("Pantheon+SH0ES_STAT+SYS.npy")
                if cov_pan.shape != (len(z_pan), len(z_pan)):
                    raise ValueError("Pantheon+ covariance has the wrong shape.")
            
                # ---------------- DES ----------------
                with open("DES-Dovekie_HD.csv", "r", encoding="utf-8") as f:
                    first = f.readline().strip()
            
                cols = first.replace("VARNAMES:", "").split()
                des = pd.read_csv(
                    "DES-Dovekie_HD.csv",
                    sep=r"\s+",
                    skiprows=1,
                    names=["ROWTYPE"] + cols,
                    engine="python",
                )
                des = des[des["ROWTYPE"] == "SN:"].copy()
            
                z_des = des["zHD"].astype(float).to_numpy()
                zhel_des = des["zHEL"].astype(float).to_numpy()
                mu_des = age_correction(
                    des["MU"].astype(float).to_numpy(), z_des, AGE_CORRECTION
                )
            
                packed = np.load("STAT+SYS.npz")
                n_des = int(packed[packed.files[0]][0])
                upper = packed[packed.files[1]]
            
                inv_cov_des = np.zeros((n_des, n_des))
                inv_cov_des[np.triu_indices(n_des)] = upper
                lower_idx = np.tril_indices(n_des, -1)
                inv_cov_des[lower_idx] = inv_cov_des.T[lower_idx]
                cov_des = np.linalg.inv(inv_cov_des)
            
                if cov_des.shape != (len(z_des), len(z_des)):
                    raise ValueError("DES covariance has the wrong shape.")
            
                # ---------------- Merge, no sorting ----------------
                z = np.concatenate([z_pan, z_des])
                zhel = np.concatenate([zhel_pan, zhel_des])
                mu = np.concatenate([mu_pan, mu_des])
                cov = block_diag(cov_pan, cov_des)
                dataset_id = np.concatenate([
                    np.zeros(len(z_pan), dtype=int),
                    np.ones(len(z_des), dtype=int),
                ])
            
                return z, zhel, mu, cov, dataset_id
            
            
            class SubsetProfiler:
                """Profile bounded per-dataset H0 offsets for one redshift subset."""
            
                def __init__(
                    self,
                    z,
                    zhel,
                    mu,
                    cov,
                    dataset_id,
                    label,
                    h0_bounds=None,
                ):
                    self.z = np.asarray(z, dtype=float)
                    self.zhel = np.asarray(zhel, dtype=float)
                    self.mu = np.asarray(mu, dtype=float)
                    self.cov = np.asarray(cov, dtype=float)
                    self.dataset_id = np.asarray(dataset_id, dtype=int)
                    self.label = label
            
                    # Dataset 0 = Pantheon+, dataset 1 = DES.
                    if h0_bounds is None:
                        h0_bounds = {
                            0: (60.0, 80.0),  # Pantheon+
                            1: (60.0, 80.0),  # DES
                        }
            
                    self.h0_bounds = h0_bounds
            
                    n = len(self.mu)
            
                    if n == 0:
                        raise ValueError(f"The {label} subset is empty.")
            
                    if not (
                        len(self.z)
                        == len(self.zhel)
                        == len(self.mu)
                        == len(self.dataset_id)
                    ):
                        raise ValueError(
                            f"The arrays in the {label} subset have inconsistent lengths."
                        )
            
                    if self.cov.shape != (n, n):
                        raise ValueError(
                            f"The {label} covariance has the wrong shape: "
                            f"{self.cov.shape}; expected {(n, n)}."
                        )
            
                    if np.any(~np.isfinite(self.z)):
                        raise ValueError(f"The {label} subset contains non-finite z values.")
            
                    if np.any(~np.isfinite(self.zhel)):
                        raise ValueError(
                            f"The {label} subset contains non-finite heliocentric redshifts."
                        )
            
                    if np.any(~np.isfinite(self.mu)):
                        raise ValueError(
                            f"The {label} subset contains non-finite distance moduli."
                        )
            
                    if np.any(self.z <= 0.0):
                        raise ValueError(
                            f"The {label} subset contains non-positive cosmological redshifts."
                        )
            
                    self.active_ids = np.unique(self.dataset_id)
            
                    if not np.all(np.isin(self.active_ids, [0, 1])):
                        raise ValueError("dataset_id must contain only 0 and 1.")
            
                    # Cholesky whitening of the covariance matrix.
                    self.L = cholesky(
                        self.cov,
                        lower=True,
                        check_finite=False,
                    )
            
                    # One calibration column for each dataset present in this subset.
                    self.X = np.column_stack(
                        [
                            (self.dataset_id == dataset).astype(float)
                            for dataset in self.active_ids
                        ]
                    )
            
                    self.WX = solve_triangular(
                        self.L,
                        self.X,
                        lower=True,
                        check_finite=False,
                    )
            
                    # Construct beta bounds corresponding to the requested H0 bounds.
                    #
                    # beta = -5 log10(H0)
                    # H0   = 10^(-beta/5)
                    #
                    # Since this relation decreases with beta, the bound order reverses.
                    beta_lower = []
                    beta_upper = []
            
                    for dataset in self.active_ids:
                        dataset = int(dataset)
            
                        if dataset not in self.h0_bounds:
                            raise ValueError(
                                f"No H0 bounds supplied for dataset {dataset}."
                            )
            
                        h0_min, h0_max = self.h0_bounds[dataset]
            
                        if (
                            not np.isfinite(h0_min)
                            or not np.isfinite(h0_max)
                            or h0_min <= 0.0
                            or h0_max <= h0_min
                        ):
                            raise ValueError(
                                f"Invalid H0 bounds for dataset {dataset}: "
                                f"{h0_min}, {h0_max}"
                            )
            
                        beta_lower.append(-5.0 * np.log10(h0_max))
                        beta_upper.append(-5.0 * np.log10(h0_min))
            
                    self.beta_lower = np.asarray(beta_lower, dtype=float)
                    self.beta_upper = np.asarray(beta_upper, dtype=float)
            
                    # ---------------------------------------------------------------
                    # IMPORTANT:
                    # Construct the integration grid only over the redshift interval
                    # required by this subset.
                    #
                    # Do NOT force the grid to z=2.6 for a low-z subset. Doing so would
                    # reject models whose E^2 becomes negative only at redshifts that
                    # are irrelevant to the subset being fitted.
                    # ---------------------------------------------------------------
                    self.data_zmax = float(np.max(self.z))
                    self.zmax = 2.3 # 1.001 * self.data_zmax
            
                    self.n_int = 20000
            
                    self.z_int = np.linspace(
                        0.0,
                        self.zmax,
                        self.n_int,
                        dtype=float,
                    )
            
                    self.zp1_int = 1.0 + self.z_int
                    self.dz_int = np.diff(self.z_int)
            
                    print(
                        f"{self.label}: "
                        f"data zmax={self.data_zmax:.6f}, "
                        f"integration zmax={self.zmax:.6f}"
                    )
            
                def wcdm_integral_at_data(self, omega_de, w):
                    """
                    Return the dimensionless comoving-distance integral
            
                        integral_0^z dz' / E(z')
            
                    evaluated at the redshifts in this subset.
            
                    Returns None when the model is not real over the redshift interval
                    actually required by this subset.
                    """
                    omega_de = float(omega_de)
                    w = float(w)
                    omega_m = 1.0 - omega_de
            
                    if not np.isfinite(omega_de) or not np.isfinite(w):
                        return None
            
                    with np.errstate(
                        over="ignore",
                        invalid="ignore",
                        divide="ignore",
                        under="ignore",
                    ):
                        e2 = (
                            omega_m * self.zp1_int**3
                            + omega_de
                            * self.zp1_int**(3.0 * (1.0 + w))
                        )
            
                    invalid = (~np.isfinite(e2)) | (e2 <= 0.0)
            
                    if np.any(invalid):
                        # This is a genuinely invalid model over this subset's
                        # required integration interval.
                        return None
            
                    inv_e = 1.0 / np.sqrt(e2)
            
                    if np.any(~np.isfinite(inv_e)):
                        return None
            
                    integral_grid = np.empty_like(self.z_int)
                    integral_grid[0] = 0.0
            
                    integral_grid[1:] = np.cumsum(
                        0.5
                        * (inv_e[:-1] + inv_e[1:])
                        * self.dz_int
                    )
            
                    integral_at_data = np.interp(
                        self.z,
                        self.z_int,
                        integral_grid,
                    )
            
                    if np.any(~np.isfinite(integral_at_data)):
                        return None
            
                    return integral_at_data
            
                def mu_without_h0(self, omega_de, w):
                    """
                    Compute the theoretical distance modulus with the dataset-specific
                    H0 offsets omitted.
            
                    The profiled calibration parameter is subsequently
            
                        beta = -5 log10(H0).
                    """
                    integral = self.wcdm_integral_at_data(omega_de, w)
            
                    if integral is None:
                        return None
            
                    distance_without_h0 = (
                        C_LIGHT
                        * (1.0 + self.zhel)
                        * integral
                    )
            
                    if (
                        np.any(~np.isfinite(distance_without_h0))
                        or np.any(distance_without_h0 <= 0.0)
                    ):
                        return None
            
                    mu0 = (
                        5.0 * np.log10(distance_without_h0)
                        + 25.0
                    )
            
                    if np.any(~np.isfinite(mu0)):
                        return None
            
                    return mu0
            
                def profile(self, omega_de, w):
                    """
                    Profile the bounded Pantheon+ and/or DES H0 calibration offsets
                    for a fixed (Omega_DE, w).
                    """
                    mu0 = self.mu_without_h0(omega_de, w)
            
                    if mu0 is None:
                        return np.inf, {}
            
                    y = self.mu - mu0
            
                    if np.any(~np.isfinite(y)):
                        return np.inf, {}
            
                    wy = solve_triangular(
                        self.L,
                        y,
                        lower=True,
                        check_finite=False,
                    )
            
                    if np.any(~np.isfinite(wy)):
                        return np.inf, {}
            
                    result = lsq_linear(
                        self.WX,
                        wy,
                        bounds=(self.beta_lower, self.beta_upper),
                        method="trf",
                        tol=1.0e-12,
                        lsmr_tol="auto",
                        max_iter=200,
                    )
            
                    if not result.success:
                        return np.inf, {}
            
                    beta = result.x
                    residual = wy - self.WX @ beta
                    chi2 = float(residual @ residual)
            
                    if not np.isfinite(chi2):
                        return np.inf, {}
            
                    h0 = {
                        int(dataset): 10.0 ** (-beta[i] / 5.0)
                        for i, dataset in enumerate(self.active_ids)
                    }
            
                    return chi2, h0
            
                def evaluate_grid(self, omega_m_grid, w_grid):
                    """
                    Evaluate the profiled chi-square over an (Omega_M, w) grid.
                    """
                    omega_m_grid = np.asarray(omega_m_grid, dtype=float)
                    w_grid = np.asarray(w_grid, dtype=float)
            
                    chi2_grid = np.full(
                        (len(w_grid), len(omega_m_grid)),
                        np.inf,
                        dtype=float,
                    )
            
                    n_grid_points = len(omega_m_grid) * len(w_grid)
            
                    print(f"\nEvaluating {self.label}: N={len(self.z)}")
                    print(
                        f"Data range: "
                        f"{np.min(self.z):.6f} <= z <= {np.max(self.z):.6f}"
                    )
                    print(
                        f"Integration range: "
                        f"0 <= z <= {self.zmax:.6f}"
                    )
                    print(
                        f"Grid: {len(omega_m_grid)} x {len(w_grid)} "
                        f"= {n_grid_points} points"
                    )
            
                    for iw, w_value in enumerate(w_grid):
                        if iw % 10 == 0:
                            print(f"  row {iw + 1:3d} of {len(w_grid)}", end="")
            
                        for io, omega_m_value in enumerate(omega_m_grid):
                            omega_de_value = 1.0 - omega_m_value
            
                            chi2, _ = self.profile(
                                omega_de_value,
                                w_value,
                            )
            
                            chi2_grid[iw, io] = chi2
                    print()
            
                    finite = np.isfinite(chi2_grid)
            
                    if not np.any(finite):
                        raise RuntimeError(
                            f"No valid grid points for {self.label}."
                        )
            
                    n_invalid = np.size(chi2_grid) - np.count_nonzero(finite)
            
                    print(
                        f"Finite grid points: {np.count_nonzero(finite)} "
                        f"of {chi2_grid.size}"
                    )
                    print(f"Invalid grid points: {n_invalid}")
            
                    # Find the best finite grid point.
                    finite_chi2 = np.where(
                        finite,
                        chi2_grid,
                        np.nan,
                    )
            
                    flat_best = np.nanargmin(finite_chi2)
            
                    iw_best, io_best = np.unravel_index(
                        flat_best,
                        chi2_grid.shape,
                    )
            
                    grid_chi2_min = chi2_grid[iw_best, io_best]
                    chi2_min = grid_chi2_min
                        
                    omega_m_best = omega_m_grid[io_best]
                    omega_de_best = 1.0 - omega_m_best
                    w_best = w_grid[iw_best]
            
                    _, h0_best = self.profile(
                        omega_de_best,
                        w_best,
                    )
            
                    delta_chi2 = chi2_grid - chi2_min
            
                    print(f"\nBest fit: {self.label}")
                    print("-" * (10 + len(self.label)))
                    print(f"Omega_M = {omega_m_best:.8f}")
                    print(f"Omega_DE= {omega_de_best:.8f}")
                    print(f"w       = {w_best:.8f}")
            
                    if 0 in h0_best:
                        print(f"H0Pan   = {h0_best[0]:.8f}")
            
                    if 1 in h0_best:
                        print(f"H0DES   = {h0_best[1]:.8f}")
            
                    print(f"grid chi2 minimum = {grid_chi2_min:.8f}")
            
                    if chi2_min != grid_chi2_min:
                        print(f"adopted chi2 min  = {chi2_min:.8f}")
                    else:
                        print(f"chi2              = {chi2_min:.8f}")
            
                    # Coasting / logarithmic point:
                    # Omega_M = 0, Omega_DE = 1, w = -1/3.
                    chi2_coast, h0_coast = self.profile(
                        1.0,
                        -1.0 / 3.0,
                    )
            
                    if np.isfinite(chi2_coast):
                        print(
                            f"Delta chi2 at coasting = "
                            f"{chi2_coast - chi2_min:.8f}"
                        )
            
                        if 0 in h0_coast:
                            print(
                                f"H0Pan at coasting = "
                                f"{h0_coast[0]:.8f}"
                            )
            
                        if 1 in h0_coast:
                            print(
                                f"H0DES at coasting = "
                                f"{h0_coast[1]:.8f}"
                            )
                    else:
                        print("Coasting point returned a non-finite chi2.")
            
                    return {
                        "chi2": chi2_grid,
                        "delta_chi2": delta_chi2,
                        "chi2_min": chi2_min,
                        "grid_chi2_min": grid_chi2_min,
                        "omega_m_best": omega_m_best,
                        "omega_de_best": omega_de_best,
                        "w_best": w_best,
                        "h0_best": h0_best,
                        "zmax_data": self.data_zmax,
                        "zmax_integration": self.zmax,
                    }
            
            
            def make_subset(
                z,
                zhel,
                mu,
                cov,
                dataset_id,
                mask,
                label,
                h0_bounds=None,
            ):
                """Construct a SubsetProfiler from a Boolean redshift-selection mask."""
                z = np.asarray(z, dtype=float)
                zhel = np.asarray(zhel, dtype=float)
                mu = np.asarray(mu, dtype=float)
                cov = np.asarray(cov, dtype=float)
                dataset_id = np.asarray(dataset_id, dtype=int)
                mask = np.asarray(mask, dtype=bool)
            
                if mask.shape != z.shape:
                    raise ValueError(
                        f"The mask for {label} has shape {mask.shape}, "
                        f"but z has shape {z.shape}."
                    )
            
                indices = np.flatnonzero(mask)
            
                if len(indices) == 0:
                    raise ValueError(
                        f"The selection mask for {label} contains no data."
                    )
            
                return SubsetProfiler(
                    z=z[indices],
                    zhel=zhel[indices],
                    mu=mu[indices],
                    cov=cov[np.ix_(indices, indices)],
                    dataset_id=dataset_id[indices],
                    label=label,
                    h0_bounds=h0_bounds,
                )
                        
            
            # ============================================================
            # Main calculation
            # ============================================================
            
            t0 = time.perf_counter()
            
            z, zhel, mu_data, cov, dataset_id = load_data()
            
            low_mask  = z <  Z_BREAK
            high_mask = z >= Z_BREAK
            
            low_label  = rf"Low  z (z <  {Z_BREAK:g})" if Z_BREAK < 2.3 else "All z"
            high_label = rf"High z (z >= {Z_BREAK:g})"

            low_fit = make_subset(
                z,
                zhel,
                mu_data,
                cov,
                dataset_id,
                low_mask,
                low_label,
                h0_bounds=H0_BOUNDS,
            )
            
            if PLOT_HIGH_Z_GRAPH == 1:
                high_fit = make_subset(
                    z,
                    zhel,
                    mu_data,
                    cov,
                    dataset_id,
                    high_mask,
                    high_label,
                    h0_bounds=H0_BOUNDS,
                )
            
            print("\nSample split")
            print("------------")
            print(f"z in (0.0,{Z_BREAK:g}): {np.count_nonzero(low_mask)} SNe")
            print(f"z in [{Z_BREAK:g},2.3): {np.count_nonzero(high_mask)} SNe")
            print(
                f"Low-z Pan+/DES: "
                f"{np.count_nonzero(low_mask & (dataset_id == 0))}/"
                f"{np.count_nonzero(low_mask & (dataset_id == 1))}"
            )
            print(
                f"High-z Pan+/DES: "
                f"{np.count_nonzero(high_mask & (dataset_id == 0))}/"
                f"{np.count_nonzero(high_mask & (dataset_id == 1))}"
            )
            
            omega_m_grid = np.linspace(OMEGA_M_MIN, OMEGA_M_MAX, N_OMEGA)
            w_grid = np.linspace(W_MIN, W_MAX, N_W)
            omega_mesh, w_mesh = np.meshgrid(omega_m_grid, w_grid)
            
            low_result = low_fit.evaluate_grid(omega_m_grid, w_grid)
            if PLOT_HIGH_Z_GRAPH == 1:
                high_result = high_fit.evaluate_grid(omega_m_grid, w_grid)
            
            # ============================================================
            # Plot all 2 contour sets in the same frame
            # ============================================================
            
            fig, ax = plt.subplots(figsize=(5, 6))
            
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.5)
            
            ax.set_xlabel(r"$\Omega_{\,m}$", fontsize=20, color="black")
            ax.set_ylabel(r"$w$", fontsize=20, color="black", rotation=0)
            ax.yaxis.set_label_coords(-0.12, 0.48)
            ax.set_xlim(OMEGA_M_MIN, OMEGA_M_MAX)
            ax.set_ylim(W_MIN, W_MAX)
            ax.set_xticks(np.arange(OMEGA_M_MIN, OMEGA_M_MAX+0.00001, 0.2))
            # ax.set_xticks([0., 0.2, 0.4, 0.6])
            ax.set_yticks(np.arange(W_MIN, W_MAX+0.00001, 0.5))
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=13,
                colors="black",
                width=1.2,
                length=6,
            )
            
            # Low-z:
            chi2 = low_result["delta_chi2"]
            chi2_min = np.min(chi2)
            
            # Filled 1σ
            ax.contourf(
                omega_mesh, w_mesh, chi2,
                levels=[chi2_min, chi2_min+2.30],
                colors=LOW_Z_PLOT_COLOR,
                alpha=0.5 if Z_BREAK < 2.3 else 0.8,
                order=1
            )
            
            if NUMBER_OF_CONTOURS_PLOT >= 2: 
                # Outline 2-sigma
                ax.contour(
                    omega_mesh, w_mesh, chi2,
                    # levels=[chi2_min+2.30, chi2_min+6.18],
                    levels=[chi2_min+6.18],
                    colors=LOW_Z_PLOT_COLOR,
                    linewidths= 1.5,
                    alpha=0.4
                )
            if NUMBER_OF_CONTOURS_PLOT == 3:
                # Outline 3-sigma
                ax.contour(
                    omega_mesh, w_mesh, chi2,
                    # levels=[chi2_min+2.30, chi2_min+6.18, chi2_min+11.83],
                    levels=[chi2_min+11.83],
                    colors=LOW_Z_PLOT_COLOR,
                    linewidths= 0.5,
                    alpha=0.6
                )
                
            # High-z:
            if PLOT_HIGH_Z_GRAPH == 1:
                chi2 = high_result["delta_chi2"]
                chi2_min = np.min(chi2)
            
                # Filled 1σ
                ax.contourf(
                    omega_mesh, w_mesh, chi2,
                    levels=[chi2_min, chi2_min+2.30],
                    colors=HIGH_Z_PLOT_COLOR,
                    alpha=0.6,
                    order=8
                )
                
                if NUMBER_OF_CONTOURS_PLOT >= 2: 
                    # Outline 2-sigma
                    ax.contour(
                        omega_mesh, w_mesh, chi2,
                        # levels=[chi2_min+2.30, chi2_min+6.18],
                        levels=[chi2_min+6.18],
                        colors=HIGH_Z_PLOT_COLOR,
                        linewidths= 1.5,
                        alpha=0.6
                    )
                if NUMBER_OF_CONTOURS_PLOT == 3:
                    # Outline 3-sigma
                    ax.contour(
                        omega_mesh, w_mesh, chi2,
                        # levels=[chi2_min+2.30, chi2_min+6.18, chi2_min+11.83],
                        levels=[chi2_min+11.83],
                        colors=HIGH_Z_PLOT_COLOR,
                        linewidths= 0.5,
                        alpha=0.6
                    )
            
            # Coasting point: Omega_M=0, w=-1/3
            ax.scatter(
                0.0,
                -1.0 / 3.0,
                marker="*",
                s=220,
                color="black",
                zorder=8,
                clip_on=False,
                label='Logarithmic $d_L$'
            )
            
            # Flat LCDM locus
            ax.axhline(-1.0, linestyle="-.", linewidth=1.3, color="black", alpha=0.8)
            
            ax.axvline( 0.0, linestyle="-.", linewidth=1.3, color="black", alpha=0.8)
            
            ax.grid(alpha=0.22)
            
            subset_handles = []
            
            if PLOT_HIGH_Z_GRAPH == 1:
                subset_handles.extend([
                    Line2D([], [],
                        marker='s',
                        linestyle='None',
                        markerfacecolor=LOW_Z_PLOT_COLOR,
                        markeredgecolor=LOW_Z_PLOT_COLOR,
                        markersize=8,
                        alpha=0.6,
                        label=rf'Low  $z$ $(< {Z_BREAK})$'),
                    Line2D([], [],
                        marker='s',
                        linestyle='None',
                        markerfacecolor=HIGH_Z_PLOT_COLOR,
                        markeredgecolor=HIGH_Z_PLOT_COLOR,
                        markersize=8,
                        alpha=0.6,
                        label=rf'High $z$ $(\,\geqslant\ {Z_BREAK})$')
                    ])
            else:
                subset_handles.append(
                    Line2D([], [],
                        marker='s',
                        linestyle='None',
                        markerfacecolor=LOW_Z_PLOT_COLOR,
                        markeredgecolor=LOW_Z_PLOT_COLOR,
                        markersize=8,
                        alpha=0.6,
                        label="Full datasets")
                )
            
            # Common entries
            subset_handles.extend([
                Line2D(
                    [], [],
                    marker='s',
                    linestyle='None',
                    markerfacecolor="grey",
                    markeredgecolor="grey",
                    markersize=8,
                    alpha=0.5,
                    label=r'Chávez et al.'
                ),
                Line2D(
                    [0], [0],
                    marker='*',
                    color='black',
                    linestyle='None',
                    markersize=12,
                    # label='Logarithmic $d_{\,L}$'
                    label=r'Kolb ($w^*$=-1/3)'
                )
            ])
            
            confidence_handles = [
                Line2D([0], [0], color=color, lw=width, label=label)
                for color, width, label in zip(
                    ["blue", "blue"], [2., 1.], ["68%", "95%"]
                )
            ]
            
            # ============================================================
            # Overlay model points
            # ============================================================
            
            solid = np.loadtxt("Chavez-fig-5-solid.txt", skiprows=1)
            rim   = np.loadtxt("..\Chavez-fig-5-rim_simplified.txt", skiprows=1)
            
            step = 2
            ax.scatter(
                solid[::step, 0],
                solid[::step, 1],
                s=2,
                color="grey",
                marker="o",
                alpha=.25,
                edgecolors="none",
                zorder=0,
            )
            
            ax.plot(
                rim[:, 0],
                rim[:, 1],
                color="grey",
                linewidth=2,
                alpha=.25,
                zorder=0,
            )
            
            ax.text(
                0.04, 0.03,
                numeric_label[numeric_count],
                transform=ax.transAxes,
                fontsize=16,
                ha="left",
                va="bottom",
                color="saddlebrown",
                fontweight="bold"
            )
            
            formatter = FuncFormatter(lambda x, pos: f'{int(x)}' if np.isclose(x, round(x)) else f'{x:.1f}')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
                 
            prefix  = "Post-ABC" if AGE_CORRECTION else "Pre-ABC"
            postfix = "Fixed $\mathbf{H_0}$" if FIXING_H0 else "Float $\mathbf{H_0}$"

            leg = ax.legend(
                title=rf"{prefix} / {postfix}:",
                # title=rf"{prefix} SNe data:",
                title_fontsize=10,
                handles=subset_handles,
                framealpha=1.,
                edgecolor="none",
                loc="upper right",
                fontsize=9,
            )
            
            leg.get_title().set_fontweight("bold")
            
            plt.tight_layout()
            
            output_pdf = "Figure_wCDM.Panel_" + numeric_label[numeric_count] + ".pdf"
            output_png = "Figure_wCDM.Panel_" + numeric_label[numeric_count] + ".png"
            plt.savefig(output_pdf, bbox_inches="tight")
            plt.savefig(output_png, dpi=300, bbox_inches="tight")
            plt.show()
            
            print("\nSaved", output_pdf)
            print("Saved", output_png)
            elapsed = round(time.perf_counter() - t0)
            if elapsed < 60:
                print(f"Elapsed: {elapsed:.1f} sec.")
            else:
                print(f"Elapsed: {elapsed/60:.1f} min.")
            numeric_count += 1
                
# np.savez(
#     "wCDM_chi2_grid_PreABC.npz",
#     w_grid=w_grid,
#     Om_grid=omega_m_grid,
#     chi2_grid=low_result["chi2"]
# )

# print("Saved wCDM_chi2_grid.npz")