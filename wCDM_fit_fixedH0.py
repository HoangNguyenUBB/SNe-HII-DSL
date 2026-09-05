# Fit wCDM model to Merged data
# (original or age corrected per YONSEI)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# User-editable inputs / filenames
AGE_CORRECTION = True
Z_MIN, Z_MAX =  0, 2.3

if AGE_CORRECTION:
# Using Kolb (global fit post-ABC)
    H0PAN_FID_KOLB = 72.56
    H0DES_FID_KOLB = 68.79
# Using wCDM (global fit post-ABC, common w and OM)
    H0PAN_FID_WCDM = 72.78
    H0DES_FID_WCDM = 69.20
else:
# Using Kolb (global fit pre-ABC)
    H0PAN_FID_KOLB = 70.64
    H0DES_FID_KOLB = 65.66
# Using wCDM (global fit pre-ABC, common w and OM)
    H0PAN_FID_WCDM = 72.70
    H0DES_FID_WCDM = 69.13
    
# --------------------------

print("wCDM fit |", ("Post-ABC" if AGE_CORRECTION else "Pre-ABC"),
      "| H0PAN_FID_WCDM = ", H0PAN_FID_WCDM, "| H0DES_FID_WCDM = ", H0DES_FID_WCDM,
      "| H0PAN_FID_KOLB = ", H0PAN_FID_KOLB, "| H0DES_FID_KOLB = ", H0DES_FID_KOLB)
      
def age_correction(mu, z, AGE_CORRECTION):
    mu_corrected = mu.copy()
    if AGE_CORRECTION:
        correction = 0.183 * (1.0 - np.exp(-2.2*z))
        mu_corrected -= correction
    return mu_corrected

c_light = 299792.458

# Load data

# ============================================================
# Pantheon+
# ============================================================

DAT_FILE = "Pantheon+SH0ES.dat"

df = pd.read_csv(DAT_FILE, sep=r"\s+", comment="#", encoding="utf-8")

zPan = df["zHD"].values
zHEL_Pan = df["zHEL"].values
muPan = df["MU_SH0ES"].values

muPan = age_correction(muPan, zPan, AGE_CORRECTION)

COV_FILE = "Pantheon+SH0ES_STAT+SYS.npy"
covPan = np.load(COV_FILE)

assert covPan.shape[0] == len(zPan)

sigmuPan = np.sqrt(np.diag(covPan))

# ============================================================
# DES
# ============================================================

DAT_FILE = "DES-Dovekie_HD.csv"

with open(DAT_FILE, "r") as f:
    first = f.readline().strip()

cols = first.replace("VARNAMES:", "").split()

df = pd.read_csv(
    DAT_FILE,
    sep=r"\s+",
    skiprows=1,
    names=["ROWTYPE"] + cols,
    engine="python"
)

df = df[df["ROWTYPE"] == "SN:"].copy()

zDES = df["zHD"].astype(float).values
zHEL_DES = df["zHEL"].astype(float).values
muDES = df["MU"].astype(float).values

muDES = age_correction(muDES, zDES, AGE_CORRECTION)

COV_FILE = "STAT+SYS.npz"

d = np.load(COV_FILE)
N_from_file = int(d[d.files[0]][0])
upper = d[d.files[1]]

inv_cov_DES = np.zeros((N_from_file, N_from_file))
inv_cov_DES[np.triu_indices(N_from_file)] = upper

i_lower = np.tril_indices(N_from_file, -1)
inv_cov_DES[i_lower] = inv_cov_DES.T[i_lower]

covDES = np.linalg.inv(inv_cov_DES)

assert covDES.shape[0] == len(zDES)

sigmuDES = np.sqrt(np.diag(covDES))

# ------------------------------------------------------------
# Pantheon+
# ------------------------------------------------------------

maskPan = (zPan >= Z_MIN) & (zPan < Z_MAX)
idxPan = np.where(maskPan)[0]

zPan = zPan[idxPan]
zHEL_Pan = zHEL_Pan[idxPan]
muPan = muPan[idxPan]

# Slice both covariance axes using the same retained indices
covPan = covPan[np.ix_(idxPan, idxPan)]

assert covPan.shape == (len(zPan), len(zPan))

print()
print(f"Pantheon+ restricted to z between {Z_MIN} and {Z_MAX}:")
print(f"  N = {len(zPan)}")
print(f"  z range = [{zPan.min():.6f}, {zPan.max():.6f}]")

# ------------------------------------------------------------
# DES
# ------------------------------------------------------------

maskDES = (zDES >= Z_MIN) & (zDES < Z_MAX)
idxDES = np.where(maskDES)[0]

zDES = zDES[idxDES]
zHEL_DES = zHEL_DES[idxDES]
muDES = muDES[idxDES]

covDES = covDES[np.ix_(idxDES, idxDES)]

assert covDES.shape == (len(zDES), len(zDES))

print()
print(f"DES5Y restricted to z between {Z_MIN} and {Z_MAX}:")
print(f"  N = {len(zDES)}")
print(f"  z range = [{zDES.min():.6f}, {zDES.max():.6f}]")


# ============================================================
# Build dynamic design matrix
# ============================================================

def make_X(dataset_id):
    unique_ids = np.unique(dataset_id)
    X = np.zeros((len(dataset_id), len(unique_ids)))

    for k, sid in enumerate(unique_ids):
        X[dataset_id == sid, k] = 1.0

    return X, unique_ids


# ============================================================
# Combine and NO SORTING
# ============================================================

from scipy.linalg import block_diag

# Merge arrays: Pan+ first, DES second
z = np.concatenate([zPan, zDES])
zHEL = np.concatenate([zHEL_Pan, zHEL_DES])
mu_data = np.concatenate([muPan, muDES])

# Block diagonal covariance
cov = block_diag(covPan, covDES)

# Dataset label: 0 = Pan+, 1 = DES
dataset_id = np.concatenate([
    np.zeros(len(zPan), dtype=int),
    np.ones(len(zDES), dtype=int)
])

print("Merged dataset: N Pan+ =", len(zPan),", N DES =", len(zDES),", N total =", len(z))


# ============================================================
# Flat wCDM fit in mu-space: merged Pantheon+ and DES
#
# Fits:
#     H0Pan
#     H0DES
#     OmegaDE
#     w
#
# Dataset identifiers:
#     idx = 0 : Pantheon+
#     idx = 1 : DES
#
# Flat wCDM:
#     OmegaM = 1 - OmegaDE
#
#     E(z)^2 =
#         OmegaM*(1+z)^3
#         + OmegaDE*(1+z)^[3(1+w)]
#
# Luminosity distance:
#     dL_i = (c/H0_i)*(1+zHEL_i)
#            * integral_0^z_i dz'/E(z')
#
# Required inputs:
#     z, zHEL, mu_data, cov, idx, c_light
#
# Exports:
#     wCDM_mu_fit.txt
# ============================================================

import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import cholesky, solve_triangular


# ------------------------------------------------------------
# 1. Convert and validate input arrays
# ------------------------------------------------------------

idx = dataset_id
z = np.asarray(z, dtype=float)
zHEL = np.asarray(zHEL, dtype=float)
mu_data = np.asarray(mu_data, dtype=float)
cov = np.asarray(cov, dtype=float)
idx = np.asarray(idx, dtype=int)

n_data = len(mu_data)

if z.shape != (n_data,):
    raise ValueError("z must have the same length as mu_data.")

if zHEL.shape != (n_data,):
    raise ValueError("zHEL must have the same length as mu_data.")

if idx.shape != (n_data,):
    raise ValueError("idx must have the same length as mu_data.")

if cov.shape != (n_data, n_data):
    raise ValueError(
        f"cov has shape {cov.shape}; "
        f"expected ({n_data}, {n_data})."
    )

if np.any(z < 0.0):
    raise ValueError("All cosmological redshifts z must be nonnegative.")

if np.any(zHEL <= -1.0):
    raise ValueError("All zHEL values must satisfy zHEL > -1.")

if not np.all(np.isin(idx, [0, 1])):
    raise ValueError(
        "idx must contain only 0 for Pantheon+ and 1 for DES."
    )

if not np.any(idx == 0):
    raise ValueError("No Pantheon+ data found: idx contains no zero.")

if not np.any(idx == 1):
    raise ValueError("No DES data found: idx contains no one.")



Z_INT_MAX = 2.30
N_INT = 20000

z_grid = np.linspace(
    0.0,
    Z_INT_MAX,
    N_INT
)

# ------------------------------------------------------------
# 3. Cholesky factorization of full merged covariance
#
# cov = L @ L.T
# ------------------------------------------------------------

L = cholesky(
    cov,
    lower=True,
    check_finite=False
)


# ------------------------------------------------------------
# 4. Dimensionless comoving-distance integral
# ------------------------------------------------------------

def wcdm_integral(OmegaDE, w):
    """
    Return

        I(z_i) = integral_0^z_i dz' / E(z')

    for every observed redshift.

    Flat wCDM:

        OmegaM = 1 - OmegaDE

        E(z)^2 =
            OmegaM*(1+z)^3
            + OmegaDE*(1+z)^[3(1+w)]
    """

    OmegaM = 1.0 - OmegaDE

    zp1 = 1.0 + z_grid

    with np.errstate(
        over="ignore",
        invalid="ignore",
        divide="ignore"
    ):
        dark_energy_term = (
            OmegaDE
            * zp1**(3.0 * (1.0 + w))
        )

        E2 = (
            OmegaM * zp1**3
            + dark_energy_term
        )

    if (
        np.any(~np.isfinite(E2))
        or np.any(E2 <= 0.0)
    ):
        return None

    inv_E = 1.0 / np.sqrt(E2)

    dz = np.diff(z_grid)

    trapezoids = (
        0.5
        * (inv_E[:-1] + inv_E[1:])
        * dz
    )

    integral_grid = np.empty_like(z_grid)
    integral_grid[0] = 0.0
    integral_grid[1:] = np.cumsum(trapezoids)

    return np.interp(
        z,
        z_grid,
        integral_grid
    )


# ------------------------------------------------------------
# 5. wCDM distance-modulus model
# ------------------------------------------------------------

def mu_wcdm(params, H0Pan, H0DES):
    """
    Flat wCDM distance modulus with fixed H0 values.

    Parameters
    ----------
    params[0] : OmegaDE
    params[1] : w
    """

    OmegaDE, w = params

    H0_row = np.where(
        idx == 0,
        H0Pan,
        H0DES
    )

    integral = wcdm_integral(OmegaDE, w)

    if integral is None:
        return np.full(n_data, 1.0e30)

    dL = (
        (c_light / H0_row)
        * (1.0 + zHEL)
        * integral
    )

    if (
        np.any(~np.isfinite(dL))
        or np.any(dL <= 0.0)
    ):
        return np.full(n_data, 1.0e30)

    return 5.0 * np.log10(dL) + 25.0


def whitened_resid(params, H0Pan, H0DES):
    """
    Whitened residual:

        r_white = L^{-1}(mu_obs - mu_model)

    Therefore:

        chi2 = r_white.T @ r_white
    """

    residual = mu_data - mu_wcdm(params, H0Pan, H0DES)

    return solve_triangular(
        L,
        residual,
        lower=True,
        check_finite=False
    )


# ------------------------------------------------------------
# 6. Fit parameters
#
# Bounds are deliberately broad enough to include:
#
#     LCDM:
#         w = -1
#
#     coasting/Kolb-like boundary:
#         OmegaDE = 1
#         w = -1/3
#
# The upper OmegaDE bound is exactly 1 here, corresponding to
# nonnegative OmegaM in a flat model.
# ------------------------------------------------------------

p0 = np.array([
    0.70,    # OmegaDE
    -1.0     # w
])

lower_bounds = np.array([
    0.0,     # OmegaDE
    -3.0     # w
])

upper_bounds = np.array([
    1.0,     # OmegaDE
    0.5      # w
])

res = least_squares(
    whitened_resid,
    p0,
    args=(H0PAN_FID_WCDM, H0DES_FID_WCDM),
    bounds=(lower_bounds, upper_bounds),
    method="trf",
    jac="2-point",
    xtol=1.0e-12,
    ftol=1.0e-12,
    gtol=1.0e-12,
    max_nfev=20000
)

if not res.success:
    raise RuntimeError(
        "wCDM fit did not converge: " + res.message
    )


# ------------------------------------------------------------
# 7. Best-fit parameters
# ------------------------------------------------------------

OmegaDE_best = float(res.x[0])
w_best = float(res.x[1])

chi2 = float(np.dot(res.fun, res.fun))

n_par = 2
dof = n_data - n_par


# ------------------------------------------------------------
# 8. Parameter covariance and formal uncertainties
#
# No chi2/dof rescaling is applied because the supplied
# observational covariance is treated as known.
#
# Caution:
# If OmegaDE reaches its bound at 1, the symmetric Hessian
# errors are only local formal errors and should not be
# interpreted as a complete confidence interval.
# ------------------------------------------------------------

J = res.jac
JTJ = J.T @ J

cov_par = np.linalg.pinv(JTJ)

parameter_variances = np.maximum(
    np.diag(cov_par),
    0.0
)

parameter_errors = np.sqrt(parameter_variances)

sigOmegaDE = float(parameter_errors[0])
sigw = float(parameter_errors[1])
# sigOmegaM = sigOmegaDE

denom = np.outer(parameter_errors, parameter_errors)

corr_par = np.divide(
    cov_par,
    denom,
    out=np.full_like(cov_par, np.nan),
    where=(denom > 0.0)
)


# ------------------------------------------------------------
# 9. Information criteria
# ------------------------------------------------------------

AIC = chi2 + 2.0 * n_par
BIC = chi2 + n_par * np.log(n_data)


# ------------------------------------------------------------
# 10. Dataset counts
# ------------------------------------------------------------

n_pan = int(np.count_nonzero(idx == 0))
n_des = int(np.count_nonzero(idx == 1))


# ------------------------------------------------------------
# 11. Useful diagnostics
# ------------------------------------------------------------

omega_at_lower = np.isclose(
    OmegaDE_best,
    lower_bounds[0],
    rtol=0.0,
    atol=1.0e-6
)

omega_at_upper = np.isclose(
    OmegaDE_best,
    upper_bounds[0],
    rtol=0.0,
    atol=1.0e-6
)

w_at_lower = np.isclose(
    w_best,
    lower_bounds[1],
    rtol=0.0,
    atol=1.0e-6
)

w_at_upper = np.isclose(
    w_best,
    upper_bounds[1],
    rtol=0.0,
    atol=1.0e-6
)


# ------------------------------------------------------------
# 12. Report
# ------------------------------------------------------------

print()
print("Flat wCDM mu-space fit")
print("----------------------")
print("Merged dataset: separate H0Pan and H0DES")
print()

print(f"N Pan+   = {n_pan}")
print(f"N DES    = {n_des}")
print(f"N total  = {n_data}")
print()

print(
    f"H0Pan    = {H0PAN_FID_WCDM:.2f}"
)

print(
    f"H0DES    = {H0DES_FID_WCDM:.2f}"
)

print(
    f"w        = {w_best:.3f}"
    f" +/- {sigw:.3f}"
)

print(
    f"OmegaM   = {1.0 - OmegaDE_best:.3f}"
    f" +/- {sigOmegaDE:.3f}"
)

print()
print(f"chi2     = {chi2:.1f}")
print(f"AIC      = {AIC:.1f}")
print(f"BIC      = {BIC:.1f}")


# ============================================================
# Chi2 at the Kolb point
#
# Kolb:
#     OmegaM  = 0
#     OmegaDE = 1
#     w       = -1/3
#
# H0Pan and H0DES are independently fitted.
# ============================================================

OMEGADE_KOLB = 1.0
W_KOLB = -1.0 / 3.0


def mu_kolb_fixed():
    H0_row = np.where(
        idx == 0,
        H0PAN_FID_KOLB,
        H0DES_FID_KOLB
    )

    integral = wcdm_integral(
        OMEGADE_KOLB,
        W_KOLB
    )

    dL = (
        (c_light / H0_row)
        * (1.0 + zHEL)
        * integral
    )

    return 5.0 * np.log10(dL) + 25.0


residual_kolb = (
    mu_data
    - mu_kolb_fixed()
)

rwhite_kolb = solve_triangular(
    L,
    residual_kolb,
    lower=True,
    check_finite=False
)

chi2_kolb = float(
    np.dot(rwhite_kolb, rwhite_kolb)
)

n_par_kolb = 0

AIC_kolb = chi2_kolb
BIC_kolb = chi2_kolb

# ------------------------------------------------------------
# Kolb information criteria
# ------------------------------------------------------------

print()
print("Kolb point, fixed H0")
print("--------------------")
print(f"H0Pan_Kolb = {H0PAN_FID_KOLB:.2f}")
print(f"H0DES_Kolb = {H0DES_FID_KOLB:.2f}")
print(f"chi2_Kolb  = {chi2_kolb:.1f}")
print(f"AIC Kolb   = {AIC_kolb:.1f}")
print(f"BIC Kolb   = {BIC_kolb:.1f}")

print()
print(
    f"Delta chi2 (Kolb - wCDM) = "
    f"{chi2_kolb - chi2:+.1f}"
)

print(
    f"Delta AIC  (Kolb - wCDM) = "
    f"{AIC_kolb - AIC:+.1f}"
)

print(
    f"Delta BIC  (Kolb - wCDM) = "
    f"{BIC_kolb - BIC:+.1f}"
)