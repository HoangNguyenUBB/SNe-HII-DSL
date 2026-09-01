# Fit wCDM model to Merged data
# (original or age corrected per Son et al)

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
import warnings
import time
import sys
warnings.filterwarnings("ignore", category=UserWarning)

# User-editable inputs / filenames
DATASETS_IN_USE = "All"  # Must be "All" for this scripts

AGE_CORRECTION = 1
USE_FULL_COVAR = 1

Z_BEND = 0.

# --------------------------

print("wCDM fit | Use", DATASETS_IN_USE, "data |", ("Age-corrected" if AGE_CORRECTION else "Raw"), "|", ("Full covar" if USE_FULL_COVAR else "Only diagonal covar"),
      # "| Fid H0 =", H0)
)
      
def age_correction(mu, z, AGE_CORRECTION):
    mu_corrected = mu.copy()
    if AGE_CORRECTION:
        correction = 0.183 * (1.0 - np.exp(-2.2*z))
        correction *= np.minimum(1.0, z / Z_BEND)
        # correction = 0.19 * (1.0 - np.exp(-z/0.5))
        mu_corrected -= correction
    return mu_corrected

c_light = 299792.458

# --------------------------
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

# dLPan = 10.0**((muPan - 25.0)/5.0)
# yPan = dLPan * H0_PAN / c_light / (1.0 + zHEL_Pan)

# yerrPan = yPan * np.log(10.0) / 5.0 * sigmuPan

# zero off-diagonals in place
if not USE_FULL_COVAR:
    covPan = np.diag(np.diag(covPan))         # create matrix with only diagonal entries

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

# dLDES = 10.0**((muDES - 25.0)/5.0)
# yDES = dLDES * H0_DES / c_light / (1.0 + zHEL_DES)

# yerrDES = yDES * np.log(10.0) / 5.0 * sigmuDES

# zero off-diagonals in place
if not USE_FULL_COVAR:
    covDES = np.diag(np.diag(covDES))         # create matrix with only diagonal entries


# ============================================================
# Restrict both datasets to z <= Z_MAX
# ============================================================

Z_MIN, Z_MAX = 0., 2.5

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
print(f"Pantheon+ restricted to z <= {Z_MAX}:")
print(f"  N = {len(zPan)}")
print(f"  z range = [{zPan.min():.6f}, {zPan.max():.6f}]")


# ------------------------------------------------------------
# DES
# ------------------------------------------------------------

maskDES = zDES <= Z_MAX
idxDES = np.where(maskDES)[0]

zDES = zDES[idxDES]
zHEL_DES = zHEL_DES[idxDES]
muDES = muDES[idxDES]

covDES = covDES[np.ix_(idxDES, idxDES)]

assert covDES.shape == (len(zDES), len(zDES))

print()
print(f"DES restricted to z <= {Z_MAX}:")
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

if DATASETS_IN_USE == "All":
    from scipy.linalg import block_diag
    
    # Hoang hack to run only "Pan"
    # zDES = zPan
    # zHEL_DES = zHEL_Pan
    # muDES = muPan
    # covDES = covPan
    
    # Hoang hack to run only "DES"
    # zPan = zDES
    # zHEL_Pan = zHEL_DES
    # muPan = muDES
    # covPan = covDES
    
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




Z_INT_MAX = 2.3
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

def mu_wcdm(params):
    """
    Parameters
    ----------
    params[0] : H0Pan
    params[1] : H0DES
    params[2] : OmegaDE
    params[3] : w
    """

    H0Pan, H0DES, OmegaDE, w = params

    if (
        not np.isfinite(H0Pan)
        or not np.isfinite(H0DES)
        or not np.isfinite(OmegaDE)
        or not np.isfinite(w)
        or H0Pan <= 0.0
        or H0DES <= 0.0
    ):
        return np.full(n_data, 1.0e30)

    # Assign a separate fitted H0 to each dataset
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


def whitened_resid(params):
    """
    Whitened residual:

        r_white = L^{-1}(mu_obs - mu_model)

    Therefore:

        chi2 = r_white.T @ r_white
    """

    residual = mu_data - mu_wcdm(params)

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
    72.0,    # H0Pan
    69.0,    # H0DES
    0.70,    # OmegaDE
    -1.0     # w
])

lower_bounds = np.array([
    30.0,    # H0Pan
    30.0,    # H0DES
    0.0,     # OmegaDE
    -3.0     # w
])

upper_bounds = np.array([
    120.0,   # H0Pan
    120.0,   # H0DES
    2.0,     # OmegaDE
    0.5      # w
])

res = least_squares(
    whitened_resid,
    p0,
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

H0Pan_best = float(res.x[0])
H0DES_best = float(res.x[1])
OmegaDE_best = float(res.x[2])
w_best = float(res.x[3])

# OmegaM_best = 1.0 - OmegaDE_best

chi2 = float(np.dot(res.fun, res.fun))

n_par = 4
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

sigH0Pan = float(parameter_errors[0])
sigH0DES = float(parameter_errors[1])
sigOmegaDE = float(parameter_errors[2])
sigw = float(parameter_errors[3])

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
    lower_bounds[2],
    rtol=0.0,
    atol=1.0e-6
)

omega_at_upper = np.isclose(
    OmegaDE_best,
    upper_bounds[2],
    rtol=0.0,
    atol=1.0e-6
)

w_at_lower = np.isclose(
    w_best,
    lower_bounds[3],
    rtol=0.0,
    atol=1.0e-6
)

w_at_upper = np.isclose(
    w_best,
    upper_bounds[3],
    rtol=0.0,
    atol=1.0e-6
)


# ------------------------------------------------------------
# 12. Report
# ------------------------------------------------------------

# print()
# print("Flat wCDM mu-space fit")
# print("----------------------")
# print("Merged dataset: separate H0Pan and H0DES")
# print()

# print(f"N Pan+   = {n_pan}")
# print(f"N DES    = {n_des}")
# print(f"N total  = {n_data}")
print()

print(
    f"H0Pan    = {H0Pan_best:.8f}"
    f" +/- {sigH0Pan:.10f}"
)

print(
    f"H0DES    = {H0DES_best:.8f}"
    f" +/- {sigH0DES:.10f}"
)

print(
    f"OmegaDE  = {OmegaDE_best:.8f}"
    f" +/- {sigOmegaDE:.8f}"
)

print(
    f"OmegaM   = {1.0 - OmegaDE_best:.8f}"
    f" +/- {sigOmegaDE:.8f}"
)

print(
    f"w        = {w_best:.8f}"
    f" +/- {sigw:.8f}"
)

# print()
# print("Parameter correlation matrix:")
# print("               H0Pan       H0DES     OmegaDE           w")

# labels = [
#     "H0Pan  ",
#     "H0DES  ",
#     "OmegaDE",
#     "w      "
# ]

# for i, label in enumerate(labels):
#     print(
#         f"{label} "
#         f"{corr_par[i,0]:12.8f} "
#         f"{corr_par[i,1]:12.8f} "
#         f"{corr_par[i,2]:12.8f} "
#         f"{corr_par[i,3]:12.8f}"
#     )

print()
print(f"chi2     = {chi2:.10f}")
# print(f"dof      = {dof}")
print(f"chi2/dof = {chi2 / dof:.6f}")
print(f"AIC      = {AIC:.6f}")
print(f"BIC      = {BIC:.6f}")

if omega_at_lower:
    print()
    print("WARNING: OmegaDE reached its lower bound.")

if omega_at_upper:
    print()
    print("WARNING: OmegaDE reached its upper bound OmegaDE = 1.")
    print(
        "The best fit may lie on the pure-w-fluid boundary; "
        "formal symmetric errors should be treated cautiously."
    )

if w_at_lower:
    print()
    print("WARNING: w reached its lower bound.")

if w_at_upper:
    print()
    print("WARNING: w reached its upper bound.")


# ------------------------------------------------------------
# 13. Export fitted values
# ------------------------------------------------------------

mu_fit = mu_wcdm([
    H0Pan_best,
    H0DES_best,
    OmegaDE_best,
    w_best
])

residue = mu_data - mu_fit

H0_fit_row = np.where(
    idx == 0,
    H0Pan_best,
    H0DES_best
)

output = np.column_stack((
    idx,
    z,
    zHEL,
    H0_fit_row,
    mu_data,
    mu_fit,
    residue
))

np.savetxt(
    "wCDM_mu_fit.txt",
    output,
    fmt=[
        "%d",       # idx
        "%.10f",    # z
        "%.10f",    # zHEL
        "%.10f",    # H0_fit
        "%.10f",    # mu_obs
        "%.10f",    # mu_fit
        "%.10f"     # residue
    ],
    header=(
        "idx z zHEL H0_fit "
        "mu_obs mu_fit residue"
    ),
    comments=""
)

print()
print("Saved wCDM_mu_fit.txt")


def chi2_by_dataset(params):
    r = whitened_resid(params)

    mask_pan = (idx == 0)
    mask_des = (idx == 1)

    chi2_pan = np.sum(r[mask_pan]**2)
    chi2_des = np.sum(r[mask_des]**2)
    chi2_tot = chi2_pan + chi2_des

    return chi2_tot, chi2_pan, chi2_des

params_best = [
    H0Pan_best,
    H0DES_best,
    OmegaDE_best,
    w_best
]

chi2_tot_w, chi2_pan_w, chi2_des_w = chi2_by_dataset(params_best)

chi2_tot_w, chi2_pan_w, chi2_des_w = chi2_by_dataset(params_best)

print(f"Pantheon+ chi2 = {chi2_pan_w:.4f}")
print(f"DES       chi2 = {chi2_des_w:.4f}")
print(f"Total     chi2 = {chi2_tot_w:.4f}")


sys.exit()