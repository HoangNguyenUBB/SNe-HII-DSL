# Dolgov-Barrow model
# Profiling over H0Pan and H0DES
# Merged dataset (Pan+ and DES)

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
# from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve
# from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from math import sqrt, exp
import matplotlib.pyplot as plt
import warnings
import time
import sys
warnings.filterwarnings("ignore", category=UserWarning)

# User-editable inputs / filenames
DATASETS_IN_USE = "All"   # Pan, DES or All

AGE_CORRECTION = 1
USE_FULL_COVAR = 1

w_grid    = np.linspace(-1., 0, 11)
zeta_grid = np.linspace(-1, 1, 21)

# --------------------------

print("Dolgov-Barrow model | Use", DATASETS_IN_USE, "data |", ("Age-corrected" if AGE_CORRECTION else "Raw"))

def age_correction(mu, z, AGE_CORRECTION):
    mu_corrected = mu.copy()
    if AGE_CORRECTION:
        correction = 0.183 * (1.0 - np.exp(-2.2*z))
        mu_corrected -= correction
    return mu_corrected

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

if DATASETS_IN_USE == "Pan":
    z = zPan
    zHEL = zHEL_Pan
    mu_data = muPan
    cov = covPan
    dataset_id = np.zeros(len(zPan), dtype=int)
    
if DATASETS_IN_USE == "DES":
    z = zDES
    zHEL = zHEL_DES
    mu_data = muDES
    cov = covDES
    dataset_id = np.ones(len(zDES), dtype=int)



t0 = time.perf_counter()

# ============================================================
# Required inputs already defined:
# z, zHEL, mu_data, cov, dataset_id
#
# dataset_id:
#   0 = Pan+
#   1 = DES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

c_light = 299792.458

def make_X(dataset_id):
    unique_ids = np.unique(dataset_id)
    X = np.zeros((len(dataset_id), len(unique_ids)))
    for k, sid in enumerate(unique_ids):
        X[dataset_id == sid, k] = 1.0
    return X, unique_ids


cho = cho_factor(cov, lower=True, check_finite=False)

X, unique_ids = make_X(dataset_id)
CinvX = cho_solve(cho, X, check_finite=False)
A = X.T @ CinvX
Ainv = np.linalg.inv(A)


def DB_integral(z, zeta, mu_power):
    """
    c ~ a^(-zeta)
    a ~ t^mu
    1+z = a^(-1-zeta)

    b = 1/a
    bmax = (1+z)^(1/(1+zeta))

    I = integral_1^bmax b^(zeta - 1/mu) db
    """

    z = np.asarray(z, dtype=float)

    if mu_power <= 0:
        return None

    if 1.0 + zeta <= 0:
        return None

    bmax = (1.0 + z)**(1.0 / (1.0 + zeta))

    p = zeta - 1.0 / mu_power

    if abs(p + 1.0) < 1e-12:
        I = np.log(bmax)
    else:
        I = (bmax**(p + 1.0) - 1.0) / (p + 1.0)

    if np.any(I <= 0):
        return None

    return I


def mu_shape_DB_noH0(z, zHEL, zeta, mu_power):
    I = DB_integral(z, zeta, mu_power)

    if I is None:
        return None

    dL_noH0 = c_light * (1.0 + zHEL) * I

    if np.any(dL_noH0 <= 0):
        return None

    return 5.0 * np.log10(dL_noH0) + 25.0


def profiled_chi2_DB(zeta, mu_power):
    mu_shape = mu_shape_DB_noH0(z, zHEL, zeta, mu_power)

    if mu_shape is None:
        return np.inf, np.full(X.shape[1], np.nan)

    y = mu_data - mu_shape

    Cinvy = cho_solve(cho, y, check_finite=False)

    b = X.T @ Cinvy
    gamma = y @ Cinvy

    x_best = Ainv @ b
    chi2 = gamma - b @ x_best

    return chi2, x_best




# ============================================================
# Scan DB in the (w, zeta) plane
# ============================================================

# First index = zeta
# Second index = w
chi2_grid = np.full((len(zeta_grid), len(w_grid)), np.nan)

n_x = X.shape[1]
x_grid = np.full(
    (len(zeta_grid), len(w_grid), n_x),
    np.nan
)

for i, zeta_val in enumerate(zeta_grid):

    for j, w_val in enumerate(w_grid):

        # Dolgov single-fluid relation
        mu_val = 2.0 / (3.0 * (1.0 + w_val))

        chi2, xbest = profiled_chi2_DB(
            zeta_val,
            mu_val
        )

        chi2_grid[i, j] = chi2
        x_grid[i, j, :] = xbest

    if i % 10 == 0:
        print(f"zeta row {i}/{len(zeta_grid)}", end="\r")


# ============================================================
# Best fit
# ============================================================

chi2_min = np.nanmin(chi2_grid)
Delta_chi2 = chi2_grid - chi2_min

idx_best = np.unravel_index(
    np.nanargmin(chi2_grid),
    chi2_grid.shape
)

i_zeta, i_w = idx_best

best_zeta = zeta_grid[i_zeta]
best_w    = w_grid[i_w]
best_mu   = 2.0 / (3.0 * (1.0 + best_w))

best_x = x_grid[i_zeta, i_w, :]

print("\n\nBest fit:")
print(f"w     = {best_w:.6f}")
print(f"zeta  = {best_zeta:.6f}")
print(f"mu    = {best_mu:.6f}")
print(f"eta   = {(1.0 + best_zeta)*best_mu:.6f}")
print(f"chi2  = {chi2_min:.6f}")

for k, sid in enumerate(unique_ids):
    H0 = 10.0**(-best_x[k] / 5.0)

    name = (
        "Pan+" if sid == 0
        else "DES" if sid == 1
        else f"Dataset {sid}"
    )

    print(f"{name}: H0 = {H0:.6f}")


# ============================================================
# Plot DB likelihood in (w, zeta)
# ============================================================

fig, ax = plt.subplots(figsize=(3.8, 4.5))

# # Filled likelihood
# ax.contourf(
#     w_grid,
#     zeta_grid,
#     Delta_chi2,
#     levels=60
# )

# # 1, 2 sigma for two parameters
# cs = ax.contour(
#     w_grid,
#     zeta_grid,
#     Delta_chi2,
#     levels=[2.30, 6.18],
#     linewidths=1.5
# )

# shade inside 1 sigma
ax.contourf(
    w_grid,
    zeta_grid,
    Delta_chi2,
    levels=[0.0, 2.30],
    alpha=0.4 if AGE_CORRECTION == 0 else 0.8,
    colors="red" if AGE_CORRECTION == 0 else "orange"
)

# draw 2-sigma boundary
ax.contour(
    w_grid,
    zeta_grid,
    Delta_chi2,
    levels=[2.30, 6.18],
    linewidths=[1., 1.],
    colors=["red"] if AGE_CORRECTION == 0 else ["orange"]
)

# ============================================================
# SIG line
#
# eta = (1+zeta) mu = 1
#
# with
#
# mu = 2/[3(1+w)]
#
# gives
#
# zeta = (1 + 3w)/2
# ============================================================

w_line = np.linspace(
    w_grid.min(),
    w_grid.max(),
    100
)

zeta_line = 0.5 * (1.0 + 3.0*w_line)

mask = (
    (zeta_line >= zeta_grid.min())
    & (zeta_line <= zeta_grid.max())
)

ax.plot(
    w_line[mask],
    zeta_line[mask],
    "--",
    lw=2.,
    color="black",
    label=r"SIG: $\zeta=$(1+3w)/2"
)


# ============================================================
# Constant-c Dolgov line
# ============================================================

ax.axhline(
    0.0,
    ls=":",
    lw=1.5,
    # label=r"Dolgov: $\zeta=0$"
)


# ============================================================
# Special points
# ============================================================

# Kolb:
# w = -1/3, mu = 1, zeta = 0
ax.scatter(
    [-1.0/3.0],
    [0.0],
    marker="*",
    s=200,
    color="blue",
    zorder=10,
    label=r"Kolb: $w^*=\!-\!$1/3"
)

# # SIG / EdS:
# # w = 0, mu = 2/3, zeta = 1/2
# ax.scatter(
#     [0.0],
#     [0.5],
#     marker="o",
#     s=50,
#     zorder=10,
#     label="SIG-EdS"
# )

# # Numerical best fit
# ax.scatter(
#     [best_w],
#     [best_zeta],
#     marker="x",
#     s=70,
#     zorder=11,
#     label="Best fit"
# )

ax.set_xlabel(r"$w$", fontsize=14, fontweight="bold")
ax.set_ylabel(r"$\zeta$", fontsize=17, fontweight="bold", rotation=0)
ax.tick_params(axis="both", labelsize=11)

ax.xaxis.set_label_coords(0.50, -0.10)
ax.yaxis.set_label_coords(-0.15, 0.46)

ax.set_xlim(w_grid.min(), w_grid.max())
ax.set_ylim(zeta_grid.min(), zeta_grid.max())

ax.set_xticks([-1., -0.75, -0.5, -0.25, 0.0])
ax.set_yticks([-1., -0.5, 0., 0.5, 1.0])

from matplotlib.ticker import FormatStrFormatter

ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

# ax.set_title("Dolgov-Barrow profile likelihood")

if AGE_CORRECTION == 0:
    ax.legend(
        framealpha=1.,
        edgecolor="none",
        loc="upper left",
        fontsize=11,
        bbox_to_anchor=(0.03, 0.99)
    )

ax.text(
    0.95, 0.05,
    "(I)  Pre-ABC SNe" if AGE_CORRECTION == 0 else "(II)  Post-ABC SNe",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=12,
    fontweight="bold"
)
    
plt.tight_layout()
plt.show()
    
np.savez(
    "DB_chi2_grid_PreABC.npz",
    w_grid=w_grid,
    zeta_grid=zeta_grid,
    chi2_grid=chi2_grid
)

print("Saved DB_chi2_grid.npz")



sys.exit()