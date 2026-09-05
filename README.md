# Reproducibility Package

This repository contains the Python scripts used to reproduce the main numerical analyses and figures of the manuscript

*Empirical indications from SNe Ia and H II galaxies toward a dynamical-speed-of-light cosmology*.

---

# Requirements

The scripts were developed with Python 3.

Required packages:

```text
numpy
scipy
pandas
matplotlib
```

No additional packages are required.

---

# Input data

The analyses use the public Pantheon+ and DES-SN5YR data. The data files are not included in this repository and should be downloaded from the official Pantheon+ and DES-SN5YR sites:
https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR
https://github.com/des-science/DES-SN5YR/tree/main/4_DISTANCES_COVMAT

Required files (downloadable from Pantheon+ and DES-SN5YR sites):

```
1) Pantheon+SH0ES.dat
2) Pantheon+SH0ES_STAT+SYS.cov
```
(Note: Before running the analyses, convert this .cov file once to a NumPy array by running `convert_cov_to_npy.py`. This generates `Pantheon+SH0ES_STAT+SYS.npy` which is subsequently used by all fitting scripts.)

```
3) DES-Dovekie_HD.csv
4) STAT+SYS.npz
```
(Note: The DES covariance matrix is distributed under the filename `STAT+SYS.npz` in the official DES-SN5YR data release. Our scripts adopt this original filename without renaming it.)

The Pantheon+ covariance matrix is stored as a NumPy array (`.npy`), while the DES covariance matrix is read from the published compressed inverse covariance (`STAT+SYS.npz`).

---

# Repository structure

```text
convert_cov_to_npy.py
    Converts the Pantheon+ covariance matrix to NumPy format.

wCDM_fit_floatH0.py
    Computes the flat-wCDM best fits for Pantheon+ and DES,
    with the corresponding H0 values adjusted independently.

wCDM_fit_fixedH0.py
    Computes the flat-wCDM best fits for Pantheon+ and DES,
    with the corresponding H0 values fixed.

Fig1_wCDM_PrePostABC.py
    Reproduces the eight panels of Fig. 1, showing the pre-ABC
    and post-ABC flat-wCDM confidence regions and redshift-split
    diagnostics.

Fig2_DolgovBarrow_fit.py
    Reproduces the two panels of Fig. 2 in the Dolgov-Barrow
    (w, zeta) parameter plane.
    
---

# Age correction

The progenitor-age correction used throughout the repository is $\Delta\mu(z)=0.183\left[1-\exp(-2.2z)\right]$ which is subtracted from the observed distance modulus whenever `AGE_CORRECTION = 1` is selected.

---

# Running the Scripts

To reproduce the numerical results presented in the manuscript, the scripts should be executed in the following order:

## Step 0. Download the supernova data

Download the Pantheon+ and DES-SN5YR data (see the **Input data** section) and place the four required files in the same directory as the scripts.

## Step 1. Convert the Pantheon+ covariance matrix
```
convert_cov_to_npy.py
```

This converts the official Pantheon+ covariance matrix `Pantheon+SH0ES_STAT+SYS.cov` into the NumPy binary file `Pantheon+SH0ES_STAT+SYS.npy` which is subsequently used by all fitting scripts.

## Step 2. Produce Figs 1 and 2:
```
Fig1_wCDM_PrePostABC.py
Fig2_DolgovBarrow_fit.py
```

## Step 3. Compute the best-fit wCDM parameters
```
wCDM_fit_floatH0.py
wCDM_fit_fixedH0.py```
```

---

# Numerical methodology:

The analyses employ generalized least squares using the published covariance matrices.

For the Pantheon+ and DES joint analyses, the two surveys are combined using a block-diagonal covariance matrix.

When fitting the joint datasets, the parameters $H_{0}^{\rm Pan}$ and $H_{0}^{\rm DES}$ are analytically profiled independently. No other nuisance parameters are introduced.
---

# Notes on Reproducibility

The plotting scripts are intended to reproduce the scientific results presented in the manuscript.

The scripts have therefore been written to emphasize clarity and reproducibility rather than computational efficiency or publication-quality graphics.
