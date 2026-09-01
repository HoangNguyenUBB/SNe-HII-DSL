# Reproducibility Package

This repository contains the Python scripts used to reproduce the numerical results, tables, and figures of the manuscript *Indications of a parsimonious luminosity-distance relation in progenitor-age-corrected Type Ia supernova data*.

This repository contains all Python scripts required to reproduce the numerical analyses, tables, and figures presented in the manuscript.

---

# Requirements

The scripts were developed with Python 3.

Required packages

```text
numpy
scipy
pandas
matplotlib
```

No additional packages are required.

---

### SciPy compatibility

The scripts were originally developed using the `cumtrapz` function from `scipy.integrate`.

In recent versions of SciPy, `cumtrapz` has been renamed to `cumulative_trapezoid`. If an `ImportError` occurs, simply replace

```python
from scipy.integrate import cumtrapz
```

with
```python
from scipy.integrate import cumulative_trapezoid as cumtrapz
```

No other modifications are required.

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

The workflow is

```text
Raw data
      │
      ▼
fit_*.py
      │
      ▼
intermediate output (*.txt)
      │
      ├──► make_Table_*.py
      └──► make_Figure_*.py

```

The fitting scripts perform the numerical analyses and generate intermediate output files used by the table and figure generation scripts.

The table scripts regenerate the manuscript tables.

The figure scripts regenerate the manuscript figures.

---

# Age correction

The progenitor-age correction used throughout the repository is $\Delta\mu(z)=0.183\left[1-\exp(-2.2z)\right]$ which is subtracted from the observed distance modulus whenever `AGE_CORRECTION = 1` is selected. (The distributed scripts have already been configured with the appropriate value of `AGE_CORRECTION` for each analysis (original or age-corrected data). Users should not modify this setting.)

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

## Step 2. Run the fitting scripts:

#### Flat ΛCDM

```
fit_LCDM.Pan_original.py
fit_LCDM.DES_original.py
fit_LCDM.Joint_original.py
fit_LCDM.Pan_age_corrected.py
fit_LCDM.DES_age_corrected.py
fit_LCDM.Joint_age_corrected.py
```

#### Flat wCDM

```
fit_wCDM.Pan_original.py
fit_wCDM.DES_original.py
fit_wCDM.Joint_original.py
fit_wCDM.Pan_age_corrected.py
fit_wCDM.DES_age_corrected.py
fit_wCDM.Joint_age_corrected.py
```

#### Logarithmic luminosity-distance relation

```
fit_Loga.Pan_original.py
fit_Loga.DES_original.py
fit_Loga.Joint_original.py
fit_Loga.Pan_age_corrected.py
fit_Loga.DES_age_corrected.py
fit_Loga.Joint_age_corrected.py
```

#### Quadratic logarithmic relation

```
fit_QuadraticLoga.Joint_original.py
fit_QuadraticLoga.Joint_age_corrected.py
```

The fitting scripts generate intermediate output text files containing only the numerical quantities required by the plotting and table-generation scripts.

## Step 3. Generate the manuscript tables

```
make_Table_1.py
make_Table_C1.py
make_Tables_D1_and_D2.py
make_Table_E1.py
make_Tables_G1_and_G2.py
```

## Step 4. Generate the manuscript figures

```
make_Figure_1.py
make_Figure_2_upper_panel.py
make_Figure_2_lower_panel.py
make_Figure_3.py
```

## Optional: Run the complete workflow

For convenience, the repository also includes

```
RUN_ALL.py
```

which executes all scripts in the correct order to reproduce the complete set of numerical results, tables, and figures. The total runtime is approximately 5 minutes.

---

# Numerical methodology:

The analyses employ generalized least squares using the published covariance matrices.

For the Pantheon+ and DES joint analyses, the two surveys are combined using a block-diagonal covariance matrix.

When fitting the joint datasets, the parameters $H_{0}^{\rm Pan}$ and $H_{0}^{\rm DES}$ are analytically profiled independently. No other nuisance parameters are introduced.

---

# Output

The fitting scripts generate intermediate output text files that are subsequently used by the table and figure generation scripts. These intermediate files should not be edited manually.

Running the scripts in the order described above reproduces

* Best-fit cosmological parameters and their error bars,
* Likelihood contours,
* Manuscript tables,
* Manuscript figures.

---

# Notes on Reproducibility

The plotting scripts are intended to reproduce the scientific results presented in the manuscript.

The scripts have therefore been written to emphasize clarity and reproducibility rather than computational efficiency or publication-quality graphics.
