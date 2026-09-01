"""
Convert a covariance matrix from the original ASCII .cov format
to a compressed NumPy (.npz) file.

Input
-----
Pantheon+SH0ES_STAT+SYS.cov

Output
------
Pantheon+SH0ES_STAT+SYS.npz

The Pantheon+ covariance file stores the matrix as a flattened
column-major (Fortran-order) array, preceded by the matrix dimension.
This script reconstructs the full covariance matrix and saves it in
compressed NumPy format for faster loading in subsequent analyses.

Author: Hoang Nguyen
"""
import numpy as np

arr = np.loadtxt("Pantheon+SH0ES_STAT+SYS.cov")

N = int(arr[0])
data = arr[1:]
cov = data.reshape((N, N), order="F").copy()

np.save("Pantheon+SH0ES_STAT+SYS.npy", cov)