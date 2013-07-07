#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport floor
# ==============================================================
def hist2D(arr1, arr2, weight=None, min=None, max=None, int nbins=100, normed=False):
    """
    """
    # ==========================================================
    if weight == None:
        weight = np.ones(arr1, dtype=np.float64)
    # ==========================================================
    # typed variables:
    cdef ssize_t  i, j, bin1, bin2
    cdef double   lo, hi, sz, tot
    # typed arrays:
    cdef double[:] flat_arr1 = np.ravel(arr1).astype(np.float64)
    cdef double[:] flat_arr2 = np.ravel(arr2).astype(np.float64)
    cdef double[:] flat_weight = np.ravel(weight).astype(np.float64)
    cdef double[:,:] X = np.zeros((nbins, nbins), dtype=np.float64)
    cdef double[:,:] Y = np.zeros((nbins, nbins), dtype=np.float64)
    cdef double[:,:] H = np.zeros((nbins, nbins), dtype=np.float64)
    # ==========================================================
    if min==None: lo = min(arr1.min(), arr2.min())
    else:         lo = min
    # 
    if max==None: hi = max(arr1.max(), arr2.max())
    else:         hi = max
    # 
    sz=(hi-lo)/<double>(nbins-1)
    lo = lo-(1./2.) * sz
    hi = hi+(1./2.) * sz
    # ==========================================================
    for i in range(flat_arr1.size):
        bin1 = <int> floor( (flat_arr1[i] - lo) / sz )
        bin2 = <int> floor( (flat_arr2[i] - lo) / sz )
        # 
        if ((bin1 >= 0) and (bin1 < nbins) and 
            (bin2 >= 0) and (bin2 < nbins)):
            H[bin1,bin2] += flat_weight[i]
    # ==========================================================
    for i in range(nbins):
        for j in range(nbins):
            X[i,j] = lo + sz * <double> (i+0.5)
            Y[i,j] = lo + sz * <double> (j+0.5)
    # ==========================================================
    if normed == True:
        tot = np.sum(H)
        for i in range(nbins):
            for j in range(nbins):
                H[i,j] = H[i,j]/(tot*sz)
    # ==========================================================
    return np.asarray(H)
# ==============================================================
