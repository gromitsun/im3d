#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport floor
# ==============================================================
def hist1D(arr, weight=None, min=None, max=None, int nbins=100, normed=False):
    """
    DESCRIPTION
    ===========
        1D histogram of an array
    
    
    INPUTS
    ======
        arr --> numpy array, list, tuple, etc
                The values for the histogram
                The data can be any rank (e.g. 1D, 2D, ...) and
                just about any datatype (np.array, list, ...)
        
        weight --> numpy array, list, tuple, etc; optional
                The weights assigned to each location
                This array must have the same rank and 
                dimensions as the <arr> array
        
        nbins -> int; optional; default=100
                
        
        min --> float; optional; default=min value of <arr>
                
        
        max --> float; optional; default=max value of <arr>
                
    
    
    OUTPUTS
    =======
        hist -> [2 x nbins] numpy array
                hist[0] is the bin locations
                hist[1] is the frequency value for that bin
    """
    # ==========================================================
    if weight == None:
        weight = np.ones_like(arr, dtype=np.float64)
    # ==========================================================
    # typed variables:
    cdef ssize_t  i, bin
    cdef double   lo, hi, sz, tot
    # typed arrays:
    cdef double[:] flat_arr = np.ravel(arr).astype(np.float64)
    cdef double[:] flat_w = np.ravel(weight).astype(np.float64)
    cdef double[:] X = np.zeros((nbins), dtype=np.float64)
    cdef double[:] H = np.zeros((nbins), dtype=np.float64)
    # ==========================================================
    if min==None: lo = np.min(arr)
    else:         lo = min
    # 
    if max==None: hi = np.max(arr)
    else:         hi = max
    # 
    sz=(hi-lo)/<double>(nbins-1)
    lo = lo-(1./2.) * sz
    hi = hi+(1./2.) * sz
    # ==========================================================
    for i in range(flat_arr.size):
        bin = <int> floor( (flat_arr[i] - lo) / sz )
        if (bin >= 0) and (bin < nbins):
            H[bin] += flat_w[i]
    # ==========================================================
    for i in range(nbins):
        X[i] = lo + sz * <double> (i+0.5)
    # ==========================================================
    if normed == True:
        tot = np.sum(H)
        for i in range(nbins):
            H[i] = H[i]/(tot*sz)
    # ==========================================================
    return np.array((X,H))

# ==============================================================
