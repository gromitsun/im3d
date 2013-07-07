#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport log, floor
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
    if weight == None:
        weight = np.ones_like(arr)
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
            H[i] = H[i] * <double> (nbins-1) * (hi-lo-sz)/(tot)
    # ==========================================================
    return np.array((X,H))


# ==============================================================
def hist2D(arr1, arr2, bin_min=None, bin_max=None, nbins=100):
    cdef ssize_t  i, j, bin1, bin2, n=arr1.size, nB=nbins
    cdef double   lo, hi, sz
    #
    cdef double[:] flat1=np.ravel(arr1)
    cdef double[:] flat2=np.ravel(arr2)
    cdef double[:,:] HJ = np.zeros((nB, nB), dtype=np.float64)
    #
    if bin_min==None: lo = min(arr1.min(), arr2.min())
    else:             lo = bin_min
    
    if bin_max==None: hi = max(arr1.max(), arr2.max())
    else:             hi = bin_max
    
    sz=(hi-lo)/<double>(nB-1)
    #
    for i in range(n):
        bin1 = <int> floor( (flat1[i] - lo) / sz + 0.5)
        bin2 = <int> floor( (flat2[i] - lo) / sz + 0.5)
        # 
        if ((bin1 >= 0) and (bin1 < nB) and (bin2 >= 0) and (bin2 < nB)):
            HJ[bin1,bin2] += 1.0
    #
    return np.asarray(HJ)
# ==============================================================
def MI(arr1, arr2, bin_min=None, bin_max=None, nbins=100):
    cdef ssize_t  i, j, bin1, bin2, n=arr1.size, nB=nbins
    cdef double   MI=0.0, arr_sz=arr1.size
    cdef double   lo, hi, sz
    #
    cdef double[:] flat1=np.ravel(arr1)
    cdef double[:] flat2=np.ravel(arr2)
    #
    cdef double[:] H1 = np.zeros((nB), dtype=np.float64)
    cdef double[:] H2 = np.zeros((nB), dtype=np.float64)
    cdef double[:,:] HJ = np.zeros((nB, nB), dtype=np.float64)
    #
    if bin_min==None: lo = min(arr1.min(), arr2.min())
    else:             lo = bin_min
    
    if bin_max==None: hi = max(arr1.max(), arr2.max())
    else:             hi = bin_max
    
    sz=(hi-lo)/<double>(nB-1)
    #
    for i in range(n):
        bin1 = <int> floor( (flat1[i] - lo) / sz + 0.5)
        bin2 = <int> floor( (flat2[i] - lo) / sz + 0.5)
        # 
        if ((bin1 >= 0) and (bin1 < nB) and (bin2 >= 0) and (bin2 < nB)):
            H1[bin1] += 1.0
            H2[bin2] += 1.0
            HJ[bin1,bin2] += 1.0
    # ==========================================================
    for i in range(nB):
        if H1[i] > 0.0:
            MI += -(H1[i]/arr_sz) * log(H1[i]/arr_sz)
        if H2[i] > 0.0:
            MI += -(H2[i]/arr_sz) * log(H2[i]/arr_sz)
        for j in range(nB):
            if HJ[i,j] > 0.0:
                MI += (HJ[i,j]/arr_sz) * log(HJ[i,j]/arr_sz)
    #
    return MI
