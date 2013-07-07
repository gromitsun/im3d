#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange
# ==============================================================
def rms(arr):
    """
    DESCRIPTION
    ===========
        Root mean squared (rms) value of an array
    
    
    INPUTS
    ======
        arr --> numpy array
                The array for which the rms value will be 
                calculated
    
    
    OUTPUTS
    =======
        rms --> float
                rms value of the array
    
    """
    # Arrays:
    cdef double[:] cy_arr = np.ravel(arr)
    # Variables
    cdef ssize_t  i, n=cy_arr.size
    cdef double   err, 
    #
    with nogil:
      for i in prange(n):
        err += cy_arr[i]*cy_arr[i]
      # end for loop
    # end nogil
    err = err/n
    err = sqrt(err)
    #
    return err

