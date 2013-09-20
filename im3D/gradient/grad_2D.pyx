#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

# cython: profile=False
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
cimport cython
from cython.parallel cimport prange
from libc.math cimport sqrt
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def order_1(double[:,:] arr):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cdef:
        ssize_t  x, nX=arr.shape[0]
        ssize_t  y, nY=arr.shape[1]
        double  gX, gY
        #
        double[:,:] grad = np.zeros(shape=(nX,nY), dtype=np.float64)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with nogil:
        for x in prange(1, nX-1):
            for y in range(1, nY-1):
                gX = (arr[x+1, y] - arr[x-1, y])/2.0
                gY = (arr[x, y+1] - arr[x, y-1])/2.0
                grad[x, y] = (gX**2 + gY**2)
            # end y for loop
        # end x for loop
        # 
        # Fill in edge cells:
        for y in range(0, nY):
            grad[   0, y] = grad[   1, y]
            grad[nX-1, y] = grad[nX-2, y]
        # end y for loop
        for x in range(0, nX):
            grad[x,    0] = grad[x,    1]
            grad[x, nY-1] = grad[x, nY-2]
        # end x for loop
    # end nogil
    return np.asarray(grad)
