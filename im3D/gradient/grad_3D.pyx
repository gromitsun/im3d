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
def order_1(double[:,:,:] arr):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cdef:
        ssize_t  x, nX=arr.shape[0]
        ssize_t  y, nY=arr.shape[1]
        ssize_t  z, nZ=arr.shape[2]
        double  gX, gY, gZ
        #
        double[:,:,:] grad = np.zeros(shape=(nX,nY,nZ), dtype=np.float64)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with nogil:
        for x in prange(1, nX-1):
            for y in range(1, nY-1):
                for z in range(1, nZ-1):
                    gX = (arr[x+1, y, z] - arr[x-1, y, z])/2.0
                    gY = (arr[x, y+1, z] - arr[x, y-1, z])/2.0
                    gZ = (arr[x, y, z+1] - arr[x, y, z-1])/2.0
                    grad[x, y, z] = (gX**2 + gY**2 + gZ**2)
                # end z for loop
            # end y for loop
        # end x for loop
        # 
        # Fill in edge cells:
        for y in range(0, nY):
            for z in range(0, nZ):
                grad[   0, y, z] = grad[   1, y, z]
                grad[nX-1, y, z] = grad[nX-2, y, z]
            # end z for loop
        # end y for loop
        for x in range(0, nX):
            for z in range(0, nZ):
                grad[x,    0, z] = grad[x,    1, z]
                grad[x, nY-1, z] = grad[x, nY-2, z]
            # end z for loop
        # end x for loop
        for x in range(0, nX):
            for y in range(0, nY):
                grad[x, y,    0] = grad[x, y,    1]
                grad[x, y, nZ-1] = grad[x, y, nZ-2]
            # end z for loop
        # end x for loop
    # end nogil
    return np.asarray(grad)
