import numpy as np
from cython.parallel cimport prange

def isotropic(float[::1] arr, int it, float dt, float Dx):
    # Variables:
    cdef int  i
    cdef ssize_t  x, nx=arr.shape[0]
    cdef float  gXX
    # Arrays:
    cdef float[::1] out = np.zeros((nx), dtype='float32')
    cdef float[::1] chg = np.zeros((nx), dtype='float32')
    out[:] = arr[:]
    #=== Smoothing =======================================================
    with nogil:
        for i in range(it):
            #--- compute central voxels ---
            for x in prange(1, nx-1):
                gXX = + 1.0 * out[x-1] - 2.0 * out[x] + 1.0 * out[x+1]
                chg[x] = Dx * gXX
            # end x for loop
            #--- compute first voxel ---
            x = 0
            gXX = + 2.0 * out[x+1] - 2.0 * out[x]
            chg[x] = Dx * gXX
            #--- compute last voxel ---
            x = nx - 1
            gXX = + 2.0 * out[x-1] - 2.0 * out[x]
            chg[x] = Dx * gXX
            #--- Update out ---
            for x in prange(nx):
                out[x] = out[x] + dt * chg[x]
            # end x for loop
        # end iteration for loop
    # end nogil
    return np.asarray(out, dtype='float32')

