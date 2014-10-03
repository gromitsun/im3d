import numpy as np
from cython.parallel cimport prange

def isotropic(double[:,::1] arr, int it, double dt, double Dx, double Dy):
    # Variables:
    cdef int  i
    cdef ssize_t  x, nx=arr.shape[0]
    cdef ssize_t  y, ny=arr.shape[1]
    cdef double  gXX, gYY
    # Arrays:
    cdef double[:,::1] out = np.zeros((nx,ny), dtype='float64')
    cdef double[:,::1] chg = np.zeros((nx,ny), dtype='float64')
    out[:,:] = arr[:,:]
    #=== Smoothing =======================================================
    with nogil:
        for i in range(it):
            #--- compute central voxels ---
            for x in prange(1, nx-1):
                for y in range(1, ny-1):
                    gXX = + 1.0 * out[x-1, y] - 2.0 * out[x, y] + 1.0 * out[x+1, y]
                    gYY = + 1.0 * out[x, y-1] - 2.0 * out[x, y] + 1.0 * out[x, y+1]
                    chg[x,y] = Dx*gXX + Dy*gYY
                # end y for loop
            # end x for loop
            #--- compute edge voxels ---
            x = 0
            for y in prange(1, ny-1):
                gXX = + 2.0 * out[x+1, y] - 2.0 * out[x, y]
                gYY = + 1.0 * out[x, y-1] - 2.0 * out[x, y] + 1.0 * out[x, y+1]
                chg[x,y] = Dx*gXX + Dy*gYY
            # end y for loop
            #
            x = nx - 1
            for y in prange(1, ny-1):
                gXX = + 2.0 * out[x-1, y] - 2.0 * out[x, y]
                gYY = + 1.0 * out[x, y-1] - 2.0 * out[x, y] + 1.0 * out[x, y+1]
                chg[x,y] = Dx*gXX + Dy*gYY
            # end y for loop
            #
            y = 0
            for x in prange(1, nx-1):
                gXX = + 1.0 * out[x-1, y] - 2.0 * out[x, y] + 1.0 * out[x+1, y]
                gYY = + 2.0 * out[x, y+1] - 2.0 * out[x, y]
                chg[x,y] = Dx*gXX + Dy*gYY
            # end x for loop
            #
            y = ny - 1
            for x in prange(1, nx-1):
                gXX = + 1.0 * out[x-1, y] - 2.0 * out[x, y] + 1.0 * out[x+1, y]
                gYY = + 2.0 * out[x, y-1] - 2.0 * out[x, y]
                chg[x,y] = Dx*gXX + Dy*gYY
            # end x for loop
            #--- compute corner voxels ---
            x = 0
            y = 0
            gXX = + 2.0 * out[x+1, y] - 2.0 * out[x, y]
            gYY = + 2.0 * out[x, y+1] - 2.0 * out[x, y]
            chg[x,y] = Dx*gXX + Dy*gYY
            #
            x = nx - 1
            y = 0
            gXX = + 2.0 * out[x-1, y] - 2.0 * out[x, y]
            gYY = + 2.0 * out[x, y+1] - 2.0 * out[x, y]
            chg[x,y] = Dx*gXX + Dy*gYY
            #
            x = 0
            y = ny - 1
            gXX = + 2.0 * out[x+1, y] - 2.0 * out[x, y]
            gYY = + 2.0 * out[x, y-1] - 2.0 * out[x, y]
            chg[x,y] = Dx*gXX + Dy*gYY
            #
            x = nx - 1
            y = ny - 1
            gXX = + 2.0 * out[x-1, y] - 2.0 * out[x, y]
            gYY = + 2.0 * out[x, y-1] - 2.0 * out[x, y]
            chg[x,y] = Dx*gXX + Dy*gYY
            #--- Update out ---
            for x in prange(nx):
                for y in range(ny):
                    out[x,y] = out[x,y] + dt * chg[x,y]
                # end y for loop
            # end x for loop
        # end iteration for loop
    # end nogil
    return np.asarray(out, dtype='float64')

