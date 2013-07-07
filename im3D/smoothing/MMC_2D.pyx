#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport sqrt, fabs
from cython.parallel cimport prange
# ==============================================================
def MMC_2D(double[:,:] in_arr, int iters, double dt, double eps):
    # ==========================================================
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]+2
    cdef ssize_t  y, nY=in_arr.shape[1]+2
    cdef double  gX, gY, gXX, gYY, gXY
    cdef double  grad, H
    # Arrays:
    cdef double[:,:] out   = np.zeros((nX,nY), dtype=np.float64)
    cdef double[:,:] dA_dt = np.zeros((nX,nY), dtype=np.float64)
    # Fill in values for out; central values with input array
    # and perimeter values using BCs:
    out[1:-1, 1:-1] = in_arr.copy()
    out = apply_BCs(out, 1)
    # ==========================================================
    with nogil:
        for i in range(iters):
            # compute dA_dt:
            for x in prange(1, nX-1):
                for y in range(1, nY-1):
                    gX = (- 1.0*out[x-1, y]\
                          + 1.0*out[x+1, y])/2.0
                    gY = (- 1.0*out[x, y-1]\
                          + 1.0*out[x, y+1])/2.0
                    #
                    gXX = (+ 1.0*out[x-1,y  ]\
                           - 2.0*out[x  ,y  ]\
                           + 1.0*out[x+1,y  ])/1.0
                    gYY = (+ 1.0*out[x  ,y-1]\
                           - 2.0*out[x  ,y  ]\
                           + 1.0*out[x  ,y+1])/1.0
                    #
                    gXY = (+ 1.0*out[x-1, y-1] \
                           - 1.0*out[x-1, y+1] \
                           - 1.0*out[x+1, y-1] \
                           + 1.0*out[x+1, y+1])/4.0
                    #
                    grad = sqrt(gX*gX + gY*gY)
                    #
                    if grad >= eps:
                        H = (+ gX*gX*gYY       \
                             + gY*gY*gXX       \
                             - 2.*gX*gY*gXY)   \
                             / (grad**3)
                    else:
                        H = 0.0
                    #
                    dA_dt[x,y] = H
                # end y for loop
            # end x for loop
            # ======================================================
            # Update out:
            for x in prange(nX):
                for y in range(nY):
                    out[x,y] = out[x,y] + dt*dA_dt[x,y]
                # end y for loop
            # end x for loop
            # ======================================================
            # Apply boundary conditions:
            out = apply_BCs(out, 1)
        # end iteration for loop
    # end nogil
    # ==========================================================
    # Return only the central portion of the smoothed array
    return np.asarray(out[1:-1, 1:-1])

# ==============================================================
cdef inline double[:,:] apply_BCs(double[:,:] in_arr, ssize_t bdry) nogil:
    # ==========================================================
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]
    cdef ssize_t  y, nY=in_arr.shape[1]
    cdef ssize_t  min_x=0, max_x=nX-1
    cdef ssize_t  min_y=0, max_y=nY-1
    # ==========================================================
    for i in range(bdry):
        for y in range(nY):
            in_arr[min_x+i, y] = in_arr[min_x+bdry, y]
            in_arr[max_x-i, y] = in_arr[max_x-bdry, y]
        # end y for loop
        for x in range(nX):
            in_arr[x, min_y+i] = in_arr[x, min_y+bdry]
            in_arr[x, max_y-i] = in_arr[x, max_y-bdry]
        # end x for loop
    # end for boundary width loop
    return in_arr
