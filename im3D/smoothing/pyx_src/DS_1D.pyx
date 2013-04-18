#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport sqrt, fabs
from cython.parallel cimport prange
# ==============================================================
def isotropic(double[:] in_arr, int iters, double dt, 
    double Dx, int bc_type):
    # ==========================================================
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]+2
    cdef double  gXX
    # Arrays:
    cdef double[:] out   = np.zeros((nX), dtype=np.float64)
    cdef double[:] dA_dt = np.zeros((nX), dtype=np.float64)
    # Fill in values for out; central values with input array
    # and perimeter values using BCs:
    out[1:-1] = in_arr.copy()
    out = mirror_BC(out, 1)
    # ==========================================================
    # Smoothing:
    with nogil:
        for i in range(iters):
            # Compute dA_dt:
            for x in prange(1, nX-1):
                gXX = (+ 1.0*out[x-1] \
                       - 2.0*out[x  ] \
                       + 1.0*out[x+1]) / 1.0
                #
                dA_dt[x] = Dx*gXX
            # end x for loop
            # ======================================================
            # Update out:
            for x in prange(nX):
                out[x] = out[x] + dt*dA_dt[x]
            # end x for loop
            # ======================================================
            # Apply boundary conditions:
            if bc_type==1:
                out = mirror_BC(out, 1)
            else:
                out = fixed_BC(out)
        # end iteration for loop
    # end nogil
    # ==========================================================
    # Return only the central portion of the smoothed array
    return np.asarray(out[1:-1])

# ==============================================================
cdef inline double[:] mirror_BC(double[:] in_arr, ssize_t bdry) nogil:
    """
    Copies the values from just INSIDE the computational domain
    to the ghost points OUTSIDE the computational domain.
    """
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]
    cdef ssize_t  min_x=0, max_x=nX-1
    # ==========================================================
    for i in range(bdry):
        in_arr[min_x+i] = in_arr[min_x+bdry]
        in_arr[max_x-i] = in_arr[max_x-bdry]
    # end for boundary width loop
    return in_arr

# ==============================================================
cdef inline double[:] fixed_BC(double[:] in_arr) nogil:
    """
    Copies the values from the ghost points OUTSIDE the 
    computational domain to the points that are just INSIDE the 
    computational domain.  The ghost points are used here to 
    store the original value of the boundary points.
    """
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]
    cdef ssize_t  min_x=0, max_x=nX-1
    # ==========================================================
    in_arr[min_x+1] = in_arr[min_x]
    in_arr[max_x-1] = in_arr[max_x]
    return in_arr
