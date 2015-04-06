import numpy as np
from cython.parallel cimport prange

cpdef void isotropic(double[::1] arr, double[::1] out, int niter=10, double dt=0.25, double Dx=1.0, int bc_type=1):
    # Variables:
    cdef int  i
    cdef ssize_t  x, nx=arr.shape[0]
    cdef double  gXX
    # Arrays:
    cdef double[::1] out1 = np.empty((nx), dtype='float64')
    cdef double[::1] chg = np.empty((nx), dtype='float64')
    out1[1:-1] = arr[:]
    # Apply BC
    
    #=== Smoothing =======================================================
    with nogil:
        for i in range(niter):
            #--- compute central voxels ---
            for x in prange(1, nx-1):
                gXX = + 1.0 * out1[x-1] - 2.0 * out1[x] + 1.0 * out1[x+1]
                chg[x] = Dx * gXX
            # end x for loop
            #--- compute first voxel ---
            x = 0
            gXX = + 2.0 * out1[x+1] - 2.0 * out1[x]
            chg[x] = Dx * gXX
            #--- compute last voxel ---
            x = nx - 1
            gXX = + 2.0 * out1[x-1] - 2.0 * out1[x]
            chg[x] = Dx * gXX
            #--- Update out ---
            for x in prange(nx):
                out1[x] = out1[x] + dt * chg[x]
            # end x for loop
        # end iteration for loop
    # end nogil
    # Return only the central portion of the smoothed array
    out[:] = out1[1:-1]

# ==============================================================
cdef inline void mirror_BC(double[:] in_arr, ssize_t bdry) nogil:
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
# ==============================================================
cdef inline void fixed_BC(double[:] in_arr) nogil:
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
