import numpy as np
from cython.parallel cimport prange

cpdef void isotropic(float[:,::1] arr, float[:,::1] out, int niter=10,
                     float dt=0.25, float Dx=1.0, float Dy=1.0, int bc_type=1):
    # Variables:
    cdef int  i
    cdef ssize_t  x, nx=arr.shape[0] + 2
    cdef ssize_t  y, ny=arr.shape[1] + 2
    cdef float  gXX, gYY
    # Arrays:
    cdef float[:,::1] out1 = np.empty((nx,ny), dtype='float32')
    cdef float[:,::1] chg = np.empty((nx,ny), dtype='float32')
    out1[1:-1, 1:-1] = arr[:, :]
    # Apply BC
    mirror_BC(out1, 1)
    #=== Smoothing =======================================================
    with nogil:
        for i in range(niter):
            #--- compute central voxels ---
            for x in prange(1, nx-1):
                for y in range(1, ny-1):
                    gXX = + 1.0 * out1[x-1, y] - 2.0 * out1[x, y] + 1.0 * out1[x+1, y]
                    gYY = + 1.0 * out1[x, y-1] - 2.0 * out1[x, y] + 1.0 * out1[x, y+1]
                    chg[x,y] = Dx*gXX + Dy*gYY
                # end y for loop
            # end x for loop
            # ======================================================
            # Update out:
            for x in prange(nx):
                for y in range(ny):
                    out1[x,y] = out1[x,y] + dt*chg[x,y]
                # end y for loop
            # end x for loop
            # ======================================================
            # Apply BC
            if bc_type == 1:
                mirror_BC(out1, 1)
            else:
                fixed_BC(out1)
        # end iteration for loop
    # end nogil
    # Return only the central portion of the smoothed array
    out[:, :] = out1[1:-1, 1:-1]


# ==============================================================
cdef inline void mirror_BC(float[:,:] in_arr, ssize_t bdry) nogil:
    """
    Copies the values from just INSIDE the computational domain
    to the ghost points OUTSIDE the computational domain.
    """
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]
    cdef ssize_t  y, nY=in_arr.shape[1]
    cdef ssize_t  min_x=0, max_x=nX-1
    cdef ssize_t  min_y=0, max_y=nY-1
    # ==========================================================
    for i in range(bdry):
        for y in prange(nY):
            in_arr[min_x+i, y] = in_arr[min_x+bdry, y]
            in_arr[max_x-i, y] = in_arr[max_x-bdry, y]
        # end y for loop
        for x in prange(nX):
            in_arr[x, min_y+i] = in_arr[x, min_y+bdry]
            in_arr[x, max_y-i] = in_arr[x, max_y-bdry]
        # end x for loop
    # end for boundary width loop
# ==============================================================
cdef inline void fixed_BC(float[:,:] in_arr) nogil:
    """
    Copies the values from the ghost points OUTSIDE the 
    computational domain to the points that are just INSIDE the 
    computational domain.  The ghost points are used here to 
    store the original value of the boundary points.
    """
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]
    cdef ssize_t  y, nY=in_arr.shape[1]
    cdef ssize_t  min_x=0, max_x=nX-1
    cdef ssize_t  min_y=0, max_y=nY-1
    # ==========================================================
    for y in prange(nY):
        in_arr[min_x+1, y] = in_arr[min_x, y]
        in_arr[max_x-1, y] = in_arr[max_x, y]
    # end y for loop
    for x in prange(nX):
        in_arr[x, min_y+1] = in_arr[x, min_y]
        in_arr[x, max_y-1] = in_arr[x, max_y]
    # end x for loop
