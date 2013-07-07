#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport sqrt, fabs
from cython.parallel cimport prange
# ==============================================================
def isotropic(double[:,:] in_arr, int iters, 
    double dt, double Dx, double Dy, int bc_type):
    # ==========================================================
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]+2
    cdef ssize_t  y, nY=in_arr.shape[1]+2
    cdef double  gXX, gYY
    # Arrays:
    cdef double[:,:] out   = np.zeros((nX,nY), dtype=np.float64)
    cdef double[:,:] dA_dt = np.zeros((nX,nY), dtype=np.float64)
    # Fill in values for out; central values with input array
    # and perimeter values using BCs:
    out[1:-1, 1:-1] = in_arr.copy()
    out = mirror_BC(out, 1)
    # ==========================================================
    # Smoothing:
    with nogil:
        for i in range(iters):
            # Compute dA_dt:
            for x in prange(1, nX-1):
                for y in range(1, nY-1):
                    gXX = (+ 1.0*out[x-1,y  ]\
                           - 2.0*out[x  ,y  ]\
                           + 1.0*out[x+1,y  ])/1.0
                    gYY = (+ 1.0*out[x  ,y-1]\
                           - 2.0*out[x  ,y  ]\
                           + 1.0*out[x  ,y+1])/1.0
                    #
                    dA_dt[x,y] = Dx*gXX + Dy*gYY
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
            if bc_type==1:
                out = mirror_BC(out, 1)
            else:
                out = fixed_BC(out)
        # end iteration for loop
    # end nogil
    # ==========================================================
    # Return only the central portion of the smoothed array
    return np.asarray(out[1:-1, 1:-1])

# ==============================================================
def anisotropic(double[:,:] in_arr, int iters, 
    double dt, int bc_type):
    # ==========================================================
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]+2
    cdef ssize_t  y, nY=in_arr.shape[1]+2
    cdef double   g_x, g_y, g_xx, g_yy, grad
    cdef double   N_x, N_y, N, T_x, T_y
    # Arrays:
    cdef double[:,:] out   = np.zeros((nX,nY), dtype=np.float64)
    cdef double[:,:] dA_dt = np.zeros((nX,nY), dtype=np.float64)
    # Fill in values for out; central values with input array
    # and perimeter values using BCs:
    out[1:-1, 1:-1] = in_arr.copy()
    out = mirror_BC(out, 1)
    # ==========================================================
    # Smoothing:
    with nogil:
        for i in range(iters):
            # Compute dA_dt:
            for x in prange(1, nX-1):
                for y in range(1, nY-1):
                    # first derivatives:
                    g_x = (- 1.0*out[x-1,y  ]\
                           + 1.0*out[x+1,y  ])/2.0
                    g_y = (- 1.0*out[x  ,y-1]\
                           + 1.0*out[x  ,y+1])/2.0
                    # second derivatives:
                    g_xx = (+ 1.0*out[x-1,y  ]\
                            - 2.0*out[x  ,y  ]\
                            + 1.0*out[x+1,y  ])/1.0
                    g_yy = (+ 1.0*out[x  ,y-1]\
                            - 2.0*out[x  ,y  ]\
                            + 1.0*out[x  ,y+1])/1.0
                    # magnitude of the first derivatives
                    grad = sqrt(g_x*g_x + g_y*g_y)
                    # normal components
                    N_x = g_x/grad
                    N_y = g_y/grad
                    # magnitude of the normal
                    N = sqrt(N_x*N_x + N_y*N_y)
                    # tangent components
                    T_x = fabs(+N_y/N)
                    T_y = fabs(-N_x/N)
                    #
                    dA_dt[x,y] = T_x*g_xx + T_y*g_yy
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
            if bc_type==1:
                out = mirror_BC(out, 1)
            else:
                out = fixed_BC(out)
        # end iteration for loop
    # end nogil
    # ==========================================================
    # Return only the central portion of the smoothed array
    return np.asarray(out[1:-1, 1:-1])

# ==============================================================
cdef inline double[:,:] mirror_BC(double[:,:] in_arr, ssize_t bdry) nogil:
    """
    Copies the values from just INSIDE the computational domain
    to the ghost points OUTSIDE the computational domain.
    """
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
# ==============================================================
cdef inline double[:,:] fixed_BC(double[:,:] in_arr) nogil:
    """
    Copies the values from the ghost points OUTSIDE the 
    computational domain to the points that are just INSIDE the 
    computational domain.  The ghost points are used here to 
    store the original value of the boundary points.
    """
    # ==========================================================
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]
    cdef ssize_t  y, nY=in_arr.shape[1]
    cdef ssize_t  min_x=0, max_x=nX-1
    cdef ssize_t  min_y=0, max_y=nY-1
    # ==========================================================
    for y in range(nY):
        in_arr[min_x+1, y] = in_arr[min_x, y]
        in_arr[max_x-1, y] = in_arr[max_x, y]
    # end y for loop
    for x in range(nX):
        in_arr[x, min_y+1] = in_arr[x, min_y]
        in_arr[x, max_y-1] = in_arr[x, max_y]
    # end x for loop
    return in_arr
