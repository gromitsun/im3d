import numpy as np
from cython.parallel cimport prange
from math_functions cimport sqrtf, fabsf, f_max, f_min, f_sign
#=== 32-bit floats, Godunov's upwind method reinitialization ===================
cpdef void reinit(float[:,::1] phi_0, float[:,::1] phi,
                  float dt=0.4, int max_it=25, float band=3.0, int verbose=0):
    #--- Define variables ------------------------------------------------------
    cdef float[:,::1] phi_t
    cdef ssize_t  x, nx=phi_0.shape[0]
    cdef ssize_t  y, ny=phi_0.shape[1]
    cdef ssize_t  xmin=1, xmax=nx-2  # Bounds of the 'good' data
    cdef ssize_t  ymin=1, ymax=ny-2  # Bounds of the 'good' data
    cdef float  gX, gXm, gXp, gY, gYm, gYp, G
    cdef float  sgn, dist
    cdef float  max_phi, max_err
    cdef int  i, iter=0, num=1
    #--- Initialize arrays -----------------------------------------------------
    phi_t = np.empty(shape=(nx,ny), dtype=np.float32)
    #
    if &phi[0, 0] != &phi_0[0, 0]:
        phi[...] = phi_0[...]
    #--- Do the reinitialization -----------------------------------------------
    iter = 0
    while iter < max_it:
        with nogil:
            iter += 1
            for x in prange(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    sgn = phi_0[x, y] / sqrtf(phi_0[x, y]**2 + 1.0**2)
                    gXm = phi[x+0, y] - phi[x-1, y]
                    gXp = phi[x+1, y] - phi[x-0, y]
                    gYm = phi[x, y+0] - phi[x, y-1]
                    gYp = phi[x, y+1] - phi[x, y-0]
                    if sgn > 0.0:
                        gXm = f_max(gXm, 0.0)
                        gXp = f_min(gXp, 0.0)
                        gYm = f_max(gYm, 0.0)
                        gYp = f_min(gYp, 0.0)
                    else:
                        gXm = f_min(gXm, 0.0)
                        gXp = f_max(gXp, 0.0)
                        gYm = f_min(gYm, 0.0)
                        gYp = f_max(gYp, 0.0)
                    gX = f_max(gXm**2, gXp**2)
                    gY = f_max(gYm**2, gYp**2)
                    G = sqrtf(gX + gY)
                    phi_t[x, y] = sgn * (1.0 - G)
                # end boundary/non-boundary if
              # end y for loop
            # end x for loop
            #--- Update phi and apply BCs --------------------------------------
            for x in prange(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    phi[x,y] = phi[x,y] + dt*phi_t[x,y]
                # end y for loop
            # end x for loop
            phi = constant_value_BC(phi)
            #--- print current status ------------------------------------------
            max_phi = 0.0
            max_err = 0.0
            for x in range(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    max_phi = f_max(fabsf(phi[x,y]), max_phi)
                    if fabsf(phi[x,y]) <= band:
                        max_err = f_max(fabsf(phi_t[x,y]), max_err)
                    # end near-the-interface if statement
                # end y for loop
            # end x for loop
        # end nogil
        if verbose:
            fmts = " | {:5d} | {:10d} | {:10.3f} | {:10.3f} | "
            print(fmts.format(iter, num, max_err, max_phi))
        # end print loop
    # end while loop

#=== Set the first derivative normal to the boundary to zero ===================
cdef float[:, ::1] constant_value_BC(float[:, ::1] arr) nogil:
    #--- Define variables ------------------------------------------------------
    cdef ssize_t x, nx=arr.shape[0]
    cdef ssize_t y, ny=arr.shape[1]
    cdef ssize_t xmin=1, xmax=nx-2  # Bounds of 'good' data
    cdef ssize_t ymin=1, ymax=ny-2  # Bounds of 'good' data
    #--- Apply the boundary conditions -----------------------------------------
    for x in prange(nx):
        arr[x, ymin-1] = arr[x, ymin]
        arr[x, ymax+1] = arr[x, ymax]
    for y in prange(ny):
        arr[xmin-1, y] = arr[xmin, y]
        arr[xmax+1, y] = arr[xmax, y]
    #--- Return result ---------------------------------------------------------
    return arr

#=== Continuous derivative boundary conditions =================================
cdef float[:, ::1] continuous_BC(float[:, ::1] arr) nogil:
    # X and Y derivatives are zero at the boundaries
    #--- Define variables ------------------------------------------------------
    cdef ssize_t x, nx=arr.shape[0]
    cdef ssize_t y, ny=arr.shape[1]
    cdef ssize_t xmin=1, xmax=nx-2  # Bounds of 'good' data
    cdef ssize_t ymin=1, ymax=ny-2  # Bounds of 'good' data
    #--- Apply the boundary conditions -----------------------------------------
    for x in prange(nx):
        arr[x, ymin-1] = arr[x, ymin] + (arr[x, ymin+0] - arr[x, ymin+1])
        arr[x, ymax+1] = arr[x, ymax] + (arr[x, ymax-0] - arr[x, ymax-1])
    for y in prange(ny):
        arr[xmin-1, y] = arr[xmin, y] + (arr[xmin+0, y] - arr[xmin+1, y])
        arr[xmax+1, y] = arr[xmax, y] + (arr[xmax-0, y] - arr[xmax-1, y])
    #--- Return result ---------------------------------------------------------
    return arr
