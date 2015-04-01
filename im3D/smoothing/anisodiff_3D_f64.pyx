import numpy as np
from cython.parallel cimport prange
from libc.math cimport exp


ctypedef double (*cfptr) (double, double) nogil


def anisodiff1(double[:, :, :] arr, int it, double kappa, double dt, double Dx, double Dy, double Dz, int option, double[:, :, :] out):
    """
    This function has bugs.
    """
    cdef int  i
    cdef ssize_t  x, nx=arr.shape[2] + 2
    cdef ssize_t  y, ny=arr.shape[1] + 2
    cdef ssize_t  z, nz=arr.shape[0] + 2
    cdef double[:, :, :] phi_x = np.zeros((nz, ny, nx), dtype='float64')
    cdef double[:, :, :] phi_y = np.zeros((nz, ny, nx), dtype='float64')
    cdef double[:, :, :] phi_z = np.zeros((nz, ny, nx), dtype='float64')

    cdef cfptr gfunc
    if option == 1:
        gfunc = gfunc1
    elif option == 2:
        gfunc = gfunc2
    else:
        raise KeyError("Option %s not understood!" % option)

    if &out[0, 0, 0] != &arr[0, 0, 0]:
        out[:, :, :] = arr.copy()

    with nogil:
        for i in range(it):
            for z in prange(nz - 3):
                for y in range(ny - 3):
                    for x in range(nx - 3):
                        phi_x[z + 1,  y + 1,  x + 1] = (out[z, y, x + 1] - out[z, y, x]) * gfunc(out[z, y, x + 1] - out[z, y, x], kappa)
                        phi_y[z + 1,  y + 1,  x + 1] = (out[z, y + 1, x] - out[z, y, x]) * gfunc(out[z, y + 1, x] - out[z, y, x], kappa)
                        phi_z[z + 1,  y + 1,  x + 1] = (out[z + 1, y, x] - out[z, y, x]) * gfunc(out[z + 1, y, x] - out[z, y, x], kappa)
            # Compute edge values of phi
            z = nz - 2
            for y in prange(ny - 3):
                for x in range(nx - 3):
                    phi_x[z + 1,  y + 1,  x + 1] = (out[z, y, x + 1] - out[z, y, x]) * gfunc(out[z, y, x + 1] - out[z, y, x], kappa)
                    phi_y[z + 1,  y + 1,  x + 1] = (out[z, y + 1, x] - out[z, y, x]) * gfunc(out[z, y + 1, x] - out[z, y, x], kappa)
            y = ny - 2
            for z in prange(nz - 3):
                for x in range(nx - 3):
                    phi_x[z + 1,  y + 1,  x + 1] = (out[z, y, x + 1] - out[z, y, x]) * gfunc(out[z, y, x + 1] - out[z, y, x], kappa)
                    phi_z[z + 1,  y + 1,  x + 1] = (out[z + 1, y, x] - out[z, y, x]) * gfunc(out[z + 1, y, x] - out[z, y, x], kappa)
            x = nx - 2
            for z in prange(nz - 3):
                for y in range(ny - 3):
                    phi_y[z + 1,  y + 1,  x + 1] = (out[z, y + 1, x] - out[z, y, x]) * gfunc(out[z, y + 1, x] - out[z, y, x], kappa)
                    phi_z[z + 1,  y + 1,  x + 1] = (out[z + 1, y, x] - out[z, y, x]) * gfunc(out[z + 1, y, x] - out[z, y, x], kappa)

            z = nz - 2
            y = ny - 2
            for x in prange(nx - 3):
                phi_x[z + 1,  y + 1,  x + 1] = (out[z, y, x + 1] - out[z, y, x]) * gfunc(out[z, y, x + 1] - out[z, y, x], kappa)
                phi_y[z + 1,  y + 1,  x + 1] = (out[z, y + 1, x] - out[z, y, x]) * gfunc(out[z, y + 1, x] - out[z, y, x], kappa)
            y = ny - 2
            x = nx - 2
            for z in prange(nz - 3):
                phi_x[z + 1,  y + 1,  x + 1] = (out[z, y, x + 1] - out[z, y, x]) * gfunc(out[z, y, x + 1] - out[z, y, x], kappa)
                phi_z[z + 1,  y + 1,  x + 1] = (out[z + 1, y, x] - out[z, y, x]) * gfunc(out[z + 1, y, x] - out[z, y, x], kappa)
            z = nz - 2
            x = nx - 2
            for y in prange(ny - 3):
                phi_y[z + 1,  y + 1,  x + 1] = (out[z, y + 1, x] - out[z, y, x]) * gfunc(out[z, y + 1, x] - out[z, y, x], kappa)
                phi_z[z + 1,  y + 1,  x + 1] = (out[z + 1, y, x] - out[z, y, x]) * gfunc(out[z + 1, y, x] - out[z, y, x], kappa)
            # Compute out
            for z in prange(nz - 2):
                for y in range(ny - 2):
                    for x in range(nx - 2):
                        out[z, y, x] += dt * (Dx * (phi_x[z + 1,  y + 1,  x + 1] - phi_x[z + 1, y + 1, x]) +
                                              Dy * (phi_y[z + 1,  y + 1,  x + 1] - phi_y[z + 1, y, x + 1]) +
                                              Dz * (phi_z[z + 1,  y + 1,  x + 1] - phi_z[z, y + 1, x + 1]))
 
                        


def anisodiff(double[:, :, :] arr, int it, double kappa, double dt, double Dx, double Dy, double Dz, int option, double[:, :, :] out):
    cdef int  i
    cdef ssize_t  x, nx=arr.shape[2] + 2
    cdef ssize_t  y, ny=arr.shape[1] + 2
    cdef ssize_t  z, nz=arr.shape[0] + 2
    cdef double[:, :, :] phi_x = np.empty((nz, ny, nx), dtype='float64')
    cdef double[:, :, :] phi_y = np.empty((nz, ny, nx), dtype='float64')
    cdef double[:, :, :] phi_z = np.empty((nz, ny, nx), dtype='float64')
    cdef double[:, :, :] out1 = np.empty((nz, ny, nx), dtype='float64')

    cdef cfptr gfunc
    if option == 1:
        gfunc = gfunc1
    elif option == 2:
        gfunc = gfunc2
    else:
        raise KeyError("Option %s not understood!" % option)

    out1[1:-1, 1:-1, 1:-1] = arr.copy()
    mirror_BC(out1, 1)

    with nogil:
        for i in range(it):
            for z in prange(1, nz - 1):
                for y in range(1, ny - 1):
                    for x in range(1, nx - 1):
                        phi_x[z, y, x] = (out1[z, y, x + 1] - out1[z, y, x]) * gfunc(out1[z, y, x + 1] - out1[z, y, x], kappa)
                        phi_y[z, y, x] = (out1[z, y + 1, x] - out1[z, y, x]) * gfunc(out1[z, y + 1, x] - out1[z, y, x], kappa)
                        phi_z[z, y, x] = (out1[z + 1, y, x] - out1[z, y, x]) * gfunc(out1[z + 1, y, x] - out1[z, y, x], kappa)
            # Compute out
            for z in prange(1, nz - 1):
                for y in range(1, ny - 1):
                    for x in range(1, nx - 1):
                        out1[z, y, x] += dt * (Dx * (phi_x[z, y, x] - phi_x[z, y, x - 1]) +
                                              Dy * (phi_y[z, y, x] - phi_y[z, y - 1, x]) + 
                                              Dz * (phi_z[z, y, x] - phi_z[z - 1, y, x]))
            # Apply BC
            mirror_BC(out1, 1)
    # Return output
    out[:, :, :] = out1[1:-1, 1:-1, 1:-1]


cdef inline double gfunc1(double diff, double kappa) nogil:
    return exp( - (diff / kappa) ** 2.)


cdef inline double gfunc2(double diff, double kappa) nogil:
    return 1. / (1. + (diff / kappa) ** 2.)


# ==============================================================
cdef inline void mirror_BC(double[:,:,:] in_arr, ssize_t bdry) nogil:
    """
    Copies the values from just INSIDE the computational domain
    to the ghost points OUTSIDE the computational domain.
    """
    # Variables:
    cdef int  i
    cdef ssize_t  x, nX=in_arr.shape[0]
    cdef ssize_t  y, nY=in_arr.shape[1]
    cdef ssize_t  z, nZ=in_arr.shape[2]
    cdef ssize_t  min_x=0, max_x=nX-1
    cdef ssize_t  min_y=0, max_y=nY-1
    cdef ssize_t  min_z=0, max_z=nZ-1
    # ==========================================================
    for i in range(bdry):
        for y in prange(nY):
            for z in range(nZ):
                in_arr[min_x+i, y, z] = in_arr[min_x+bdry, y, z]
                in_arr[max_x-i, y, z] = in_arr[max_x-bdry, y, z]
            # end z for loop
        # end y for loop
        for x in prange(nX):
            for z in range(nZ):
                in_arr[x, min_y+i, z] = in_arr[x, min_y+bdry, z]
                in_arr[x, max_y-i, z] = in_arr[x, max_y-bdry, z]
            # end z for loop
        # end x for loop
        for x in prange(nX):
            for y in range(nY):
                in_arr[x, y, min_z+i] = in_arr[x, y, min_z+bdry]
                in_arr[x, y, max_z-i] = in_arr[x, y, max_z-bdry]
            # end y for loop
        # end x for loop
    # end for boundary width loop


# ==============================================================
cdef inline void fixed_BC(double[:,:,:] in_arr) nogil:
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
    cdef ssize_t  z, nZ=in_arr.shape[2]
    cdef ssize_t  min_x=0, max_x=nX-1
    cdef ssize_t  min_y=0, max_y=nY-1
    cdef ssize_t  min_z=0, max_z=nZ-1
    # ==========================================================
    for y in prange(nY):
        for z in range(nZ):
            in_arr[min_x+1, y, z] = in_arr[min_x, y, z]
            in_arr[max_x-1, y, z] = in_arr[max_x, y, z]
        # end z for loop
    # end y for loop
    for x in prange(nX):
        for z in range(nZ):
            in_arr[x, min_y+1, z] = in_arr[x, min_y, z]
            in_arr[x, max_y-1, z] = in_arr[x, max_y, z]
        # end z for loop
    # end x for loop
    for x in prange(nX):
        for y in range(nY):
            in_arr[x, y, min_z+1] = in_arr[x, y, min_z]
            in_arr[x, y, max_z-1] = in_arr[x, y, max_z]
        # end y for loop
    # end x for loop


