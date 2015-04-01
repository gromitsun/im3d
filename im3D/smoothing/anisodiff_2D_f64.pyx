import numpy as np
from cython.parallel cimport prange
from libc.math cimport exp


ctypedef double (*cfptr) (double, double) nogil


def anisodiff(double[:, :] arr, int it, double kappa, double dt, double Dx, double Dy, int option, double[:, :] out):
    cdef int  i
    cdef ssize_t  x, nx=arr.shape[1]
    cdef ssize_t  y, ny=arr.shape[0]
    # cdef double phi_x[ny][nx], phi_y[ny][nx]
    # print phi_x[0][0]
    cdef double[:, :] phi_x = np.zeros((ny, nx), dtype='float64'), phi_y = np.zeros((ny, nx), dtype='float64')

    cdef cfptr gfunc
    if option == 1:
        gfunc = gfunc1
    elif option == 2:
        gfunc = gfunc2
    else:
        raise KeyError("Option %s not understood!" % option)

    if &out[0, 0] != &arr[0, 0]:
        out[:, :] = arr.copy()

    with nogil:
        for i in range(it):
            for y in prange(ny - 1):
                for x in range(nx - 1):
                    phi_x[y, x] = (out[y, x + 1] - out[y, x]) * gfunc(out[y, x + 1] - out[y, x], kappa)
                    phi_y[y, x] = (out[y + 1, x] - out[y, x]) * gfunc(out[y + 1, x] - out[y, x], kappa)
            # Compute edge values of phi
            y = ny - 1
            for x in prange(nx - 1):
                phi_x[y, x] = (out[y, x + 1] - out[y, x]) * gfunc(out[y, x + 1] - out[y, x], kappa)
            x = nx - 1
            for y in prange(ny - 1):
                phi_y[y, x] = (out[y + 1, x] - out[y, x]) * gfunc(out[y + 1, x] - out[y, x], kappa)
            # Compute out
            for y in prange(1, ny):
                for x in range(1, nx):
                    out[y, x] += dt * (Dx * (phi_x[y, x] - phi_x[y, x - 1]) + Dy * (phi_y[y, x] - phi_y[y - 1, x]))
            # Compute edge values of out
            y = 0
            for x in prange(1, nx):
                out[y, x] += dt * (Dx * (phi_x[y, x] - phi_x[y, x - 1]) + Dy * (phi_y[y, x] - 0))
            x = 0
            for y in prange(1, ny):
                out[y, x] += dt * (Dx * (phi_x[y, x] - 0) + Dy * (phi_y[y, x] - phi_y[y - 1, x]))
            x = 0
            y = 0
            out[y, x] += dt * (Dx * (phi_x[y, x] - 0) + Dy * (phi_y[y, x] - 0))


def anisodiff1(double[:, :] arr, int it, double kappa, double dt, double Dx, double Dy, int option, double[:, :] out):
    """
    This is not right, because when calculating the edge and corner values, 'out' has already been updated.
    :param arr:
    :param it:
    :param kappa:
    :param dt:
    :param Dx:
    :param Dy:
    :param option:
    :param out:
    :return:
    """
    cdef int  i
    cdef ssize_t  x, nx=arr.shape[1]
    cdef ssize_t  y, ny=arr.shape[0]
    cdef double diff_xa, diff_xb, diff_ya, diff_yb, gxa, gxb, gya, gyb

    cdef cfptr gfunc
    if option == 1:
        gfunc = gfunc1
    elif option == 2:
        gfunc = gfunc2
    else:
        raise KeyError("Option %s not understood!" % option)

    if &out[0, 0] != &arr[0, 0]:
        out[:, :] = arr.copy()

    with nogil:
        for i in range(it):
            for y in prange(1, ny - 1):
                for x in range(1, nx - 1):
                    diff_xa = out[y, x] - out[y, x - 1]
                    diff_xb = out[y, x + 1] - out[y, x]
                    diff_ya = out[y, x] - out[y - 1, x]
                    diff_yb = out[y + 1, x] - out[y, x]
                    gxa = gfunc(diff_xa, kappa)
                    gxb = gfunc(diff_xb, kappa)
                    gya = gfunc(diff_ya, kappa)
                    gyb = gfunc(diff_yb, kappa)
                    out[y, x] += dt * (Dx * (gxb * diff_xb - gxa * diff_xa) + Dy * (gyb * diff_yb - gya * diff_yb))
            #--- compute edge voxels ---
            x = 0
            for y in prange(1, ny - 1):
                diff_xa = 0
                diff_xb = out[y, x + 1] - out[y, x]
                diff_ya = out[y, x] - out[y - 1, x]
                diff_yb = out[y + 1, x] - out[y, x]
                gxa = gfunc(diff_xa, kappa)
                gxb = gfunc(diff_xb, kappa)
                gya = gfunc(diff_ya, kappa)
                gyb = gfunc(diff_yb, kappa)
                out[y, x] += dt * (Dx * (gxb * diff_xb - gxa * diff_xa) + Dy * (gyb * diff_yb - gya * diff_yb))
            x = nx - 1
            for y in prange(1, ny - 1):
                diff_xa = out[y, x] - out[y, x - 1]
                diff_xb = 0
                diff_ya = out[y, x] - out[y - 1, x]
                diff_yb = out[y + 1, x] - out[y, x]
                gxa = gfunc(diff_xa, kappa)
                gxb = gfunc(diff_xb, kappa)
                gya = gfunc(diff_ya, kappa)
                gyb = gfunc(diff_yb, kappa)
                out[y, x] += dt * (Dx * (gxb * diff_xb - gxa * diff_xa) + Dy * (gyb * diff_yb - gya * diff_yb))
            y = 0
            for x in prange(1, nx - 1):
                diff_xa = out[y, x] - out[y, x - 1]
                diff_xb = out[y, x + 1] - out[y, x]
                diff_ya = 0
                diff_yb = out[y + 1, x] - out[y, x]
                gxa = gfunc(diff_xa, kappa)
                gxb = gfunc(diff_xb, kappa)
                gya = gfunc(diff_ya, kappa)
                gyb = gfunc(diff_yb, kappa)
                out[y, x] += dt * (Dx * (gxb * diff_xb - gxa * diff_xa) + Dy * (gyb * diff_yb - gya * diff_yb))
            y = ny - 1
            for x in prange(1, nx - 1):
                diff_xa = out[y, x] - out[y, x - 1]
                diff_xb = out[y, x + 1] - out[y, x]
                diff_ya = out[y, x] - out[y - 1, x]
                diff_yb = 0
                gxa = gfunc(diff_xa, kappa)
                gxb = gfunc(diff_xb, kappa)
                gya = gfunc(diff_ya, kappa)
                gyb = gfunc(diff_yb, kappa)
                out[y, x] += dt * (Dx * (gxb * diff_xb - gxa * diff_xa) + Dy * (gyb * diff_yb - gya * diff_yb))
            #--- compute corner voxels ---
            x = 0
            y = 0
            diff_xa = 0
            diff_xb = out[y, x + 1] - out[y, x]
            diff_ya = 0
            diff_yb = out[y + 1, x] - out[y, x]
            gxa = gfunc(diff_xa, kappa)
            gxb = gfunc(diff_xb, kappa)
            gya = gfunc(diff_ya, kappa)
            gyb = gfunc(diff_yb, kappa)
            out[y, x] += dt * (Dx * (gxb * diff_xb - gxa * diff_xa) + Dy * (gyb * diff_yb - gya * diff_yb))

            x = 0
            y = ny - 1
            diff_xa = 0
            diff_xb = out[y, x + 1] - out[y, x]
            diff_ya = out[y, x] - out[y - 1, x]
            diff_yb = 0
            gxa = gfunc(diff_xa, kappa)
            gxb = gfunc(diff_xb, kappa)
            gya = gfunc(diff_ya, kappa)
            gyb = gfunc(diff_yb, kappa)
            out[y, x] += dt * (Dx * (gxb * diff_xb - gxa * diff_xa) + Dy * (gyb * diff_yb - gya * diff_yb))

            x = nx - 1
            y = 0
            diff_xa = out[y, x] - out[y, x - 1]
            diff_xb = 0
            diff_ya = 0
            diff_yb = out[y + 1, x] - out[y, x]
            gxa = gfunc(diff_xa, kappa)
            gxb = gfunc(diff_xb, kappa)
            gya = gfunc(diff_ya, kappa)
            gyb = gfunc(diff_yb, kappa)
            out[y, x] += dt * (Dx * (gxb * diff_xb - gxa * diff_xa) + Dy * (gyb * diff_yb - gya * diff_yb))

            x = nx - 1
            y = ny - 1
            diff_xa = out[y, x] - out[y, x - 1]
            diff_xb = 0
            diff_ya = out[y, x] - out[y - 1, x]
            diff_yb = 0
            gxa = gfunc(diff_xa, kappa)
            gxb = gfunc(diff_xb, kappa)
            gya = gfunc(diff_ya, kappa)
            gyb = gfunc(diff_yb, kappa)
            out[y, x] += dt * (Dx * (gxb * diff_xb - gxa * diff_xa) + Dy * (gyb * diff_yb - gya * diff_yb))


cdef inline void diff(double[:, :] arr, double[:, :] diff_x, double[:, :] diff_y) nogil:
    """
    Return finite difference
    :param arr: Input array.
    :param diff_x: diff_x = arr[:, 1:] - arr[:, :-1]
    :param diff_y: diff_y = arr[1:, :] - arr[:-1, :]
    :return:
    """
    # ==========================================================
    # Variables:
    cdef ssize_t  x, nx = arr.shape[1]
    cdef ssize_t  y, ny = arr.shape[0]
    # ==========================================================
    for y in prange(ny - 1):
        for x in range(nx - 1):
            diff_x[y, x] = arr[y, x + 1] - arr[y, x]
            diff_y[y, x] = arr[y + 1, x] - arr[y, x]


cdef inline void phi(double[:, :] arr, double[:, :] phi_x, double[:, :] phi_y, cfptr gfunc, double kappa) nogil:
    """
    Return phi = diff * g(diff).
    :param arr:
    :param phi_x:
    :param phi_y:
    :return:
    """
    # ==========================================================
    # Variables:
    cdef ssize_t  x, nx = arr.shape[1]
    cdef ssize_t  y, ny = arr.shape[0]
    # ==========================================================
    for y in prange(ny - 1):
        for x in range(nx - 1):
            phi_x[y, x] = (arr[y, x + 1] - arr[y, x]) * gfunc(arr[y, x + 1] - arr[y, x], kappa)
            phi_y[y, x] = (arr[y + 1, x] - arr[y, x]) * gfunc(arr[y + 1, x] - arr[y, x], kappa)


cdef inline double gfunc1(double diff, double kappa) nogil:
    return exp( - (diff / kappa) ** 2.)


cdef inline double gfunc2(double diff, double kappa) nogil:
    return 1. / (1. + (diff / kappa) ** 2.)