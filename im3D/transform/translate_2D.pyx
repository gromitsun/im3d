#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

# cython: profile=False
# ==============================================================
import numpy as np
#
from libc.math cimport floor
#
cimport cython
from cython.parallel cimport prange
# ==============================================================
def translate_2D(double[:,:] in_arr, double dX, double dY):
    """
    dX, dY --> Amount of translation
    """
    # ==========================================================
    cdef:
        ssize_t  x, nX=in_arr.shape[0]
        ssize_t  y, nY=in_arr.shape[1]
        ssize_t  x_lo, y_lo       # 
        ssize_t  x_hi, y_hi       # 
        double   x_est, y_est     # Estimates of where (x,y) in
                                  # the out-array corresponds to
                                  # in the input array
        double   Wx_lo, Wy_lo, Wx_hi, Wy_hi
        double   W_LL, W_LH, W_HL, W_HH
        double   eps = 1E-6
        double[:,:] arr = np.zeros((nX,nY), dtype=np.float64)
    # ==========================================================
    with nogil:
        for x in prange(nX):
            for y in range(nY):
                # translate (x,y) by (dX,dY)
                x_est = <double> x - dX
                y_est = <double> y - dY
                # interpolate
                x_lo = <int> floor(x_est)
                y_lo = <int> floor(y_est)
                x_hi = x_lo + 1 #ceil(x_est)
                y_hi = y_lo + 1 #ceil(y_est)
                # 1D interpolation weights
                Wx_lo = - x_est + <double> x_hi + eps
                Wy_lo = - y_est + <double> y_hi + eps
                Wx_hi = + x_est - <double> x_lo + eps
                Wy_hi = + y_est - <double> y_lo + eps
                # 2D interpolation weights
                W_LL = Wx_lo * Wy_lo
                W_LH = Wx_lo * Wy_hi
                W_HL = Wx_hi * Wy_lo
                W_HH = Wx_hi * Wy_hi
                #
                if ((x_lo >= 0) and (x_hi < nX) and 
                    (y_lo >= 0) and (y_hi < nY)):
                    arr[x,y] = W_LL * in_arr[x_lo, y_lo] + \
                               W_LH * in_arr[x_lo, y_hi] + \
                               W_HL * in_arr[x_hi, y_lo] + \
                               W_HH * in_arr[x_hi, y_hi]
            # end for y loop
        # end for x loop
    return np.asarray(arr)
# ==============================================================
