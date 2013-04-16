#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

# cython: profile=False
# ==============================================================
import numpy as np
from libc.math cimport sin, cos, sqrt, fabs, atan, atan2, floor, ceil
from cython.parallel cimport prange
# ==============================================================
cdef double[:,:] rotate(double[:,:] in_arr, double theta, 
    double x_ctr, double y_ctr):
    """
    theta --> rotation amount in degrees
    x_ctr, y_ctr --> center of rotation
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
        double   rad, phi         # (x,y) in polar coord
        double   Wx_lo, Wy_lo, Wx_hi, Wy_hi
        double   W_LL, W_LH, W_HL, W_HH
        double[:,:] arr = np.zeros((nX,nY), dtype=np.float64)
    # ==========================================================
    with nogil:
        for x in prange(nX):
            for y in range(nY):
                x_est = <double> x
                y_est = <double> y
                # Translate so that the center of rotation is (0,0)
                x_est = x_est - x_ctr
                y_est = y_est - y_ctr
                # determine polar coordinates, rotate the polar 
                # coordinates and transform back to cartesian
                rad = sqrt(x_est**2 + y_est**2)
                phi = atan2(y_est, x_est)
                x_est = rad*cos(phi-theta)
                y_est = rad*sin(phi-theta)
                # Invert translating the center of rotation to (0,0)
                x_est = x_est + x_ctr
                y_est = y_est + y_ctr
                # interpolate
                x_lo = <int> floor(x_est)
                y_lo = <int> floor(y_est)
                x_hi = x_lo + 1 #ceil(x_est)
                y_hi = y_lo + 1 #ceil(y_est)
                # 1D interpolation weights
                Wx_lo = - x_est + x_hi
                Wy_lo = - y_est + y_hi
                Wx_hi = + x_est - x_lo
                Wy_hi = + y_est - y_lo
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
    return arr
# ==============================================================

## determine polar coordinates
#x_est = <double> x - x_ctr
#y_est = <double> y - y_ctr
#rad = sqrt(x_est**2 + y_est**2)
#phi = atan2(y_est, x_est)
## rotate the polar coordinates and transform
## back to cartesian
#x_est = x_ctr + rad*cos(phi-theta)
#y_est = y_ctr + rad*sin(phi-theta)
