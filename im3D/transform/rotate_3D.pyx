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
def rotate_3D(double[:,:,:] in_arr,
    double x_rot, double y_rot, double z_rot,
    double x_ctr, double y_ctr, double z_ctr):
    """
    x_rot, y_rot, z_rot --> rotation amount in degrees
    x_ctr, y_ctr, z_ctr --> center of rotation
    """
    # ==========================================================
    cdef:
        ssize_t  x, nX=in_arr.shape[0]
        ssize_t  y, nY=in_arr.shape[1]
        ssize_t  z, nZ=in_arr.shape[2]
        ssize_t  x_lo, y_lo, z_lo       # 
        ssize_t  x_hi, y_hi, z_hi       # 
        double   x_est, y_est, z_est    # Estimates of where (x,y) in
                                        # the out-array corresponds to
                                        # in the input array
        double   x_rad, y_rad, z_rad    # radii for (x,y,z) in polar coord
        double   x_phi, y_phi, z_phi    # angles for (x,y,z) in polar coord
        double   Wx_lo, Wy_lo, Wz_lo
        double   Wx_hi, Wy_hi, Wz_hi
        double   W_LLL, W_LLH, W_LHL, W_LHH, W_HLL, W_HLH, W_HHL, W_HHH
        double[:,:,:] arr = np.zeros((nX,nY,nZ), dtype=np.float64)
    # ==========================================================
    with nogil:
      for x in prange(nX):
        for y in range(nY):
          for z in range(nZ):
            x_est = <double> x
            y_est = <double> y
            z_est = <double> z
            # Translate so that the center of rotation is (0,0,0)
            x_est = x_est - x_ctr
            y_est = y_est - y_ctr
            z_est = z_est - z_ctr
            # determine polar coordinates, rotate the polar 
            # coordinates and transform back to cartesian
            ### Rotate about the z-axis ###
            z_rad = sqrt(x_est**2 + y_est**2)
            z_phi = atan2(y_est, x_est)
            x_est = z_rad*cos(z_phi-z_rot)
            y_est = z_rad*sin(z_phi-z_rot)
            ### Rotate about the y-axis ###
            y_rad = sqrt(x_est**2 + z_est**2)
            y_phi = atan2(x_est, z_est)
            z_est = y_rad*cos(y_phi-y_rot)
            x_est = y_rad*sin(y_phi-y_rot)
            ### Rotate about the x-axis ###
            x_rad = sqrt(y_est**2 + z_est**2)
            x_phi = atan2(z_est, y_est)
            y_est = x_rad*cos(x_phi-x_rot)
            z_est = x_rad*sin(x_phi-x_rot)
            # Invert translating the center of rotation to (0,0,0)
            x_est = x_est + x_ctr
            y_est = y_est + y_ctr
            z_est = z_est + z_ctr
            # interpolate
            x_lo = <int> floor(x_est)
            y_lo = <int> floor(y_est)
            z_lo = <int> floor(z_est)
            x_hi = x_lo + 1 #ceil(x_est)
            y_hi = y_lo + 1 #ceil(y_est)
            z_hi = z_lo + 1 #ceil(z_est)
            # 1D interpolation weights
            Wx_lo = - x_est + x_hi
            Wy_lo = - y_est + y_hi
            Wz_lo = - z_est + z_hi
            Wx_hi = + x_est - x_lo
            Wy_hi = + y_est - y_lo
            Wz_hi = + z_est - z_lo
            # 3D interpolation weights
            W_LLL = Wx_lo * Wy_lo * Wz_lo
            W_LLH = Wx_lo * Wy_lo * Wz_hi
            W_LHL = Wx_lo * Wy_hi * Wz_lo
            W_LHH = Wx_lo * Wy_hi * Wz_hi
            W_HLL = Wx_hi * Wy_lo * Wz_lo
            W_HLH = Wx_hi * Wy_lo * Wz_hi
            W_HHL = Wx_hi * Wy_hi * Wz_lo
            W_HHH = Wx_hi * Wy_hi * Wz_hi
            #
            if ((x_lo >= 0) and (x_hi < nX) and 
                (y_lo >= 0) and (y_hi < nY) and
                (z_lo >= 0) and (z_hi < nZ)):
                arr[x,y,z] = W_LLL * in_arr[x_lo,y_lo,z_lo] + \
                             W_LLH * in_arr[x_lo,y_lo,z_hi] + \
                             W_LHL * in_arr[x_lo,y_hi,z_lo] + \
                             W_LHH * in_arr[x_lo,y_hi,z_hi] + \
                             W_HLL * in_arr[x_hi,y_lo,z_lo] + \
                             W_HLH * in_arr[x_hi,y_lo,z_hi] + \
                             W_HHL * in_arr[x_hi,y_hi,z_lo] + \
                             W_HHH * in_arr[x_hi,y_hi,z_hi]
          # end for z loop
        # end for y loop
      # end for x loop
    return np.asarray(arr)
# ==============================================================

