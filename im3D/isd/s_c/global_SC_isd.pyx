#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

# cython: profile=False
# ==============================================================
import numpy as np
from libc.math cimport fabs, sqrt, floor, ceil, cos, atan2
from cython.parallel cimport prange
# ==============================================================
cdef inline ssize_t val2bin(double val, double bin_min, double bin_max, ssize_t nbins) nogil:
    return <ssize_t>floor(0.5+(val-bin_min)/(bin_max-bin_min)*<double>(nbins-1))
# ==============================================================
def calculate_isd(double[:,:,::1] phi, double[:,:,::1] k1, double[:,:,::1] k2,
                  double[:,:,::1] P, int nbins, double C_max, double eps):
    # Typed values:
    cdef ssize_t  x, y, z, nx, ny, nz
    cdef ssize_t  S_bin, C_bin
    cdef double  C, C_min=+0.0
    cdef double  S, S_min=-1.0, S_max=+1.0
    cdef double  phi_x, phi_y, phi_z, grad
    cdef double  delta, area, pi=3.141592653589793
    # Typed arrays:
    cdef double[:,:] ISD=np.zeros((nbins,nbins), dtype=np.float64)
    #
    nx, ny, nz = phi.shape[0], phi.shape[1], phi.shape[2]
    # 
    with nogil:
      for x in range(1,nx-1):
        for y in range(1,ny-1):
          for z in range(1,nz-1):
            if fabs(phi[x,y,z]) <= eps:
              phi_x = (phi[x+1,y,z] - phi[x-1,y,z])/2.0
              phi_y = (phi[x,y+1,z] - phi[x,y-1,z])/2.0
              phi_z = (phi[x,y,z+1] - phi[x,y,z-1])/2.0
              #
              grad = sqrt(phi_x**2 + phi_y**2 + phi_z**2)
              #
              delta = 1./(2.0*eps) * (1.0 + cos(phi[x,y,z]*pi/eps)) 
              area = delta * grad
              #
              C = sqrt(k1[x,y,z]**2 + k2[x,y,z]**2)
              S = 0.5 + 2/pi * atan2(k1[x,y,z], k2[x,y,z])
              #
              S_bin = val2bin(S, S_min, S_max, nbins)
              C_bin = val2bin(C, C_min, C_max, nbins)
              #
              if (S_bin >= 0) and (S_bin < nbins) and \
                 (C_bin >= 0) and (C_bin < nbins):
                ISD[S_bin, C_bin] += area * P[x,y,z]
          # end z for loop
        # end y for loop
      # end x for loop
    # end nogil
    return np.asarray(ISD)

    
