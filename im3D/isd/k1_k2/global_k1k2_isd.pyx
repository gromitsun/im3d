#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

# cython: profile=False
# ==============================================================
import numpy as np
from libc.math cimport fabs, sqrt, floor, ceil, cos
from cython.parallel cimport prange
# ==============================================================
cdef inline ssize_t val2bin(double val, double bin_min, double bin_max, ssize_t nbins) nogil:
    return <ssize_t>floor(0.5+(val-bin_min)/(bin_max-bin_min)*<double>(nbins-1))
# ==============================================================
def calculate_isd(double[:,:,::1] phi, double[:,:,::1] k1, double[:,:,::1] k2,
                  double[:,:,::1] P, int nbins, double bin_min, double bin_max, double eps):
    # Typed values:
    cdef ssize_t  x, y, z, nx, ny, nz
    cdef ssize_t  k1_bin, k2_bin
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
              k1_bin = val2bin(k1[x,y,z], bin_min, bin_max, nbins)
              k2_bin = val2bin(k2[x,y,z], bin_min, bin_max, nbins)
              #
              if (k1_bin >= 0) and (k1_bin < nbins) and \
                 (k2_bin >= 0) and (k2_bin < nbins):
                ISD[k1_bin, k2_bin] += area * P[x,y,z]
          # end z for loop
        # end y for loop
      # end x for loop
    # end nogil
    return np.asarray(ISD)

    
