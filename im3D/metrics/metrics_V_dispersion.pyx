#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport fabs, sqrt, cos, floor
from cython.parallel cimport prange

cdef inline int round(double val) nogil:
    return <int> floor(val + 0.5)

# ==============================================================
def curv_V_hist(k1, k2, A, V, k_bin_lim=(-2.5, +2.5), double k_nbins=201, 
                V_bin_lim=(-2.5, +2.5), double V_nbins=101):
    """
    INPUTS
    ======
      k1 ---> 3D numpy array, required
              One principal curvature
      
      k2 ---> 3D numpy array, required
              The other principal curvature
              
      A ----> 3D numpy array, required
              Interfacial area of each voxel
      
      V ----> 3D numpy array, required
              Interfacial velocity of each voxel
      
      k_bin_lim -> tuple, length=2, optional (default=(-2.5, +2.5))
              Bin limits for curvatures
      
      k_nbins -> integer, optional (default=201)
              Number of bins for curvatures
    
      V_bin_lim -> tuple, length=2, optional (default=(-2.5, +2.5))
              Bin limits for velocity
      
      V_nbins -> integer, optional (default=101)
              Number of bins for velocity
    
    OUTPUTS
    =======
      hist -> 3D numpy array
              Histogram of area per (k1, k2, V) bin
              axis order: k1, k2, V
    
    NOTES
    =====
      None
    """
    # ==========================================================
    # Typed arrays:
    cdef double[:,:,::1] cy_k1 = np.require(k1, dtype=np.float64, requirements=('C_CONTIGUOUS', 'ALIGNED'))
    cdef double[:,:,::1] cy_k2 = np.require(k2, dtype=np.float64, requirements=('C_CONTIGUOUS', 'ALIGNED'))
    cdef double[:,:,::1] cy_A  = np.require(A,  dtype=np.float64, requirements=('C_CONTIGUOUS', 'ALIGNED'))
    cdef double[:,:,::1] cy_V  = np.require(V,  dtype=np.float64, requirements=('C_CONTIGUOUS', 'ALIGNED'))
    cdef double[:,:,::1] hist  = np.zeros((k_nbins, k_nbins, V_nbins), dtype=np.float64)
    # Typed values:
    cdef ssize_t  x, nX=cy_k1.shape[0]
    cdef ssize_t  y, nY=cy_k1.shape[1]
    cdef ssize_t  z, nZ=cy_k1.shape[2]
    cdef ssize_t  k1_bin, k2_bin, V_bin
    cdef double   k_min=k_bin_lim[0], k_max=k_bin_lim[1]
    cdef double   V_min=V_bin_lim[0], V_max=V_bin_lim[1]
    #
    with nogil:
      for x in prange(nX):
        for y in range(nY):
          for z in range(nZ):
            if cy_A[x,y,z] > 0:
              k1_bin = round((cy_k1[x,y,z] - k_min) / (k_max - k_min) * k_nbins)
              k2_bin = round((cy_k2[x,y,z] - k_min) / (k_max - k_min) * k_nbins)
              V_bin  = round((cy_V[x,y,z]  - V_min) / (V_max - V_min) * V_nbins)
              # 
              if (k1_bin >= 0) and (k1_bin < k_nbins) and \
                 (k2_bin >= 0) and (k2_bin < k_nbins) and \
                 (V_bin  >= 0) and (V_bin  < V_nbins):
                hist[k1_bin, k2_bin, V_bin] = hist[k1_bin, k2_bin, V_bin] + cy_A[x, y, z]
    return np.asarray(hist)
