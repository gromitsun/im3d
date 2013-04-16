#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport fabs, sqrt, cos
from cython.parallel cimport prange
# ==============================================================
def sv(arr, double eps=2.5):
    """
    INPUTS
    ======
      phi --> 2D or 3D numpy array, required
              Data array.  It must be a signed distance
              function.
              
      eps --> float, optional (default=2.5)
              Half-width of the delta function (DD).
    
    OUTPUTS
    =======
      Sv ---> float
              Surface area per unit volume
    
    NOTES
    =====
      None
    """
    # ==========================================================
    # Typed arrays:
    cdef double[:,:,:] cy_arr = np.atleast_3d(arr)
    # Typed values:
    cdef ssize_t  x, nX=cy_arr.shape[0]
    cdef ssize_t  y, nY=cy_arr.shape[1]
    cdef ssize_t  z, nZ=cy_arr.shape[2]
    cdef double   err, n=cy_arr.size
    cdef double   phi_x, phi_y, phi_z, grad
    cdef double   delta, Sv=0.0
    cdef double   pi=3.141592653589793
    #
    with nogil:
      for x in prange(1,nX-1):
        for y in range(1,nY-1):
          for z in range(1,nZ-1):
            if fabs(cy_arr[x,y,z]) <= eps:
              phi_x = (cy_arr[x+1,y,z] - cy_arr[x-1,y,z])/2.0
              phi_y = (cy_arr[x,y+1,z] - cy_arr[x,y-1,z])/2.0
              phi_z = (cy_arr[x,y,z+1] - cy_arr[x,y,z-1])/2.0
              grad = sqrt(phi_x**2 + phi_y**2 + phi_z**2)
              #
              delta = 1./(2.0*eps) * (1.0 + cos(cy_arr[x,y,z]*pi/eps)) 
              #
              Sv += delta * grad
          # end z for loop
        # end y for loop
      # end x for loop
    # end nogil
    Sv = Sv/n
    #
    return Sv
