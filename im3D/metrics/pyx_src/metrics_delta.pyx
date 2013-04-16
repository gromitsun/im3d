#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport sqrt, cos, fabs
from cython.parallel cimport prange
# ==============================================================
def delta(arr, eps=1.5):
    """
    INPUTS
    ======
      arr --> 2D or 3D numpy array, required
              Data array.  This must be a signed distance
              function.
              
      eps --> float, optional (default=1.5)
              Half-width of the delta function.
    
    OUTPUTS
    =======
      delta -> 2D or 3D numpy array
              Delta function array
    
    NOTES
    =====
      None
    """
    # ==========================================================
    if arr.ndim == 2:
        return delta_2D(np.require(arr, dtype=np.float64), eps)
    elif arr.ndim == 3:
        return delta_3D(np.require(arr, dtype=np.float64), eps)

cpdef delta_2D(double[:,:] arr, double eps):
    # ==========================================================
    # Typed values:
    cdef ssize_t  x, nx=arr.shape[0]
    cdef ssize_t  y, ny=arr.shape[1]
    cdef double   phi_x, phi_y, phi_z, grad
    cdef double   pi=3.141592653589793
    # Typed arrays:
    cdef double[:,:] delta = np.zeros_like(arr)
    #
    with nogil:
      for x in prange(1,nx-1):
        for y in range(1,ny-1):
          if fabs(arr[x,y]) < eps:
            phi_x = (arr[x+1,y] - arr[x-1,y])/2.0
            phi_y = (arr[x,y+1] - arr[x,y-1])/2.0
            grad = sqrt(phi_x**2 + phi_y**2)
            #
            delta[x,y] = grad/(2.0*eps) * (1.0 + cos(arr[x,y]*pi/eps)) 
          # end if
        # end y for loop
      # end x for loop
    # end nogil
    return np.asarray(delta)

cpdef delta_3D(double[:,:,:] arr, double eps):
    # ==========================================================
    # Typed values:
    cdef ssize_t  x, nx=arr.shape[0]
    cdef ssize_t  y, ny=arr.shape[1]
    cdef ssize_t  z, nz=arr.shape[2]
    cdef double   phi_x, phi_y, phi_z, grad
    cdef double   pi=3.141592653589793
    # Typed arrays:
    cdef double[:,:,:] delta = np.zeros_like(arr)
    #
    with nogil:
      for x in prange(1,nx-1):
        for y in range(1,ny-1):
          for z in range(1,nz-1):
            if fabs(arr[x,y,z]) < eps:
              phi_x = (arr[x+1,y,z] - arr[x-1,y,z])/2.0
              phi_y = (arr[x,y+1,z] - arr[x,y-1,z])/2.0
              phi_z = (arr[x,y,z+1] - arr[x,y,z-1])/2.0
              grad = sqrt(phi_x**2 + phi_y**2 + phi_z**2)
              #
              delta[x,y,z] = grad/(2.0*eps) * (1.0 + cos(arr[x,y,z]*pi/eps)) 
            # end if
          # end z for loop
        # end y for loop
      # end x for loop
    # end nogil
    return np.asarray(delta)


