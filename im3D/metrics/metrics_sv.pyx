#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
cdef extern from "math.h":
    float sqrtf(float) nogil
    float fabsf(float) nogil
    float cosf(float) nogil
    double sqrt(double) nogil
    double fabs(double) nogil
    double cos(double) nogil
from cython.parallel cimport prange
#=== interface function ========================================================
def sv(arr, eps=2.5):
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
    dtype = arr.dtype
    arr = np.atleast_3d(arr)
    arr = np.require(arr, requirements=['C_contiguous'])
    if dtype == 'float32':
        Sv_val = sv_f32(arr, eps)
    elif dtype == 'float64':
        Sv_val = sv_f64(arr, eps)
    else:
        arr = arr.astype('float32')
        Sv_val = sv_f32(arr, eps)
    
    return Sv_val

#=== 32-bit Sv calculation =====================================================
def sv_f32(float[:,:,::1] arr, float eps):
    #--- Typed values ----------------------------------------------------------
    cdef ssize_t  x, nX=arr.shape[0]
    cdef ssize_t  y, nY=arr.shape[1]
    cdef ssize_t  z, nZ=arr.shape[2]
    cdef float  pi=3.141592653589793
    cdef float  err, n=arr.size
    cdef float  phi_x, phi_y, phi_z, grad
    cdef float  delta
    cdef double  Sv=0.0
    #
    with nogil:
      for x in prange(1,nX-1):
        for y in range(1,nY-1):
          for z in range(1,nZ-1):
            if fabsf(arr[x,y,z]) <= eps:
              phi_x = (arr[x+1,y,z] - arr[x-1,y,z])/2.0
              phi_y = (arr[x,y+1,z] - arr[x,y-1,z])/2.0
              phi_z = (arr[x,y,z+1] - arr[x,y,z-1])/2.0
              grad = sqrtf(phi_x**2 + phi_y**2 + phi_z**2)
              #
              delta = 1./(2.0*eps) * (1.0 + cosf(arr[x,y,z]*pi/eps)) 
              #
              Sv += delta * grad
          # end z for loop
        # end y for loop
      # end x for loop
    # end nogil
    Sv = Sv/n
    #
    return Sv

#=== 64-bit Sv calculation =====================================================
def sv_f64(double[:,:,::1] arr, double eps):
    #--- Typed values ----------------------------------------------------------
    cdef ssize_t  x, nX=arr.shape[0]
    cdef ssize_t  y, nY=arr.shape[1]
    cdef ssize_t  z, nZ=arr.shape[2]
    cdef double  pi=3.141592653589793
    cdef double  err, n=arr.size
    cdef double  phi_x, phi_y, phi_z, grad
    cdef double  delta
    cdef double  Sv=0.0
    #
    with nogil:
      for x in prange(1,nX-1):
        for y in range(1,nY-1):
          for z in range(1,nZ-1):
            if fabs(arr[x,y,z]) <= eps:
              phi_x = (arr[x+1,y,z] - arr[x-1,y,z])/2.0
              phi_y = (arr[x,y+1,z] - arr[x,y-1,z])/2.0
              phi_z = (arr[x,y,z+1] - arr[x,y,z-1])/2.0
              grad = sqrt(phi_x**2 + phi_y**2 + phi_z**2)
              #
              delta = 1./(2.0*eps) * (1.0 + cos(arr[x,y,z]*pi/eps)) 
              #
              Sv += delta * grad
          # end z for loop
        # end y for loop
      # end x for loop
    # end nogil
    Sv = Sv/n
    #
    return Sv
