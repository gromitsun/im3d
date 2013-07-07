#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport sqrt
# ==============================================================
def circle(int n, double R):
    """
    n -> size of the array
    R -> radius of the circle
    """
    # Variables
    cdef int x_i, nX=n
    cdef int y_i, nY=n
    cdef double x_c, y_c
    cdef double dist, ctr=(n-1.0)/2.0
    # Arrays:
    cdef double[:,:] arr = np.zeros((nX,nY), dtype=np.float64)
    #
    with nogil:
      for x_i in range(nX):
        for y_i in range(nY):
          x_c = <double> x_i - ctr
          y_c = <double> y_i - ctr
          dist = sqrt(x_c*x_c + y_c*y_c)
          #
          arr[x_i, y_i] = R - dist
    #
    return np.asarray(arr)

# ==============================================================
def sphere(int n, double R):
    """
    n -> size of the array
    R -> radius of the circle
    """
    cdef int x_i, nX=n
    cdef int y_i, nY=n
    cdef int z_i, nZ=n
    cdef double x_c, y_c, z_c
    cdef double dist, ctr = (n-1.0)/2.0
    #
    cdef double[:,:,:] arr = np.zeros((nX,nY,nZ), dtype=np.float64)
    #
    with nogil:
      for x_i in range(nX):
        for y_i in range(nY):
          for z_i in range(nZ):
            x_c = <double> x_i - ctr
            y_c = <double> y_i - ctr
            z_c = <double> z_i - ctr
            dist = sqrt(x_c*x_c + y_c*y_c + z_c*z_c)
            #
            arr[x_i, y_i, z_i] = R - dist
    #
    return np.asarray(arr)

# ==============================================================
def hyperbaloid(int n=64, double R=20, double a=1.0, double b=1.0, double c=1.0):
    """
    n -> size of the array
    R -> radius - kinda like a general size parameter
    a -> x-axis difference from R
    b -> y-axis difference from R
    c -> z-axis difference from R
    """
    cdef int x_i, nX=n
    cdef int y_i, nY=n
    cdef int z_i, nZ=n
    cdef double x_c, y_c, z_c
    cdef double r, ctr = (n-1.0)/2.0
    #
    cdef double[:,:,:] arr = np.zeros((nX,nY,nZ), dtype=np.float64)
    #
    with nogil:
      for x_i in range(nX):
        for y_i in range(nY):
          for z_i in range(nZ):
            x_c = (<double> x_i - ctr)/a
            y_c = (<double> y_i - ctr)/b
            z_c = (<double> z_i - ctr)/c
            r = sqrt(x_c*x_c + y_c*y_c)
            #
            arr[x_i, y_i, z_i] = R - r + z_c*z_c/sqrt(2*R)
    #
    return np.asarray(arr)







