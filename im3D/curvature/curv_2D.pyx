import numpy as np
from cython.parallel cimport prange
from libc.math cimport sqrt
#=== mean curvature; 32-bit floats =============================================
def H_32(float[:,::1] arr):
    # Variables
    cdef ssize_t i
    cdef ssize_t x, nX=arr.shape[0]
    cdef ssize_t y, nY=arr.shape[1]
    cdef double gX, gY, gXX, gYY, gXY, grad
    # Arrays
    cdef float[:,::1] MC = np.zeros((nX,nY), dtype=np.float32)
    # Calculate curvature
    with nogil:
      for x in prange(2, nX-2):
        for y in range(2, nY-2):
            gX = (-1.0*arr[x-1, y] \
                  +1.0*arr[x+1, y])/2.0
            gY = (-1.0*arr[x, y-1] \
                  +1.0*arr[x, y+1])/2.0
            #
            gXX = (+1.0*arr[x-1, y] \
                   -2.0*arr[x+0, y] \
                   +1.0*arr[x+1, y]) / 1.0
            gYY = (+1.0*arr[x, y-1] \
                   -2.0*arr[x, y+0] \
                   +1.0*arr[x, y+1]) / 1.0
            #
            gXY = ( + 1.0*arr[x-1, y-1] \
                    - 1.0*arr[x-1, y+1] \
                    - 1.0*arr[x+1, y-1] \
                    + 1.0*arr[x+1, y+1] ) / 4.0
            #
            grad = sqrt(gX*gX + gY*gY)
            #
            MC[x,y] = (+ gX*gX*gYY       \
                       + gY*gY*gXX       \
                       - 2.*gX*gY*gXY)   \
                       / (grad**3 + 1e-6)
      # Apply boundary conditions:
      for x in range(nX):
          MC[x,  +0] = MC[x,  +2]
          MC[x,  +1] = MC[x,  +2]
          MC[x,nY-2] = MC[x,nY-3]
          MC[x,nY-1] = MC[x,nY-3]
      for y in range(nY):
          MC[  +0,y] = MC[  +2,y]
          MC[  +1,y] = MC[  +2,y]
          MC[nX-2,y] = MC[nX-3,y]
          MC[nX-1,y] = MC[nX-3,y]
    # Return curvature array
    return np.asarray(MC, dtype=np.float32)

#=== mean curvature; 64-bit floats =============================================
def H_64(double[:,::1] arr):
    # Variables
    cdef ssize_t i
    cdef ssize_t x, nX=arr.shape[0]
    cdef ssize_t y, nY=arr.shape[1]
    cdef double gX, gY, gXX, gYY, gXY, grad
    # Arrays
    cdef double[:,::1] MC = np.zeros((nX,nY), dtype=np.float64)
    # Calculate curvature
    with nogil:
      for x in prange(2, nX-2):
        for y in range(2, nY-2):
            gX = (-1.0*arr[x-1, y] \
                  +1.0*arr[x+1, y])/2.0
            gY = (-1.0*arr[x, y-1] \
                  +1.0*arr[x, y+1])/2.0
            #
            gXX = (+1.0*arr[x-1, y] \
                   -2.0*arr[x+0, y] \
                   +1.0*arr[x+1, y]) / 1.0
            gYY = (+1.0*arr[x, y-1] \
                   -2.0*arr[x, y+0] \
                   +1.0*arr[x, y+1]) / 1.0
            #
            gXY = ( + 1.0*arr[x-1, y-1] \
                    - 1.0*arr[x-1, y+1] \
                    - 1.0*arr[x+1, y-1] \
                    + 1.0*arr[x+1, y+1] ) / 4.0
            #
            grad = sqrt(gX*gX + gY*gY)
            #
            MC[x,y] = (+ gX*gX*gYY       \
                       + gY*gY*gXX       \
                       - 2.*gX*gY*gXY)   \
                       / (grad**3 + 1e-6)
      # Apply boundary conditions:
      for x in range(nX):
          MC[x,  +0] = MC[x,  +2]
          MC[x,  +1] = MC[x,  +2]
          MC[x,nY-2] = MC[x,nY-3]
          MC[x,nY-1] = MC[x,nY-3]
      for y in range(nY):
          MC[  +0,y] = MC[  +2,y]
          MC[  +1,y] = MC[  +2,y]
          MC[nX-2,y] = MC[nX-3,y]
          MC[nX-1,y] = MC[nX-3,y]
    # Return curvature array
    return np.asarray(MC, dtype=np.float64)
