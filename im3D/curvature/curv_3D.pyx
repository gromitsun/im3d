#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from cython.parallel cimport prange
from libc.math cimport sqrt
# ==============================================================
def H(double[:,:,:] arr):
    # Variables:
    cdef ssize_t i
    cdef ssize_t x, nX=arr.shape[0]
    cdef ssize_t y, nY=arr.shape[1]
    cdef ssize_t z, nZ=arr.shape[2]
    cdef double  gX, gY, gZ
    cdef double  gXX, gYY, gZZ
    cdef double  gXY, gXZ, gYZ
    cdef double  grad
    # Arrays:
    cdef double[:,:,:] MC = np.zeros((nX,nY,nZ), dtype=np.float64)
    # Calculate curvature
    with nogil:
      for x in prange(2, nX-2):
        for y in range(2, nY-2):
          for z in range(2, nZ-2):
              gX = (+1.0*arr[x-2, y, z] \
                    -8.0*arr[x-1, y, z] \
                    +8.0*arr[x+1, y, z] \
                    -1.0*arr[x+2, y, z])/12.0
              gY = (+1.0*arr[x, y-2, z] \
                    -8.0*arr[x, y-1, z] \
                    +8.0*arr[x, y+1, z] \
                    -1.0*arr[x, y+2, z])/12.0
              gZ = (+1.0*arr[x, y, z-2] \
                    -8.0*arr[x, y, z-1] \
                    +8.0*arr[x, y, z+1] \
                    -1.0*arr[x, y, z+2])/12.0
              #
              gXX = (- 1.0*arr[x-2, y, z] \
                     +16.0*arr[x-1, y, z] \
                     -30.0*arr[x+0, y, z] \
                     +16.0*arr[x+1, y, z] \
                     - 1.0*arr[x+2, y, z] ) / 12.0
              gYY = (- 1.0*arr[x, y-2, z] \
                     +16.0*arr[x, y-1, z] \
                     -30.0*arr[x, y+0, z] \
                     +16.0*arr[x, y+1, z] \
                     - 1.0*arr[x, y+2, z] ) / 12.0
              gZZ = (- 1.0*arr[x, y, z-2] \
                     +16.0*arr[x, y, z-1] \
                     -30.0*arr[x, y, z+0] \
                     +16.0*arr[x, y, z+1] \
                     - 1.0*arr[x, y, z+2] ) / 12.0
              #
              gXY = ( + 1.0*arr[x-2, y-2, z] \
                      - 8.0*arr[x-2, y-1, z] \
                      + 8.0*arr[x-2, y+1, z] \
                      - 1.0*arr[x-2, y+2, z] \
                      - 8.0*arr[x-1, y-2, z] \
                      +64.0*arr[x-1, y-1, z] \
                      -64.0*arr[x-1, y+1, z] \
                      + 8.0*arr[x-1, y+2, z] \
                      + 8.0*arr[x+1, y-2, z] \
                      -64.0*arr[x+1, y-1, z] \
                      +64.0*arr[x+1, y+1, z] \
                      - 8.0*arr[x+1, y+2, z] \
                      - 1.0*arr[x+2, y-2, z] \
                      + 8.0*arr[x+2, y-1, z] \
                      - 8.0*arr[x+2, y+1, z] \
                      + 1.0*arr[x+2, y+2, z] ) / 144.0
              gXZ = ( + 1.0*arr[x-2, y, z-2] \
                      - 8.0*arr[x-2, y, z-1] \
                      + 8.0*arr[x-2, y, z+1] \
                      - 1.0*arr[x-2, y, z+2] \
                      - 8.0*arr[x-1, y, z-2] \
                      +64.0*arr[x-1, y, z-1] \
                      -64.0*arr[x-1, y, z+1] \
                      + 8.0*arr[x-1, y, z+2] \
                      + 8.0*arr[x+1, y, z-2] \
                      -64.0*arr[x+1, y, z-1] \
                      +64.0*arr[x+1, y, z+1] \
                      - 8.0*arr[x+1, y, z+2] \
                      - 1.0*arr[x+2, y, z-2] \
                      + 8.0*arr[x+2, y, z-1] \
                      - 8.0*arr[x+2, y, z+1] \
                      + 1.0*arr[x+2, y, z+2] ) / 144.0
              gYZ = ( + 1.0*arr[x, y-2, z-2] \
                      - 8.0*arr[x, y-2, z-1] \
                      + 8.0*arr[x, y-2, z+1] \
                      - 1.0*arr[x, y-2, z+2] \
                      - 8.0*arr[x, y-1, z-2] \
                      +64.0*arr[x, y-1, z-1] \
                      -64.0*arr[x, y-1, z+1] \
                      + 8.0*arr[x, y-1, z+2] \
                      + 8.0*arr[x, y+1, z-2] \
                      -64.0*arr[x, y+1, z-1] \
                      +64.0*arr[x, y+1, z+1] \
                      - 8.0*arr[x, y+1, z+2] \
                      - 1.0*arr[x, y+2, z-2] \
                      + 8.0*arr[x, y+2, z-1] \
                      - 8.0*arr[x, y+2, z+1] \
                      + 1.0*arr[x, y+2, z+2] ) / 144.0
              #
              grad = sqrt(gX*gX + gY*gY + gZ*gZ)
              #
              MC[x,y,z] = ( + gX*gX*gYY + gX*gX*gZZ  \
                            + gY*gY*gXX + gY*gY*gZZ  \
                            + gZ*gZ*gXX + gZ*gZ*gYY  \
                            - 2.*gX*gY*gXY           \
                            - 2.*gX*gZ*gXZ           \
                            - 2.*gY*gZ*gYZ)          \
                            / (2.0 * grad**3 + 1e-6)
      # Apply boundary conditions:
      for y in range(nY):
          for z in range(nZ):
              MC[   0,y,z] = MC[   2,y,z]
              MC[   1,y,z] = MC[   2,y,z]
              MC[nX-2,y,z] = MC[nX-3,y,z]
              MC[nX-1,y,z] = MC[nX-3,y,z]
      for x in range(nX):
          for z in range(nZ):
              MC[x,   0,z] = MC[x,   2,z]
              MC[x,   1,z] = MC[x,   2,z]
              MC[x,nY-2,z] = MC[x,nY-3,z]
              MC[x,nY-1,z] = MC[x,nY-3,z]
      for x in range(nX):
          for y in range(nY):
              MC[x,y,   0] = MC[x,y,   2]
              MC[x,y,   1] = MC[x,y,   2]
              MC[x,y,nZ-2] = MC[x,y,nZ-3]
              MC[x,y,nZ-1] = MC[x,y,nZ-3]
    # Return curvature array
    return np.asarray(MC)

# ==============================================================
def K(double[:,:,:] arr):
    # Variables:
    cdef ssize_t i
    cdef ssize_t x, nX=arr.shape[0]
    cdef ssize_t y, nY=arr.shape[1]
    cdef ssize_t z, nZ=arr.shape[2]
    cdef double  gX, gY, gZ
    cdef double  gXX, gYY, gZZ
    cdef double  gXY, gXZ, gYZ
    cdef double  grad
    # Arrays:
    cdef double[:,:,:] GC = np.zeros((nX,nY,nZ), dtype=np.float64)
    # Calculate curvature
    with nogil:
      for x in prange(2, nX-2):
        for y in range(2, nY-2):
          for z in range(2, nZ-2):
              gX = (+1.0*arr[x-2, y, z] \
                    -8.0*arr[x-1, y, z] \
                    +8.0*arr[x+1, y, z] \
                    -1.0*arr[x+2, y, z])/12.0
              gY = (+1.0*arr[x, y-2, z] \
                    -8.0*arr[x, y-1, z] \
                    +8.0*arr[x, y+1, z] \
                    -1.0*arr[x, y+2, z])/12.0
              gZ = (+1.0*arr[x, y, z-2] \
                    -8.0*arr[x, y, z-1] \
                    +8.0*arr[x, y, z+1] \
                    -1.0*arr[x, y, z+2])/12.0
              #
              gXX = (- 1.0*arr[x-2, y, z] \
                     +16.0*arr[x-1, y, z] \
                     -30.0*arr[x+0, y, z] \
                     +16.0*arr[x+1, y, z] \
                     - 1.0*arr[x+2, y, z] ) / 12.0
              gYY = (- 1.0*arr[x, y-2, z] \
                     +16.0*arr[x, y-1, z] \
                     -30.0*arr[x, y+0, z] \
                     +16.0*arr[x, y+1, z] \
                     - 1.0*arr[x, y+2, z] ) / 12.0
              gZZ = (- 1.0*arr[x, y, z-2] \
                     +16.0*arr[x, y, z-1] \
                     -30.0*arr[x, y, z+0] \
                     +16.0*arr[x, y, z+1] \
                     - 1.0*arr[x, y, z+2] ) / 12.0
              #
              gXY = ( + 1.0*arr[x-2, y-2, z] \
                      - 8.0*arr[x-2, y-1, z] \
                      + 8.0*arr[x-2, y+1, z] \
                      - 1.0*arr[x-2, y+2, z] \
                      - 8.0*arr[x-1, y-2, z] \
                      +64.0*arr[x-1, y-1, z] \
                      -64.0*arr[x-1, y+1, z] \
                      + 8.0*arr[x-1, y+2, z] \
                      + 8.0*arr[x+1, y-2, z] \
                      -64.0*arr[x+1, y-1, z] \
                      +64.0*arr[x+1, y+1, z] \
                      - 8.0*arr[x+1, y+2, z] \
                      - 1.0*arr[x+2, y-2, z] \
                      + 8.0*arr[x+2, y-1, z] \
                      - 8.0*arr[x+2, y+1, z] \
                      + 1.0*arr[x+2, y+2, z] ) / 144.0
              gXZ = ( + 1.0*arr[x-2, y, z-2] \
                      - 8.0*arr[x-2, y, z-1] \
                      + 8.0*arr[x-2, y, z+1] \
                      - 1.0*arr[x-2, y, z+2] \
                      - 8.0*arr[x-1, y, z-2] \
                      +64.0*arr[x-1, y, z-1] \
                      -64.0*arr[x-1, y, z+1] \
                      + 8.0*arr[x-1, y, z+2] \
                      + 8.0*arr[x+1, y, z-2] \
                      -64.0*arr[x+1, y, z-1] \
                      +64.0*arr[x+1, y, z+1] \
                      - 8.0*arr[x+1, y, z+2] \
                      - 1.0*arr[x+2, y, z-2] \
                      + 8.0*arr[x+2, y, z-1] \
                      - 8.0*arr[x+2, y, z+1] \
                      + 1.0*arr[x+2, y, z+2] ) / 144.0
              gYZ = ( + 1.0*arr[x, y-2, z-2] \
                      - 8.0*arr[x, y-2, z-1] \
                      + 8.0*arr[x, y-2, z+1] \
                      - 1.0*arr[x, y-2, z+2] \
                      - 8.0*arr[x, y-1, z-2] \
                      +64.0*arr[x, y-1, z-1] \
                      -64.0*arr[x, y-1, z+1] \
                      + 8.0*arr[x, y-1, z+2] \
                      + 8.0*arr[x, y+1, z-2] \
                      -64.0*arr[x, y+1, z-1] \
                      +64.0*arr[x, y+1, z+1] \
                      - 8.0*arr[x, y+1, z+2] \
                      - 1.0*arr[x, y+2, z-2] \
                      + 8.0*arr[x, y+2, z-1] \
                      - 8.0*arr[x, y+2, z+1] \
                      + 1.0*arr[x, y+2, z+2] ) / 144.0
              #
              grad = sqrt(gX*gX + gY*gY + gZ*gZ)
              #
              GC[x,y,z] = ( + gX**2*(gYY*gZZ - gYZ**2)  \
                            + gY**2*(gXX*gZZ - gXZ**2)  \
                            + gZ**2*(gXX*gYY - gXY**2)  \
                            + 2.*gX*gY*(gXZ*gYZ - gXY*gZZ)  \
                            + 2.*gX*gZ*(gXY*gYZ - gXZ*gYY)  \
                            + 2.*gY*gZ*(gXY*gXZ - gYZ*gXX) )\
                            / (grad**4 + 1e-6)
      # Apply boundary conditions:
      for y in range(nY):
          for z in range(nZ):
              GC[   0,y,z] = GC[   2,y,z]
              GC[   1,y,z] = GC[   2,y,z]
              GC[nX-2,y,z] = GC[nX-3,y,z]
              GC[nX-1,y,z] = GC[nX-3,y,z]
      for x in range(nX):
          for z in range(nZ):
              GC[x,   0,z] = GC[x,   2,z]
              GC[x,   1,z] = GC[x,   2,z]
              GC[x,nY-2,z] = GC[x,nY-3,z]
              GC[x,nY-1,z] = GC[x,nY-3,z]
      for x in range(nX):
          for y in range(nY):
              GC[x,y,   0] = GC[x,y,   2]
              GC[x,y,   1] = GC[x,y,   2]
              GC[x,y,nZ-2] = GC[x,y,nZ-3]
              GC[x,y,nZ-1] = GC[x,y,nZ-3]
    # Return curvature array
    return np.asarray(GC)

# ==============================================================
def K1(double[:,:,:] arr):
    # Variables:
    cdef ssize_t i
    cdef ssize_t x, nX=arr.shape[0]
    cdef ssize_t y, nY=arr.shape[1]
    cdef ssize_t z, nZ=arr.shape[2]
    cdef double  gX, gY, gZ
    cdef double  gXX, gYY, gZZ
    cdef double  gXY, gXZ, gYZ
    cdef double  grad, H, K
    # Arrays:
    cdef double[:,:,:] K1 = np.zeros((nX,nY,nZ), dtype=np.float64)
    # Calculate curvature
    with nogil:
      for x in prange(2, nX-2):
        for y in range(2, nY-2):
          for z in range(2, nZ-2):
              gX = (+1.0*arr[x-2, y, z] \
                    -8.0*arr[x-1, y, z] \
                    +8.0*arr[x+1, y, z] \
                    -1.0*arr[x+2, y, z])/12.0
              gY = (+1.0*arr[x, y-2, z] \
                    -8.0*arr[x, y-1, z] \
                    +8.0*arr[x, y+1, z] \
                    -1.0*arr[x, y+2, z])/12.0
              gZ = (+1.0*arr[x, y, z-2] \
                    -8.0*arr[x, y, z-1] \
                    +8.0*arr[x, y, z+1] \
                    -1.0*arr[x, y, z+2])/12.0
              #
              gXX = (- 1.0*arr[x-2, y, z] \
                     +16.0*arr[x-1, y, z] \
                     -30.0*arr[x+0, y, z] \
                     +16.0*arr[x+1, y, z] \
                     - 1.0*arr[x+2, y, z] ) / 12.0
              gYY = (- 1.0*arr[x, y-2, z] \
                     +16.0*arr[x, y-1, z] \
                     -30.0*arr[x, y+0, z] \
                     +16.0*arr[x, y+1, z] \
                     - 1.0*arr[x, y+2, z] ) / 12.0
              gZZ = (- 1.0*arr[x, y, z-2] \
                     +16.0*arr[x, y, z-1] \
                     -30.0*arr[x, y, z+0] \
                     +16.0*arr[x, y, z+1] \
                     - 1.0*arr[x, y, z+2] ) / 12.0
              #
              gXY = ( + 1.0*arr[x-2, y-2, z] \
                      - 8.0*arr[x-2, y-1, z] \
                      + 8.0*arr[x-2, y+1, z] \
                      - 1.0*arr[x-2, y+2, z] \
                      - 8.0*arr[x-1, y-2, z] \
                      +64.0*arr[x-1, y-1, z] \
                      -64.0*arr[x-1, y+1, z] \
                      + 8.0*arr[x-1, y+2, z] \
                      + 8.0*arr[x+1, y-2, z] \
                      -64.0*arr[x+1, y-1, z] \
                      +64.0*arr[x+1, y+1, z] \
                      - 8.0*arr[x+1, y+2, z] \
                      - 1.0*arr[x+2, y-2, z] \
                      + 8.0*arr[x+2, y-1, z] \
                      - 8.0*arr[x+2, y+1, z] \
                      + 1.0*arr[x+2, y+2, z] ) / 144.0
              gXZ = ( + 1.0*arr[x-2, y, z-2] \
                      - 8.0*arr[x-2, y, z-1] \
                      + 8.0*arr[x-2, y, z+1] \
                      - 1.0*arr[x-2, y, z+2] \
                      - 8.0*arr[x-1, y, z-2] \
                      +64.0*arr[x-1, y, z-1] \
                      -64.0*arr[x-1, y, z+1] \
                      + 8.0*arr[x-1, y, z+2] \
                      + 8.0*arr[x+1, y, z-2] \
                      -64.0*arr[x+1, y, z-1] \
                      +64.0*arr[x+1, y, z+1] \
                      - 8.0*arr[x+1, y, z+2] \
                      - 1.0*arr[x+2, y, z-2] \
                      + 8.0*arr[x+2, y, z-1] \
                      - 8.0*arr[x+2, y, z+1] \
                      + 1.0*arr[x+2, y, z+2] ) / 144.0
              gYZ = ( + 1.0*arr[x, y-2, z-2] \
                      - 8.0*arr[x, y-2, z-1] \
                      + 8.0*arr[x, y-2, z+1] \
                      - 1.0*arr[x, y-2, z+2] \
                      - 8.0*arr[x, y-1, z-2] \
                      +64.0*arr[x, y-1, z-1] \
                      -64.0*arr[x, y-1, z+1] \
                      + 8.0*arr[x, y-1, z+2] \
                      + 8.0*arr[x, y+1, z-2] \
                      -64.0*arr[x, y+1, z-1] \
                      +64.0*arr[x, y+1, z+1] \
                      - 8.0*arr[x, y+1, z+2] \
                      - 1.0*arr[x, y+2, z-2] \
                      + 8.0*arr[x, y+2, z-1] \
                      - 8.0*arr[x, y+2, z+1] \
                      + 1.0*arr[x, y+2, z+2] ) / 144.0
              #
              grad = sqrt(gX*gX + gY*gY + gZ*gZ)
              #
              H = ( + gX*gX*gYY + gX*gX*gZZ  \
                    + gY*gY*gXX + gY*gY*gZZ  \
                    + gZ*gZ*gXX + gZ*gZ*gYY  \
                    - 2.*gX*gY*gXY           \
                    - 2.*gX*gZ*gXZ           \
                    - 2.*gY*gZ*gYZ)          \
                    / (2.0 * grad**3 + 1e-6)
              #
              K = ( + gX**2*(gYY*gZZ - gYZ**2)  \
                    + gY**2*(gXX*gZZ - gXZ**2)  \
                    + gZ**2*(gXX*gYY - gXY**2)  \
                    + 2.*gX*gY*(gXZ*gYZ - gXY*gZZ)  \
                    + 2.*gX*gZ*(gXY*gYZ - gXZ*gYY)  \
                    + 2.*gY*gZ*(gXY*gXZ - gYZ*gXX) )\
                    / (grad**4 + 1e-6)
              #
              if H*H > K:
                  K1[x,y,z] = H - sqrt(H*H - K)
              else:
                  K1[x,y,z] = H
      # Apply boundary conditions:
      for y in range(nY):
          for z in range(nZ):
              K1[   0,y,z] = K1[   2,y,z]
              K1[   1,y,z] = K1[   2,y,z]
              K1[nX-2,y,z] = K1[nX-3,y,z]
              K1[nX-1,y,z] = K1[nX-3,y,z]
      for x in range(nX):
          for z in range(nZ):
              K1[x,   0,z] = K1[x,   2,z]
              K1[x,   1,z] = K1[x,   2,z]
              K1[x,nY-2,z] = K1[x,nY-3,z]
              K1[x,nY-1,z] = K1[x,nY-3,z]
      for x in range(nX):
          for y in range(nY):
              K1[x,y,   0] = K1[x,y,   2]
              K1[x,y,   1] = K1[x,y,   2]
              K1[x,y,nZ-2] = K1[x,y,nZ-3]
              K1[x,y,nZ-1] = K1[x,y,nZ-3]
    # Return curvature array
    return np.asarray(K1)

# ==============================================================
def K2(double[:,:,:] arr):
    # Variables:
    cdef ssize_t i
    cdef ssize_t x, nX=arr.shape[0]
    cdef ssize_t y, nY=arr.shape[1]
    cdef ssize_t z, nZ=arr.shape[2]
    cdef double  gX, gY, gZ
    cdef double  gXX, gYY, gZZ
    cdef double  gXY, gXZ, gYZ
    cdef double  grad, H, K
    # Arrays:
    cdef double[:,:,:] K2 = np.zeros((nX,nY,nZ), dtype=np.float64)
    # Calculate curvature
    with nogil:
      for x in prange(2, nX-2):
        for y in range(2, nY-2):
          for z in range(2, nZ-2):
              gX = (+1.0*arr[x-2, y, z] \
                    -8.0*arr[x-1, y, z] \
                    +8.0*arr[x+1, y, z] \
                    -1.0*arr[x+2, y, z])/12.0
              gY = (+1.0*arr[x, y-2, z] \
                    -8.0*arr[x, y-1, z] \
                    +8.0*arr[x, y+1, z] \
                    -1.0*arr[x, y+2, z])/12.0
              gZ = (+1.0*arr[x, y, z-2] \
                    -8.0*arr[x, y, z-1] \
                    +8.0*arr[x, y, z+1] \
                    -1.0*arr[x, y, z+2])/12.0
              #
              gXX = (- 1.0*arr[x-2, y, z] \
                     +16.0*arr[x-1, y, z] \
                     -30.0*arr[x+0, y, z] \
                     +16.0*arr[x+1, y, z] \
                     - 1.0*arr[x+2, y, z] ) / 12.0
              gYY = (- 1.0*arr[x, y-2, z] \
                     +16.0*arr[x, y-1, z] \
                     -30.0*arr[x, y+0, z] \
                     +16.0*arr[x, y+1, z] \
                     - 1.0*arr[x, y+2, z] ) / 12.0
              gZZ = (- 1.0*arr[x, y, z-2] \
                     +16.0*arr[x, y, z-1] \
                     -30.0*arr[x, y, z+0] \
                     +16.0*arr[x, y, z+1] \
                     - 1.0*arr[x, y, z+2] ) / 12.0
              #
              gXY = ( + 1.0*arr[x-2, y-2, z] \
                      - 8.0*arr[x-2, y-1, z] \
                      + 8.0*arr[x-2, y+1, z] \
                      - 1.0*arr[x-2, y+2, z] \
                      - 8.0*arr[x-1, y-2, z] \
                      +64.0*arr[x-1, y-1, z] \
                      -64.0*arr[x-1, y+1, z] \
                      + 8.0*arr[x-1, y+2, z] \
                      + 8.0*arr[x+1, y-2, z] \
                      -64.0*arr[x+1, y-1, z] \
                      +64.0*arr[x+1, y+1, z] \
                      - 8.0*arr[x+1, y+2, z] \
                      - 1.0*arr[x+2, y-2, z] \
                      + 8.0*arr[x+2, y-1, z] \
                      - 8.0*arr[x+2, y+1, z] \
                      + 1.0*arr[x+2, y+2, z] ) / 144.0
              gXZ = ( + 1.0*arr[x-2, y, z-2] \
                      - 8.0*arr[x-2, y, z-1] \
                      + 8.0*arr[x-2, y, z+1] \
                      - 1.0*arr[x-2, y, z+2] \
                      - 8.0*arr[x-1, y, z-2] \
                      +64.0*arr[x-1, y, z-1] \
                      -64.0*arr[x-1, y, z+1] \
                      + 8.0*arr[x-1, y, z+2] \
                      + 8.0*arr[x+1, y, z-2] \
                      -64.0*arr[x+1, y, z-1] \
                      +64.0*arr[x+1, y, z+1] \
                      - 8.0*arr[x+1, y, z+2] \
                      - 1.0*arr[x+2, y, z-2] \
                      + 8.0*arr[x+2, y, z-1] \
                      - 8.0*arr[x+2, y, z+1] \
                      + 1.0*arr[x+2, y, z+2] ) / 144.0
              gYZ = ( + 1.0*arr[x, y-2, z-2] \
                      - 8.0*arr[x, y-2, z-1] \
                      + 8.0*arr[x, y-2, z+1] \
                      - 1.0*arr[x, y-2, z+2] \
                      - 8.0*arr[x, y-1, z-2] \
                      +64.0*arr[x, y-1, z-1] \
                      -64.0*arr[x, y-1, z+1] \
                      + 8.0*arr[x, y-1, z+2] \
                      + 8.0*arr[x, y+1, z-2] \
                      -64.0*arr[x, y+1, z-1] \
                      +64.0*arr[x, y+1, z+1] \
                      - 8.0*arr[x, y+1, z+2] \
                      - 1.0*arr[x, y+2, z-2] \
                      + 8.0*arr[x, y+2, z-1] \
                      - 8.0*arr[x, y+2, z+1] \
                      + 1.0*arr[x, y+2, z+2] ) / 144.0
              #
              grad = sqrt(gX*gX + gY*gY + gZ*gZ)
              #
              H = ( + gX*gX*gYY + gX*gX*gZZ  \
                    + gY*gY*gXX + gY*gY*gZZ  \
                    + gZ*gZ*gXX + gZ*gZ*gYY  \
                    - 2.*gX*gY*gXY           \
                    - 2.*gX*gZ*gXZ           \
                    - 2.*gY*gZ*gYZ)          \
                    / (2.0 * grad**3 + 1e-6)
              #
              K = ( + gX**2*(gYY*gZZ - gYZ**2)  \
                    + gY**2*(gXX*gZZ - gXZ**2)  \
                    + gZ**2*(gXX*gYY - gXY**2)  \
                    + 2.*gX*gY*(gXZ*gYZ - gXY*gZZ)  \
                    + 2.*gX*gZ*(gXY*gYZ - gXZ*gYY)  \
                    + 2.*gY*gZ*(gXY*gXZ - gYZ*gXX) )\
                    / (grad**4 + 1e-6)
              #
              if H*H > K:
                  K2[x,y,z] = H + sqrt(H*H - K)
              else:
                  K2[x,y,z] = H
      # Apply boundary conditions:
      for y in range(nY):
          for z in range(nZ):
              K2[   0,y,z] = K2[   2,y,z]
              K2[   1,y,z] = K2[   2,y,z]
              K2[nX-2,y,z] = K2[nX-3,y,z]
              K2[nX-1,y,z] = K2[nX-3,y,z]
      for x in range(nX):
          for z in range(nZ):
              K2[x,   0,z] = K2[x,   2,z]
              K2[x,   1,z] = K2[x,   2,z]
              K2[x,nY-2,z] = K2[x,nY-3,z]
              K2[x,nY-1,z] = K2[x,nY-3,z]
      for x in range(nX):
          for y in range(nY):
              K2[x,y,   0] = K2[x,y,   2]
              K2[x,y,   1] = K2[x,y,   2]
              K2[x,y,nZ-2] = K2[x,y,nZ-3]
              K2[x,y,nZ-1] = K2[x,y,nZ-3]
    # Return curvature array
    return np.asarray(K2)

# ==============================================================
