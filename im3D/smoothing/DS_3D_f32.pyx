import numpy as np
from cython.parallel cimport prange

def isotropic(float[:,:,::1] arr, int it, float dt, float Dx, float Dy, float Dz):
    # Variables:
    cdef int  i
    cdef ssize_t  x, nx=arr.shape[0]
    cdef ssize_t  y, ny=arr.shape[1]
    cdef ssize_t  z, nz=arr.shape[2]
    cdef float  gXX, gYY, gZZ
    # Arrays:
    cdef float[:,:,::1] out = np.zeros((nx,ny,nz), dtype='float32')
    cdef float[:,:,::1] chg = np.zeros((nx,ny,nz), dtype='float32')
    out[:,:,:] = arr[:,:,:]
    # ==========================================================
    with nogil:
      for i in range(it):
        #--- compute central voxels ---
        for x in prange(1, nx-1):
          for y in range(1, ny-1):
            for z in range(1, nz-1):
              gXX = + 1 * out[x-1, y, z] - 2 * out[x, y, z] + 1 * out[x+1, y, z]
              gYY = + 1 * out[x, y-1, z] - 2 * out[x, y, z] + 1 * out[x, y+1, z]
              gZZ = + 1 * out[x, y, z-1] - 2 * out[x, y, z] + 1 * out[x, y, z+1]
              chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
            # end z for loop
          # end y for loop
        # end x for loop
        #--- compute face voxels ---
        x = 0
        for y in prange(1, ny-1):
          for z in range(1, nz-1):
            gXX = + 2 * out[x+1, y, z] - 2 * out[x, y, z]
            gYY = + 1 * out[x, y-1, z] - 2 * out[x, y, z] + 1 * out[x, y+1, z]
            gZZ = + 1 * out[x, y, z-1] - 2 * out[x, y, z] + 1 * out[x, y, z+1]
            chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          # end y for loop
        # end z for loop
        #
        x = nx - 1
        for y in prange(1, ny-1):
          for z in range(1, nz-1):
            gXX = + 2 * out[x-1, y, z] - 2 * out[x, y, z]
            gYY = + 1 * out[x, y-1, z] - 2 * out[x, y, z] + 1 * out[x, y+1, z]
            gZZ = + 1 * out[x, y, z-1] - 2 * out[x, y, z] + 1 * out[x, y, z+1]
            chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          # end z for loop
        # end y for loop
        #
        y = 0
        for x in prange(1, nx-1):
          for z in range(1, nz-1):
            gXX = + 1 * out[x-1, y, z] - 2 * out[x, y, z] + 1 * out[x+1, y, z]
            gYY = + 2 * out[x, y+1, z] - 2 * out[x, y, z]
            gZZ = + 1 * out[x, y, z-1] - 2 * out[x, y, z] + 1 * out[x, y, z+1]
            chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          # end z for loop
        # end x for loop
        #
        y = ny - 1
        for x in prange(1, nx-1):
          for z in range(1, nz-1):
            gXX = + 1 * out[x-1, y, z] - 2 * out[x, y, z] + 1 * out[x+1, y, z]
            gYY = + 2 * out[x, y-1, z] - 2 * out[x, y, z]
            gZZ = + 1 * out[x, y, z-1] - 2 * out[x, y, z] + 1 * out[x, y, z+1]
            chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          # end z for loop
        # end x for loop
        #
        z = 0
        for x in prange(1, nx-1):
          for y in range(1, ny-1):
            gXX = + 1 * out[x-1, y, z] - 2 * out[x, y, z] + 1 * out[x+1, y, z]
            gYY = + 1 * out[x, y-1, z] - 2 * out[x, y, z] + 1 * out[x, y+1, z]
            gZZ = + 2 * out[x, y, z+1] - 2 * out[x, y, z]
            chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          # end y for loop
        # end x for loop
        #
        z = nz - 1
        for x in prange(1, nx-1):
          for y in range(1, ny-1):
            gXX = + 1 * out[x-1, y, z] - 2 * out[x, y, z] + 1 * out[x+1, y, z]
            gYY = + 1 * out[x, y-1, z] - 2 * out[x, y, z] + 1 * out[x, y+1, z]
            gZZ = + 2 * out[x, y, z-1] - 2 * out[x, y, z]
            chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          # end y for loop
        # end x for loop
        #--- compute edge voxels ---
        for x in prange(1, nx-1):
          y = 0
          z = 0
          gXX = + 1 * out[x-1, y, z] - 2 * out[x, y, z] + 1 * out[x+1, y, z]
          gYY = + 2 * out[x, y+1, z] - 2 * out[x, y, z]
          gZZ = + 2 * out[x, y, z+1] - 2 * out[x, y, z]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          #
          y = ny - 1
          z = 0
          gXX = + 1 * out[x-1, y, z] - 2 * out[x, y, z] + 1 * out[x+1, y, z]
          gYY = + 2 * out[x, y-1, z] - 2 * out[x, y, z]
          gZZ = + 2 * out[x, y, z+1] - 2 * out[x, y, z]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          #
          y = 0
          z = nz - 1
          gXX = + 1 * out[x-1, y, z] - 2 * out[x, y, z] + 1 * out[x+1, y, z]
          gYY = + 2 * out[x, y+1, z] - 2 * out[x, y, z]
          gZZ = + 2 * out[x, y, z-1] - 2 * out[x, y, z]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          #
          y = ny - 1
          z = nz - 1
          gXX = + 1 * out[x-1, y, z] - 2 * out[x, y, z] + 1 * out[x+1, y, z]
          gYY = + 2 * out[x, y-1, z] - 2 * out[x, y, z]
          gZZ = + 2 * out[x, y, z-1] - 2 * out[x, y, z]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        # end x for loop
        for y in prange(1, ny-1):
          x = 0
          z = 0
          gXX = + 2 * out[x+1, y, z] - 2 * out[x, y, z]
          gYY = + 1 * out[x, y-1, z] - 2 * out[x, y, z] + 1 * out[x, y+1, z]
          gZZ = + 2 * out[x, y, z+1] - 2 * out[x, y, z]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          #
          x = nx - 1
          z = 0
          gXX = + 2 * out[x-1, y, z] - 2 * out[x, y, z]
          gYY = + 1 * out[x, y-1, z] - 2 * out[x, y, z] + 1 * out[x, y+1, z]
          gZZ = + 2 * out[x, y, z+1] - 2 * out[x, y, z]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          #
          x = 0
          z = nz - 1
          gXX = + 2 * out[x+1, y, z] - 2 * out[x, y, z]
          gYY = + 1 * out[x, y-1, z] - 2 * out[x, y, z] + 1 * out[x, y+1, z]
          gZZ = + 2 * out[x, y, z-1] - 2 * out[x, y, z]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          #
          x = nx - 1
          z = nz - 1
          gXX = + 2 * out[x-1, y, z] - 2 * out[x, y, z]
          gYY = + 1 * out[x, y-1, z] - 2 * out[x, y, z] + 1 * out[x, y+1, z]
          gZZ = + 2 * out[x, y, z-1] - 2 * out[x, y, z]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        # end y for loop
        for z in prange(1, nz-1):
          x = 0
          y = 0
          gXX = + 2 * out[x+1, y, z] - 2 * out[x, y, z]
          gYY = + 2 * out[x, y+1, z] - 2 * out[x, y, z]
          gZZ = + 1 * out[x, y, z-1] - 2 * out[x, y, z] + 1 * out[x, y, z+1]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          # 
          x = nx - 1
          y = 0
          gXX = + 2 * out[x-1, y, z] - 2 * out[x, y, z]
          gYY = + 2 * out[x, y+1, z] - 2 * out[x, y, z]
          gZZ = + 1 * out[x, y, z-1] - 2 * out[x, y, z] + 1 * out[x, y, z+1]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          # 
          x = 0
          y = ny - 1
          gXX = + 2 * out[x+1, y, z] - 2 * out[x, y, z]
          gYY = + 2 * out[x, y-1, z] - 2 * out[x, y, z]
          gZZ = + 1 * out[x, y, z-1] - 2 * out[x, y, z] + 1 * out[x, y, z+1]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
          # 
          x = nx - 1
          y = ny - 1
          gXX = + 2 * out[x-1, y, z] - 2 * out[x, y, z]
          gYY = + 2 * out[x, y-1, z] - 2 * out[x, y, z]
          gZZ = + 1 * out[x, y, z-1] - 2 * out[x, y, z] + 1 * out[x, y, z+1]
          chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        # end z for loop
        #--- compute corner voxels ---
        x = 0
        y = 0
        z = 0
        gXX = + 2 * out[x+1, y, z] - 2 * out[x, y, z]
        gYY = + 2 * out[x, y+1, z] - 2 * out[x, y, z]
        gZZ = + 2 * out[x, y, z+1] - 2 * out[x, y, z]
        chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        #
        x = 0
        y = 0
        z = nz - 1
        gXX = + 2 * out[x+1, y, z] - 2 * out[x, y, z]
        gYY = + 2 * out[x, y+1, z] - 2 * out[x, y, z]
        gZZ = + 2 * out[x, y, z-1] - 2 * out[x, y, z]
        chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        #
        x = 0
        y = ny - 1
        z = 0
        gXX = + 2 * out[x+1, y, z] - 2 * out[x, y, z]
        gYY = + 2 * out[x, y-1, z] - 2 * out[x, y, z]
        gZZ = + 2 * out[x, y, z+1] - 2 * out[x, y, z]
        chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        #
        x = 0
        y = ny - 1
        z = nz - 1
        gXX = + 2 * out[x+1, y, z] - 2 * out[x, y, z]
        gYY = + 2 * out[x, y-1, z] - 2 * out[x, y, z]
        gZZ = + 2 * out[x, y, z-1] - 2 * out[x, y, z]
        chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        #
        x = nx - 1
        y = 0
        z = 0
        gXX = + 2 * out[x-1, y, z] - 2 * out[x, y, z]
        gYY = + 2 * out[x, y+1, z] - 2 * out[x, y, z]
        gZZ = + 2 * out[x, y, z+1] - 2 * out[x, y, z]
        chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        #
        x = nx - 1
        y = 0
        z = nz - 1
        gXX = + 2 * out[x-1, y, z] - 2 * out[x, y, z]
        gYY = + 2 * out[x, y+1, z] - 2 * out[x, y, z]
        gZZ = + 2 * out[x, y, z-1] - 2 * out[x, y, z]
        chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        #
        x = nx - 1
        y = ny - 1
        z = 0
        gXX = + 2 * out[x-1, y, z] - 2 * out[x, y, z]
        gYY = + 2 * out[x, y-1, z] - 2 * out[x, y, z]
        gZZ = + 2 * out[x, y, z+1] - 2 * out[x, y, z]
        chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        #
        x = nx - 1
        y = ny - 1
        z = nz - 1
        gXX = + 2 * out[x-1, y, z] - 2 * out[x, y, z]
        gYY = + 2 * out[x, y-1, z] - 2 * out[x, y, z]
        gZZ = + 2 * out[x, y, z-1] - 2 * out[x, y, z]
        chg[x,y,z] = Dx * gXX + Dy * gYY + Dz * gZZ
        #--- Update out ---
        for x in prange(nx):
          for y in range(ny):
            for z in range(nz):
              out[x,y,z] = out[x,y,z] + dt * chg[x,y,z]
            # end z for loop
          # end y for loop
        # end x for loop
      # end iteration for loop
    # end nogil
    return np.asarray(out, dtype='float32')
