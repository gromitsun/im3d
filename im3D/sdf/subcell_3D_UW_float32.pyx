import numpy as np
from cython.parallel cimport prange
from math_functions cimport sqrtf, fabsf, f_max, f_min, f_sign
#=== 32-bit floats, Godunov's upwind method reinitialization ===================
def reinit(float[:,:,::1] phi_0, float dt, int max_it, float band, int verbose):
    #--- Define variables ------------------------------------------------------
    cdef float[:,:,::1] phi, phi_t
    cdef ssize_t  x, nx=phi_0.shape[0]
    cdef ssize_t  y, ny=phi_0.shape[1]
    cdef ssize_t  z, nz=phi_0.shape[2]
    cdef ssize_t  xmin=1, xmax=nx-2  # Bounds of the 'good' data
    cdef ssize_t  ymin=1, ymax=ny-2  # Bounds of the 'good' data
    cdef ssize_t  zmin=1, zmax=nz-2  # Bounds of the 'good' data
    cdef float  gX, gXm, gXp, gY, gYm, gYp, gZ, gZm, gZp, G
    cdef float  sgn, dist
    cdef float  max_phi, max_err
    cdef int  i, iter=0, num=1
    #--- Initialize arrays -----------------------------------------------------
    phi   = np.zeros(shape=(nx,ny,nz), dtype=np.float32)
    phi_t = np.zeros(shape=(nx,ny,nz), dtype=np.float32)
    #
    phi[...] = phi_0[...]
    #--- Do the reinitialization -----------------------------------------------
    iter = 0
    while iter < max_it:
        with nogil:
            iter += 1
            for x in prange(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    for z in range(zmin, zmax+1):
                        phi_t[x,y,z] = 0.0
                        sgn = f_sign(phi_0[x, y, z])
                        #--- Do the interface boundary voxels first --------------------
                        if (phi_0[x, y, z] * phi_0[x-1, y, z] < 0.0) or \
                           (phi_0[x, y, z] * phi_0[x+1, y, z] < 0.0) or \
                           (phi_0[x, y, z] * phi_0[x, y-1, z] < 0.0) or \
                           (phi_0[x, y, z] * phi_0[x, y+1, z] < 0.0) or \
                           (phi_0[x, y, z] * phi_0[x, y, z-1] < 0.0) or \
                           (phi_0[x, y, z] * phi_0[x, y, z+1] < 0.0):
                            gX = 0.0
                            gX = f_max(gX, fabsf((phi_0[x+1, y, z] - phi_0[x-1, y, z]) / 2.0))
                            gX = f_max(gX, fabsf((phi_0[x+1, y, z] - phi_0[x-0, y, z])))
                            gX = f_max(gX, fabsf((phi_0[x+0, y, z] - phi_0[x-1, y, z])))
                            gX = f_max(gX, 1e-9)
                            # 
                            gY = 0.0
                            gY = f_max(gY, fabsf((phi_0[x, y+1, z] - phi_0[x, y-1, z]) / 2.0))
                            gY = f_max(gY, fabsf((phi_0[x, y+1, z] - phi_0[x, y-0, z])))
                            gY = f_max(gY, fabsf((phi_0[x, y+0, z] - phi_0[x, y-1, z])))
                            gY = f_max(gY, 1e-9)
                            # 
                            gZ = 0.0
                            gZ = f_max(gZ, fabsf((phi_0[x, y, z+1] - phi_0[x, y, z-1]) / 2.0))
                            gZ = f_max(gZ, fabsf((phi_0[x, y, z+1] - phi_0[x, y, z-0])))
                            gZ = f_max(gZ, fabsf((phi_0[x, y, z+0] - phi_0[x, y, z-1])))
                            gZ = f_max(gZ, 1e-9)
                            # 
                            #gX = (phi_0[x+1,y] - phi_0[x-1,y]) / 2.0
                            #gY = (phi_0[x,y+1] - phi_0[x,y-1]) / 2.0
                            #gY = (phi_0[x,y+1] - phi_0[x,y-1]) / 2.0
                            # 
                            dist = phi_0[x, y, z] / sqrtf(gX**2 + gY**2 + gZ**2)
                            phi_t[x, y, z] = dist - sgn * fabsf(phi[x, y, z])
                        #--- Do the non-boundary voxels next ---------------------------
                        else:
                            gXm = phi[x+0, y, z] - phi[x-1, y, z]
                            gXp = phi[x+1, y, z] - phi[x-0, y, z]
                            gYm = phi[x, y+0, z] - phi[x, y-1, z]
                            gYp = phi[x, y+1, z] - phi[x, y-0, z]
                            gZm = phi[x, y, z+0] - phi[x, y, z-1]
                            gZp = phi[x, y, z+1] - phi[x, y, z-0]
                            if sgn > 0.0:
                                gXm = f_max(gXm, 0.0)
                                gXp = f_min(gXp, 0.0)
                                gYm = f_max(gYm, 0.0)
                                gYp = f_min(gYp, 0.0)
                                gZm = f_max(gZm, 0.0)
                                gZp = f_min(gZp, 0.0)
                            else:
                                gXm = f_min(gXm, 0.0)
                                gXp = f_max(gXp, 0.0)
                                gYm = f_min(gYm, 0.0)
                                gYp = f_max(gYp, 0.0)
                                gZm = f_min(gZm, 0.0)
                                gZp = f_max(gZp, 0.0)
                            gX = f_max(gXm**2, gXp**2)
                            gY = f_max(gYm**2, gYp**2)
                            gZ = f_max(gZm**2, gZp**2)
                            G = sqrtf(gX + gY + gZ)
                            phi_t[x, y, z] = sgn * (1.0 - G)
                        # end boundary/non-boundary if
                    # end z for loop
                # end y for loop
            # end x for loop
            #---  ----------------------------------------------------------------
            for x in prange(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    for z in range(zmin, zmax+1):
                        phi[x, y, z] = phi[x, y, z] + dt * phi_t[x, y, z]
                    # end z for loop
                # end y for loop
            # end x for loop
            #--- apply BCs and print current status ----------------------------------------------------------------
            max_phi = 0.0
            max_err = 0.0
            BCs_const_first_deriv(phi)
            for x in range(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    for z in range(zmin, zmax+1):
                        max_phi = f_max(fabsf(phi[x,y,z]), max_phi)
                        if fabsf(phi[x,y,z]) <= band:
                            max_err = f_max(fabsf(phi_t[x,y,z]), max_err)
                        # end if statement
                    # end z for loop
                # end y for loop
            # end x for loop
        # end nogil
        if verbose==1:
            fmts = " | {:5d} | {:10d} | {:10.3f} | {:10.3f} | "
            print(fmts.format(iter, num, max_err, max_phi))
        # end print loop
    # end while loop
    return np.asarray(phi, dtype=np.float32)

#=== Constant first derivative boundary conditions =============================
# Actually doing a constant first derivative boundary condition is unstable
cdef void BCs_const_first_deriv(float[:,:,::1] arr) nogil:
    #
    cdef ssize_t x, nx=arr.shape[0]
    cdef ssize_t y, ny=arr.shape[1]
    cdef ssize_t z, nz=arr.shape[2]
    cdef ssize_t xmin=1, xmax=nx-2  # Bounds of the 'good' data
    cdef ssize_t ymin=1, ymax=ny-2  # Bounds of the 'good' data
    cdef ssize_t zmin=1, zmax=nz-2  # Bounds of the 'good' data
    #
    for x in prange(nx):
      for y in range(ny):
        arr[x, y, zmin-1] = arr[x, y, zmin]# - (arr[x, y, zmin+1] - arr[x, y, zmin+0])
        arr[x, y, zmax+1] = arr[x, y, zmax]# - (arr[x, y, zmax-1] - arr[x, y, zmax+0])
    #
    for x in prange(nx):
      for z in range(nz):
        arr[x, ymin-1, z] = arr[x, ymin, z]# - (arr[x, ymin+1, z] - arr[x, ymin+0, z])
        arr[x, ymax+1, z] = arr[x, ymax, z]# - (arr[x, ymax-1, z] - arr[x, ymax+0, z])
    #
    for y in prange(ny):
      for z in range(nz):
        arr[xmin-1, y, z] = arr[xmin, y, z]# - (arr[xmin+1, y, z] - arr[xmin+0, y, z])
        arr[xmax+1, y, z] = arr[xmax, y, z]# - (arr[xmax-1, y, z] - arr[xmax+0, y, z])
