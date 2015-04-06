import numpy as np
from cython.parallel cimport prange
from math_functions cimport sqrtf, fabsf, f_max, f_min, f_sign
#=== 32-bit floats, WENO method reinitialization ===============================
cpdef void reinit(float[:,:,::1] phi_0, float[:,:,::1] phi,
                  float dt=0.4, int max_it=25, float band=3.0, int verbose=0):
    #--- Define variables ------------------------------------------------------
    cdef float[:,:,::1] phi_t
    cdef ssize_t  x, nx=phi_0.shape[0]
    cdef ssize_t  y, ny=phi_0.shape[1]
    cdef ssize_t  z, nz=phi_0.shape[2]
    cdef ssize_t  xmin=3, xmax=nx-4  # Bounds of the 'good' data
    cdef ssize_t  ymin=3, ymax=ny-4  # Bounds of the 'good' data
    cdef ssize_t  zmin=3, zmax=nz-4  # Bounds of the 'good' data
    cdef float  gX, gXm, gXp, gY, gYm, gYp, gZ, gZm, gZp, G
    cdef float  V1, V2, V3, V4, V5
    cdef float  sgn, dist
    cdef float  max_phi, max_err
    cdef int  i, iter=0, num=1
    #--- Initialize arrays -----------------------------------------------------
    phi_t = np.empty(shape=(nx,ny,nz), dtype=np.float32)
    #
    if &phi[0, 0, 0] != &phi_0[0, 0, 0]:
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
                        #--- Do the interface boundary voxels first ------------
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
                            dist = phi_0[x, y, z] / sqrtf(gX**2 + gY**2 + gZ**2)
                            phi_t[x, y, z] = dist - sgn * fabsf(phi[x, y, z])
                        #--- Do the non-boundary voxels next -------------------
                        else:
                            #--- backward derivative in x-direction ---
                            V1 = phi[x-2, y, z] - phi[x-3, y, z]
                            V2 = phi[x-1, y, z] - phi[x-2, y, z]
                            V3 = phi[x+0, y, z] - phi[x-1, y, z]
                            V4 = phi[x+1, y, z] - phi[x+0, y, z]
                            V5 = phi[x+2, y, z] - phi[x+1, y, z]
                            gXm = WENO(V1, V2, V3, V4, V5)
                            #--- forward derivative in x-direction---
                            V1 = phi[x+3, y, z] - phi[x+2, y, z]
                            V2 = phi[x+2, y, z] - phi[x+1, y, z]
                            V3 = phi[x+1, y, z] - phi[x+0, y, z]
                            V4 = phi[x+0, y, z] - phi[x-1, y, z]
                            V5 = phi[x-1, y, z] - phi[x-2, y, z]
                            gXp = WENO(V1, V2, V3, V4, V5)
                            #--- backward derivative in y-direction ---
                            V1 = phi[x, y-2, z] - phi[x, y-3, z]
                            V2 = phi[x, y-1, z] - phi[x, y-2, z]
                            V3 = phi[x, y+0, z] - phi[x, y-1, z]
                            V4 = phi[x, y+1, z] - phi[x, y+0, z]
                            V5 = phi[x, y+2, z] - phi[x, y+1, z]
                            gYm = WENO(V1, V2, V3, V4, V5)
                            #--- forward derivative in y-direction ---
                            V1 = phi[x, y+3, z] - phi[x, y+2, z]
                            V2 = phi[x, y+2, z] - phi[x, y+1, z]
                            V3 = phi[x, y+1, z] - phi[x, y+0, z]
                            V4 = phi[x, y+0, z] - phi[x, y-1, z]
                            V5 = phi[x, y-1, z] - phi[x, y-2, z]
                            gYp = WENO(V1, V2, V3, V4, V5)
                            #--- backward derivative in z-direction ---
                            V1 = phi[x, y, z-2] - phi[x, y, z-3]
                            V2 = phi[x, y, z-1] - phi[x, y, z-2]
                            V3 = phi[x, y, z+0] - phi[x, y, z-1]
                            V4 = phi[x, y, z+1] - phi[x, y, z+0]
                            V5 = phi[x, y, z+2] - phi[x, y, z+1]
                            gZm = WENO(V1, V2, V3, V4, V5)
                            #--- forward derivative in z-direction ---
                            V1 = phi[x, y, z+3] - phi[x, y, z+2]
                            V2 = phi[x, y, z+2] - phi[x, y, z+1]
                            V3 = phi[x, y, z+1] - phi[x, y, z+0]
                            V4 = phi[x, y, z+0] - phi[x, y, z-1]
                            V5 = phi[x, y, z-1] - phi[x, y, z-2]
                            gZp = WENO(V1, V2, V3, V4, V5)
                            #--- use only upwind values using godunov's method ---
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
            #--- apply BCs and print current status ----------------------------
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
        if verbose:
            fmts = " | {:5d} | {:10d} | {:10.3f} | {:10.3f} | "
            print(fmts.format(iter, num, max_err, max_phi))
    # end while loop


cdef float WENO(float V1, float V2, float V3, float V4, float V5) nogil:
    #===========================================================
    #  Based on the WENO method presented in Osher and Fedkiw's 
    #  "Level set methods and dynamic implic surfaces"
    #===========================================================
    cdef float S1, S2, S3
    cdef float W1, W2, W3
    cdef float A1, A2, A3
    cdef float G1, G2, G3
    cdef float eps
    #===========================================================
    S1 = + (13./12.) * (1.*V1 - 2.*V2 + 1.*V3)**2 \
         + ( 1./ 4.) * (1.*V1 - 4.*V2 + 3.*V3)**2
    #
    S2 = + (13./12.) * (1.*V2 - 2.*V3 + 1.*V4)**2 \
         + ( 1./ 4.) * (1.*V2 - 1.*V4)**2
    #
    S3 = + (13./12.) * (1.*V3 - 2.*V4 + 1.*V5)**2 \
         + ( 1./ 4.) * (3.*V3 - 4.*V2 + 1.*V5)**2
    #===========================================================
    eps = 1E-6  #(V1+V2+V3+V4+V5)/5.*1E-6 + 1E-10
    #
    A1 = 0.1/(S1+eps)**2
    A2 = 0.6/(S2+eps)**2
    A3 = 0.3/(S3+eps)**2
    #===========================================================
    W1 = A1/(A1+A2+A3)
    W2 = A2/(A1+A2+A3)
    W3 = A3/(A1+A2+A3)
    #===========================================================
    G1 = (1./6.) * (+ 2.*V1 - 7.*V2 + 11.*V3)
    G2 = (1./6.) * (- 1.*V2 + 5.*V3 +  2.*V4)
    G3 = (1./6.) * (+ 2.*V3 + 5.*V4 -  1.*V5)
    #===========================================================
    return W1*G1 + W2*G2 + W3*G3
# ==============================================================
cdef void BCs_const_first_deriv(float[:,:,::1] arr) nogil:
    #
    cdef ssize_t  x, nx=arr.shape[0]
    cdef ssize_t  y, ny=arr.shape[1]
    cdef ssize_t  z, nz=arr.shape[2]
    cdef ssize_t  xmin=3, xmax=nx-4  # Bounds of the 'good' data
    cdef ssize_t  ymin=3, ymax=ny-4  # Bounds of the 'good' data
    cdef ssize_t  zmin=3, zmax=nz-4  # Bounds of the 'good' data
    #
    for x in prange(nx):
      for y in range(ny):
        arr[x, y, zmin-1] = arr[x, y, zmin]# - 1.0 * (arr[x, y, zmin+1] - arr[x, y, zmin+0])
        arr[x, y, zmin-2] = arr[x, y, zmin]# - 2.0 * (arr[x, y, zmin+1] - arr[x, y, zmin+0])
        arr[x, y, zmin-3] = arr[x, y, zmin]# - 3.0 * (arr[x, y, zmin+1] - arr[x, y, zmin+0])
        arr[x, y, zmax+1] = arr[x, y, zmax]# - 1.0 * (arr[x, y, zmax-1] - arr[x, y, zmax+0])
        arr[x, y, zmax+2] = arr[x, y, zmax]# - 2.0 * (arr[x, y, zmax-1] - arr[x, y, zmax+0])
        arr[x, y, zmax+3] = arr[x, y, zmax]# - 3.0 * (arr[x, y, zmax-1] - arr[x, y, zmax+0])
    #
    for x in prange(nx):
      for z in range(nz):
        arr[x, ymin-1, z] = arr[x, ymin, z]# - 1.0 * (arr[x, ymin+1, z] - arr[x, ymin+0, z])
        arr[x, ymin-2, z] = arr[x, ymin, z]# - 2.0 * (arr[x, ymin+1, z] - arr[x, ymin+0, z])
        arr[x, ymin-3, z] = arr[x, ymin, z]# - 3.0 * (arr[x, ymin+1, z] - arr[x, ymin+0, z])
        arr[x, ymax+1, z] = arr[x, ymax, z]# - 1.0 * (arr[x, ymax-1, z] - arr[x, ymax+0, z])
        arr[x, ymax+2, z] = arr[x, ymax, z]# - 2.0 * (arr[x, ymax-1, z] - arr[x, ymax+0, z])
        arr[x, ymax+3, z] = arr[x, ymax, z]# - 3.0 * (arr[x, ymax-1, z] - arr[x, ymax+0, z])
    #
    for y in prange(ny):
      for z in range(nz):
        arr[xmin-1, y, z] = arr[xmin, y, z]# - 1.0 * (arr[xmin+1, y, z] - arr[xmin+0, y, z])
        arr[xmin-2, y, z] = arr[xmin, y, z]# - 2.0 * (arr[xmin+1, y, z] - arr[xmin+0, y, z])
        arr[xmin-3, y, z] = arr[xmin, y, z]# - 3.0 * (arr[xmin+1, y, z] - arr[xmin+0, y, z])
        arr[xmax+1, y, z] = arr[xmax, y, z]# - 1.0 * (arr[xmax-1, y, z] - arr[xmax+0, y, z])
        arr[xmax+2, y, z] = arr[xmax, y, z]# - 2.0 * (arr[xmax-1, y, z] - arr[xmax+0, y, z])
        arr[xmax+3, y, z] = arr[xmax, y, z]# - 3.0 * (arr[xmax-1, y, z] - arr[xmax+0, y, z])

# ==============================================================
