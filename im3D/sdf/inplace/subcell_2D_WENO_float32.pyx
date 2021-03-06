import numpy as np
from cython.parallel cimport prange
from math_functions cimport sqrtf, fabsf, f_max, f_min, f_sign
#=== 32-bit floats, WENO method reinitialization ===============================
cpdef void reinit(float[:,::1] phi_0, float[:,::1] phi,
                  float dt=0.4, int max_it=25, float band=3.0, int verbose=0):
    #--- Define variables ------------------------------------------------------
    cdef float[:,::1] phi_t
    cdef ssize_t  x, nx=phi_0.shape[0]
    cdef ssize_t  y, ny=phi_0.shape[1]
    cdef ssize_t  xmin=3, xmax=nx-4  # Bounds of the 'good' data
    cdef ssize_t  ymin=3, ymax=ny-4  # Bounds of the 'good' data
    cdef float  gX, gXm, gXp, gY, gYm, gYp, G
    cdef float  V1, V2, V3, V4, V5
    cdef float  sgn, dist
    cdef float  max_phi, max_err
    cdef int  i, iter=0, num=1
    #--- Initialize arrays -----------------------------------------------------
    phi_t = np.empty(shape=(nx,ny), dtype=np.float32)
    #
    if &phi[0, 0] != &phi_0[0, 0]:
        phi[...] = phi_0[...]
    #--- Do the reinitialization -----------------------------------------------
    iter = 0
    while iter < max_it:
        with nogil:
            iter += 1
            for x in prange(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    sgn = f_sign(phi_0[x, y])
                    #--- Do the interface boundary voxels first --------------------
                    if (phi_0[x, y] * phi_0[x, y-1] < 0.0) or (phi_0[x, y] * phi_0[x, y+1] < 0.0) or \
                       (phi_0[x, y] * phi_0[x-1, y] < 0.0) or (phi_0[x, y] * phi_0[x+1, y] < 0.0):
                        gX = 0.0
                        gX = f_max(gX, fabsf((phi_0[x+1, y] - phi_0[x-1, y]) / 2.0))
                        gX = f_max(gX, fabsf((phi_0[x+1, y] - phi_0[x-0, y])))
                        gX = f_max(gX, fabsf((phi_0[x+0, y] - phi_0[x-1, y])))
                        gX = f_max(gX, 1e-9)
                        # 
                        gY = 0.0
                        gY = f_max(gY, fabsf((phi_0[x, y+1] - phi_0[x, y-1]) / 2.0))
                        gY = f_max(gY, fabsf((phi_0[x, y+1] - phi_0[x, y-0])))
                        gY = f_max(gY, fabsf((phi_0[x, y+0] - phi_0[x, y-1])))
                        gY = f_max(gY, 1e-9)
                        # 
                        dist = phi_0[x,y] / sqrtf(gX**2 + gY**2)
                        phi_t[x, y] = dist - sgn * fabsf(phi[x, y])
                    #--- Do the non-boundary voxels next -----------------------
                    else:
                        #--- backward derivative in x-direction ---
                        V1 = phi[x-2, y] - phi[x-3, y]
                        V2 = phi[x-1, y] - phi[x-2, y]
                        V3 = phi[x+0, y] - phi[x-1, y]
                        V4 = phi[x+1, y] - phi[x+0, y]
                        V5 = phi[x+2, y] - phi[x+1, y]
                        gXm = WENO(V1, V2, V3, V4, V5)
                        #--- forward derivative in x-direction---
                        V1 = phi[x+3, y] - phi[x+2, y]
                        V2 = phi[x+2, y] - phi[x+1, y]
                        V3 = phi[x+1, y] - phi[x+0, y]
                        V4 = phi[x+0, y] - phi[x-1, y]
                        V5 = phi[x-1, y] - phi[x-2, y]
                        gXp = WENO(V1, V2, V3, V4, V5)
                        #--- backward derivative in y-direction ---
                        V1 = phi[x, y-2] - phi[x, y-3]
                        V2 = phi[x, y-1] - phi[x, y-2]
                        V3 = phi[x, y+0] - phi[x, y-1]
                        V4 = phi[x, y+1] - phi[x, y+0]
                        V5 = phi[x, y+2] - phi[x, y+1]
                        gYm = WENO(V1, V2, V3, V4, V5)
                        #--- forward derivative in y-direction ---
                        V1 = phi[x, y+3] - phi[x, y+2]
                        V2 = phi[x, y+2] - phi[x, y+1]
                        V3 = phi[x, y+1] - phi[x, y+0]
                        V4 = phi[x, y+0] - phi[x, y-1]
                        V5 = phi[x, y-1] - phi[x, y-2]
                        gYp = WENO(V1, V2, V3, V4, V5)
                        #--- use only upwind values using godunov's method ---
                        if sgn > 0.0:
                            gXm = f_max(gXm, 0.0)
                            gXp = f_min(gXp, 0.0)
                            gYm = f_max(gYm, 0.0)
                            gYp = f_min(gYp, 0.0)
                        else:
                            gXm = f_min(gXm, 0.0)
                            gXp = f_max(gXp, 0.0)
                            gYm = f_min(gYm, 0.0)
                            gYp = f_max(gYp, 0.0)
                        gX = f_max(gXm**2, gXp**2)
                        gY = f_max(gYm**2, gYp**2)
                        G = sqrtf(gX + gY)
                        phi_t[x, y] = sgn * (1.0 - G)
                    # end boundary/non-boundary if
                # end y for loop
            # end x for loop
            #--- Update phi and apply BCs --------------------------------------
            for x in prange(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    phi[x,y] = phi[x,y] + dt*phi_t[x,y]
                # end y for loop
            # end x for loop
            constant_value_BC(phi)
            #--- print current status ------------------------------------------
            max_phi = 0.0
            max_err = 0.0
            for x in range(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    max_phi = f_max(fabsf(phi[x,y]), max_phi)
                    if fabsf(phi[x,y]) <= band:
                        max_err = f_max(fabsf(phi_t[x,y]), max_err)
                    # end near-the-interface if statement
                # end y for loop
            # end x for loop
        # end nogil
        if verbose:
            fmts = " | {:5d} | {:10d} | {:10.3f} | {:10.3f} | "
            print(fmts.format(iter, num, max_err, max_phi))
        # end print loop
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

#=== Set the first derivative normal to the boundary to zero ===================
cdef void constant_value_BC(float[:,::1] arr) nogil:
    #--- Define variables ------------------------------------------------------
    cdef ssize_t  x, nx=arr.shape[0]
    cdef ssize_t  y, ny=arr.shape[1]
    cdef ssize_t  xmin=3, xmax=nx-4  # Bounds of the 'good' data
    cdef ssize_t  ymin=3, ymax=ny-4  # Bounds of the 'good' data
    #--- Apply the boundary conditions -----------------------------------------
    for x in prange(nx):
        arr[x, ymin-1] = arr[x, ymin]# - 1.0 * (arr[x, ymin+1, z] - arr[x, ymin+0, z])
        arr[x, ymin-2] = arr[x, ymin]# - 2.0 * (arr[x, ymin+1, z] - arr[x, ymin+0, z])
        arr[x, ymin-3] = arr[x, ymin]# - 3.0 * (arr[x, ymin+1, z] - arr[x, ymin+0, z])
        arr[x, ymax+1] = arr[x, ymax]# - 1.0 * (arr[x, ymax-1, z] - arr[x, ymax+0, z])
        arr[x, ymax+2] = arr[x, ymax]# - 2.0 * (arr[x, ymax-1, z] - arr[x, ymax+0, z])
        arr[x, ymax+3] = arr[x, ymax]# - 3.0 * (arr[x, ymax-1, z] - arr[x, ymax+0, z])
    for y in prange(ny):
        arr[xmin-1, y] = arr[xmin, y]# - 1.0 * (arr[xmin+1, y, z] - arr[xmin+0, y, z])
        arr[xmin-2, y] = arr[xmin, y]# - 2.0 * (arr[xmin+1, y, z] - arr[xmin+0, y, z])
        arr[xmin-3, y] = arr[xmin, y]# - 3.0 * (arr[xmin+1, y, z] - arr[xmin+0, y, z])
        arr[xmax+1, y] = arr[xmax, y]# - 1.0 * (arr[xmax-1, y, z] - arr[xmax+0, y, z])
        arr[xmax+2, y] = arr[xmax, y]# - 2.0 * (arr[xmax-1, y, z] - arr[xmax+0, y, z])
        arr[xmax+3, y] = arr[xmax, y]# - 3.0 * (arr[xmax-1, y, z] - arr[xmax+0, y, z])
