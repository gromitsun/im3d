import numpy as np
from cython.parallel cimport prange
from math_functions cimport sqrt, fabs, d_max, d_min, d_sign
#=== 64-bit floats, Godunov's upwind method reinitialization ===================
def reinit(double[:,::1] phi_0, double dt, int max_it, double band, int verbose):
    #--- Define variables ------------------------------------------------------
    cdef double[:,::1] phi, phi_t
    cdef ssize_t  x, nx=phi_0.shape[0]
    cdef ssize_t  y, ny=phi_0.shape[1]
    cdef ssize_t  xmin=1, xmax=nx-2  # Bounds of the 'good' data
    cdef ssize_t  ymin=1, ymax=ny-2  # Bounds of the 'good' data
    cdef double  gX, gXm, gXp, gY, gYm, gYp, G
    cdef double  V1, V2, V3, V4, V5
    cdef double  sgn, dist
    cdef double  max_phi, max_err
    cdef int  i, iter=0, num=1
    #--- Initialize arrays -----------------------------------------------------
    phi   = np.zeros(shape=(nx,ny), dtype=np.float64)
    phi_t = np.zeros(shape=(nx,ny), dtype=np.float64)
    #
    phi[...] = phi_0[...]
    #--- Do the reinitialization -----------------------------------------------
    iter = 0
    while iter < max_it:
        with nogil:
            iter += 1
            for x in prange(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    sgn = phi_0[x, y] / sqrt(phi_0[x, y]**2 + 1.0**2)
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
                        gXm = d_max(gXm, 0.0)
                        gXp = d_min(gXp, 0.0)
                        gYm = d_max(gYm, 0.0)
                        gYp = d_min(gYp, 0.0)
                    else:
                        gXm = d_min(gXm, 0.0)
                        gXp = d_max(gXp, 0.0)
                        gYm = d_min(gYm, 0.0)
                        gYp = d_max(gYp, 0.0)
                    gX = d_max(gXm**2, gXp**2)
                    gY = d_max(gYm**2, gYp**2)
                    G = sqrt(gX + gY)
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
                    max_phi = d_max(fabs(phi[x,y]), max_phi)
                    if fabs(phi[x,y]) <= band:
                        max_err = d_max(fabs(phi_t[x,y]), max_err)
                    # end near-the-interface if statement
                # end y for loop
            # end x for loop
        # end nogil
        if verbose==1:
            fmts = " | {:5d} | {:10d} | {:10.3f} | {:10.3f} | "
            print(fmts.format(iter, num, max_err, max_phi))
        # end print loop
    # end while loop
    return np.asarray(phi, dtype=np.float64)

cdef double WENO(double V1, double V2, double V3, double V4, double V5) nogil:
    #===========================================================
    #  Based on the WENO method presented in Osher and Fedkiw's 
    #  "Level set methods and dynamic implic surfaces"
    #===========================================================
    cdef double S1, S2, S3
    cdef double W1, W2, W3
    cdef double A1, A2, A3
    cdef double G1, G2, G3
    cdef double eps
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
cdef void constant_value_BC(double[:,::1] arr) nogil:
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
