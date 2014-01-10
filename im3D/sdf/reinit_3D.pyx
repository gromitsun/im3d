#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

# cython: profile=False
# ==============================================================
import numpy as np
cimport cython
from cython.parallel cimport prange
# ==============================================================
# C functions:
cdef extern from "math.h":
    double sqrt(double) nogil
    double fabs(double) nogil
cdef inline double d_max(double a, double b) nogil:
    return a if a >= b else b
cdef inline double d_min(double a, double b) nogil:
    return a if a <= b else b
cdef inline double sign(double a) nogil:
    return 0.0 if a == 0.0 else sqrt(a*a)/a
# ==============================================================
def reinit(double[:,:,::1] in_arr, double dt, 
    double tol, double band, int verbose, int max_it, 
    int use_weno):
    # ==========================================================
    cdef:
        double[:,:,::1] phi, phi_t
        ssize_t  bdry=3
        ssize_t  x, nX = in_arr.shape[0] + 2*bdry
        ssize_t  y, nY = in_arr.shape[1] + 2*bdry
        ssize_t  z, nZ = in_arr.shape[2] + 2*bdry
        double   max_phi, max_err
        int      i, iter=0, num=1
    # ==========================================================
    phi = np.zeros(shape=(nX,nY,nZ), dtype=np.float64)
    phi[bdry:-bdry, bdry:-bdry, bdry:-bdry] = in_arr.copy()
    phi = ApplyBCs_3D(phi, bdry)
    #
    phi_t = np.zeros(shape=(nX,nY,nZ), dtype=np.float64)
    # ======================================================
    while num > 0:
        iter += 1
        if (max_it >= 0) and (iter > max_it):
          break
        num = 0
        max_err = 0.0
        max_phi = 0.0
        #=========================================================
        if use_weno == 1:
            phi_t = WENO_phi_t(phi)
        else:
            phi_t = UW_phi_t(phi)
        #=======================================================
        with nogil:
            for x in prange(0+bdry,nX-bdry):
              for y in range(0+bdry,nY-bdry):
                for z in range(0+bdry,nZ-bdry):
                    phi[x,y,z] = phi[x,y,z] + dt*phi_t[x,y,z]
            phi = ApplyBCs_3D(phi, bdry)
            #=========================================================
            for x in prange(0+bdry,nX-bdry):
              for y in range(0+bdry,nY-bdry):
                for z in range(0+bdry,nZ-bdry):
                    if fabs(phi[x,y,z]) <= band:
                        if fabs(phi_t[x,y,z]) > tol:
                            num += 1
            if verbose==2:
                for x in range(0+bdry,nX-bdry):
                  for y in range(0+bdry,nY-bdry):
                    for z in range(0+bdry,nZ-bdry):
                        if fabs(phi[x,y,z]) <= band:
                            max_err = d_max(fabs(phi_t[x,y,z]), max_err)
                            max_phi = d_max(fabs(phi[x,y,z]), max_phi)
        # end nogil
        if verbose==1:
            fmts = " | {:5d} | {:10d} | "
            print(fmts.format(iter, num))
        if verbose==2:
            fmts = " | {:5d} | {:10d} | {:10.3f} | {:10.3f} | "
            print(fmts.format(iter, num, max_err, max_phi))
        #=========================================================
    # end while loop
    return np.asarray(phi[bdry:-bdry, bdry:-bdry, bdry:-bdry])
# ==============================================================
cdef double[:,:,::1] UW_phi_t(double[:,:,::1] phi):
    """
    Upwind gradient
    """
    # ==========================================================
    cdef:
        ssize_t  bdry=3
        ssize_t  x, nX = phi.shape[0]
        ssize_t  y, nY = phi.shape[1]
        ssize_t  z, nZ = phi.shape[2]
        double sgn, gX, gXm, gXp, gY, gYm, gYp, gZ, gZm, gZp
        double[:,:,::1] phi_t = np.empty((nX,nY,nZ), dtype=np.float64)
    #===========================================================
    with nogil:
      for x in prange(0+bdry,nX-bdry):
        for y in range(0+bdry,nY-bdry):
          for z in range(0+bdry,nZ-bdry):
            phi_t[x,y,z] = 0.0
            #=================================================
            # CALCULATE THE 'SMEARED' SIGN OF PHI.  THE SMEARING
            # IS DONE WITH THE +1.0 IN THE SQRT AND IS DONE TO
            # SLOW THE RATE AT WHICH VALUES NEAR THE INTERFACE 
            # CHANGE.
            sgn = phi[x,y,z]/sqrt(phi[x,y,z]**2 + 1.0)
            #======================================================
            # CALCULATE BOTH BACKWARD (e.g. gXm) AND FORWARD 
            # (e.g. gXp) DERIVATIVES
            gXm = phi[x,  y,  z  ] - phi[x-1,y,  z  ]
            gXp = phi[x+1,y,  z  ] - phi[x,  y,  z  ]
            gYm = phi[x,  y,  z  ] - phi[x,  y-1,z  ]
            gYp = phi[x,  y+1,z  ] - phi[x,  y,  z  ]
            gZm = phi[x,  y,  z  ] - phi[x,  y,  z-1]
            gZp = phi[x,  y,  z+1] - phi[x,  y,  z  ]
            #===============================================
            # USE ONLY UPWIND VALUES USING GODUNOV'S METHOD:
            if sgn > 0.0:
                gXm = d_max(+gXm, 0.0)
                gXp = d_max(-gXp, 0.0)
                gYm = d_max(+gYm, 0.0)
                gYp = d_max(-gYp, 0.0)
                gZm = d_max(+gZm, 0.0)
                gZp = d_max(-gZp, 0.0)
            else:
                gXm = d_max(-gXm, 0.0)
                gXp = d_max(+gXp, 0.0)
                gYm = d_max(-gYm, 0.0)
                gYp = d_max(+gYp, 0.0)
                gZm = d_max(-gZm, 0.0)
                gZp = d_max(+gZp, 0.0)
            gX = d_max(gXm, gXp)
            gY = d_max(gYm, gYp)
            gZ = d_max(gZm, gZp)
            phi_t[x,y,z] = sgn * (1.0 - sqrt(gX*gX + gY*gY + gZ*gZ))
          # end z for loop
        # end y for loop
      # end x for loop
    return phi_t
#===============================================================
cdef double[:,:,::1] WENO_phi_t(double[:,:,::1] phi):
    """
    WENO gradient based dphi/dt
    """
    # ==========================================================
    cdef:
        ssize_t  bdry=3
        ssize_t  x, nX = phi.shape[0]
        ssize_t  y, nY = phi.shape[1]
        ssize_t  z, nZ = phi.shape[2]
        double   gX, gXm, gXp, gY, gYm, gYp, gZ, gZm, gZp
        double   V1, V2, V3, V4, V5
        double   sgn
        #
        double[:,:,::1] phi_t = np.empty((nX,nY,nZ), dtype=np.float64)
    # ==========================================================
    with nogil:
      for x in prange(0+bdry,nX-bdry):
        for y in range(0+bdry,nY-bdry):
          for z in range(0+bdry,nZ-bdry):
            phi_t[x,y,z] = 0.0
            #=====================================================
            # CALCULATE THE 'SMEARED' SIGN OF PHI.  THE SMEARING
            # IS DONE WITH THE +1.0 IN THE SQRT AND IS DONE TO
            # SLOW THE RATE AT WHICH VALUES NEAR THE INTERFACE 
            # CHANGE.
            sgn = phi[x,y,z]/sqrt(phi[x,y,z]**2 + 1.0E-2)
            #=====================================================
            # BACKWARD DERIVATIVE IN X-DIRECTION
            V1 = phi[x-2, y, z] - phi[x-3, y, z]
            V2 = phi[x-1, y, z] - phi[x-2, y, z]
            V3 = phi[x+0, y, z] - phi[x-1, y, z]
            V4 = phi[x+1, y, z] - phi[x+0, y, z]
            V5 = phi[x+2, y, z] - phi[x+1, y, z]
            
            gXm = WENO(V1, V2, V3, V4, V5)
            #=====================================================
            # FORWARD DERIVATIVE IN X-DIRECTION
            V1 = phi[x+3, y, z] - phi[x+2, y, z]
            V2 = phi[x+2, y, z] - phi[x+1, y, z]
            V3 = phi[x+1, y, z] - phi[x+0, y, z]
            V4 = phi[x+0, y, z] - phi[x-1, y, z]
            V5 = phi[x-1, y, z] - phi[x-2, y, z]
            
            gXp = WENO(V1, V2, V3, V4, V5)
            #=====================================================
            # BACKWARD DERIVATIVE IN Y-DIRECTION
            V1 = phi[x, y-2, z] - phi[x, y-3, z]
            V2 = phi[x, y-1, z] - phi[x, y-2, z]
            V3 = phi[x, y+0, z] - phi[x, y-1, z]
            V4 = phi[x, y+1, z] - phi[x, y+0, z]
            V5 = phi[x, y+2, z] - phi[x, y+1, z]
            
            gYm = WENO(V1, V2, V3, V4, V5)
            #=====================================================
            # FORWARD DERIVATIVE IN Y-DIRECTION
            V1 = phi[x, y+3, z] - phi[x, y+2, z]
            V2 = phi[x, y+2, z] - phi[x, y+1, z]
            V3 = phi[x, y+1, z] - phi[x, y+0, z]
            V4 = phi[x, y+0, z] - phi[x, y-1, z]
            V5 = phi[x, y-1, z] - phi[x, y-2, z]
            
            gYp = WENO(V1, V2, V3, V4, V5)
            #=====================================================
            # BACKWARD DERIVATIVE IN Z-DIRECTION
            V1 = phi[x, y, z-2] - phi[x, y, z-3]
            V2 = phi[x, y, z-1] - phi[x, y, z-2]
            V3 = phi[x, y, z+0] - phi[x, y, z-1]
            V4 = phi[x, y, z+1] - phi[x, y, z+0]
            V5 = phi[x, y, z+2] - phi[x, y, z+1]
            
            gZm = WENO(V1, V2, V3, V4, V5)
            #=====================================================
            # FORWARD DERIVATIVE IN Z-DIRECTION
            V1 = phi[x, y, z+3] - phi[x, y, z+2]
            V2 = phi[x, y, z+2] - phi[x, y, z+1]
            V3 = phi[x, y, z+1] - phi[x, y, z+0]
            V4 = phi[x, y, z+0] - phi[x, y, z-1]
            V5 = phi[x, y, z-1] - phi[x, y, z-2]
            
            gZp = WENO(V1, V2, V3, V4, V5)
            #=====================================================
            # USE ONLY UPWIND VALUES USING GODUNOV'S METHOD:
            if sgn > 0.0:
              gXm = d_max(+gXm, 0.0)
              gXp = d_max(-gXp, 0.0)
              gYm = d_max(+gYm, 0.0)
              gYp = d_max(-gYp, 0.0)
              gZm = d_max(+gZm, 0.0)
              gZp = d_max(-gZp, 0.0)
            else:
              gXm = d_max(-gXm, 0.0)
              gXp = d_max(+gXp, 0.0)
              gYm = d_max(-gYm, 0.0)
              gYp = d_max(+gYp, 0.0)
              gZm = d_max(-gZm, 0.0)
              gZp = d_max(+gZp, 0.0)
              #
            gX = d_max(gXm, gXp)
            gY = d_max(gYm, gYp)
            gZ = d_max(gZm, gZp)
            phi_t[x,y,z] = sgn * (1.0 - sqrt(gX*gX + gY*gY + gZ*gZ))
          # end z for loop
        # end y for loop
      # end x for loop
    return phi_t
#===============================================================
cdef double WENO(double V1, double V2, double V3, double V4, double V5) nogil:
    #===========================================================
    #  Based on the WENO method presented in Osher and Fedkiw's 
    #  "Level set methods and dynamic implic surfaces"
    #===========================================================
    cdef:
        double S1, S2, S3
        double W1, W2, W3
        double A1, A2, A3
        double G1, G2, G3
        double eps
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
cdef double[:,:,::1] ApplyBCs_3D(double[:,:,::1] phi, ssize_t bdry) nogil:
    #
    cdef:
        ssize_t i
        ssize_t x, nX=phi.shape[0]
        ssize_t y, nY=phi.shape[1]
        ssize_t z, nZ=phi.shape[2]
    #
    for i in range(bdry):
        for x in prange(nX):
          for y in range(nY):
            phi[x,y,   0+i] = phi[x,y,   0+bdry]
            phi[x,y,nZ-1-i] = phi[x,y,nZ-1-bdry]
        #
        for x in prange(nX):
          for z in range(nZ):
            phi[x,   0+i,z] = phi[x,   0+bdry,z]
            phi[x,nY-1-i,z] = phi[x,nY-1-bdry,z]
        #
        for y in prange(nY):
          for z in range(nZ):
            phi[   0+i,y,z] = phi[   0+bdry,y,z]
            phi[nX-1-i,y,z] = phi[nX-1-bdry,y,z]
    #
    return phi
# ==============================================================
