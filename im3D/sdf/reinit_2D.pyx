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
def reinit(double[:,:] in_arr, double dt, 
    double tol, double band, int verbose, int max_it, 
    int use_weno):
    # ==========================================================
    cdef:
        double[:,:] phi, phi_t
        ssize_t  bdry=3
        ssize_t  x, nX = in_arr.shape[0] + 2*bdry
        ssize_t  y, nY = in_arr.shape[1] + 2*bdry
        double   max_phi, max_err
        int      i, iter=0, num=1
    # ==========================================================
    phi = np.zeros(shape=(nX,nY), dtype=np.float64)
    phi[bdry:-bdry, bdry:-bdry] = in_arr.copy()
    phi = ApplyBCs_2D(phi, bdry)
    #
    phi_t = np.zeros(shape=(nX,nY), dtype=np.float64)
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
                    phi[x,y] = phi[x,y] + dt*phi_t[x,y]
            phi = ApplyBCs_2D(phi, bdry)
            #===================================================
            for x in prange(0+bdry,nX-bdry):
                for y in range(0+bdry,nY-bdry):
                    if fabs(phi[x,y]) <= band:
                        if fabs(phi_t[x,y]) > tol:
                            num += 1
            if verbose==2:
                for x in range(0+bdry,nX-bdry):
                    for y in range(0+bdry,nY-bdry):
                        if fabs(phi[x,y]) <= band:
                            max_err = d_max(fabs(phi_t[x,y]), max_err)
                            max_phi = d_max(fabs(phi[x,y]), max_phi)
        #=======================================================
        if verbose==1:
            fmts = " | {:5d} | {:10d} | "
            print fmts.format(iter, num)
        if verbose==2:
            fmts = " | {:5d} | {:10d} | {:10.3f} | {:10.3f} | "
            print fmts.format(iter, num, max_err, max_phi)
        #=========================================================
    # end while loop
    return np.asarray(phi[bdry:-bdry, bdry:-bdry])
# ==============================================================
cdef double[:,:] UW_phi_t(double[:,:] phi):
    """
    Upwind gradient
    """
    # ==========================================================
    cdef:
        ssize_t  bdry=3
        ssize_t  x, nX = phi.shape[0]
        ssize_t  y, nY = phi.shape[1]
        double   sgn, gX, gXm, gXp, gY, gYm, gYp
        double[:,:] phi_t = np.empty((nX,nY), dtype=np.float64)
    # ==========================================================
    with nogil:
      for x in prange(0+bdry,nX-bdry):
        for y in range(0+bdry,nY-bdry):
          #======================================================
          # CALCULATE THE 'SMEARED' SIGN OF PHI.  THE SMEARING
          # IS DONE WITH THE +1.0 IN THE SQRT AND IS DONE TO
          # SLOW THE RATE AT WHICH VALUES NEAR THE INTERFACE 
          # CHANGE.
          sgn = phi[x,y]/sqrt(phi[x,y]**2 + 1.0E-2)
          # ==========================================================
          # CALCULATE BOTH BACKWARD (e.g. gXm) AND FORWARD 
          # (e.g. gXp) DERIVATIVES
          gXm = phi[x,  y  ] - phi[x-1,y  ]
          gXp = phi[x+1,y  ] - phi[x,  y  ]
          gYm = phi[x,  y  ] - phi[x,  y-1]
          gYp = phi[x,  y+1] - phi[x,  y  ]
          #===============================================
          # USE ONLY UPWIND VALUES USING GODUNOV'S METHOD:
          if sgn > 0.0:
              gXm = d_max(+gXm, 0.0)
              gXp = d_max(-gXp, 0.0)
              gYm = d_max(+gYm, 0.0)
              gYp = d_max(-gYp, 0.0)
          else:
              gXm = d_max(-gXm, 0.0)
              gXp = d_max(+gXp, 0.0)
              gYm = d_max(-gYm, 0.0)
              gYp = d_max(+gYp, 0.0)
          gX = d_max(gXm, gXp)
          gY = d_max(gYm, gYp)
          phi_t[x,y] = sgn * (1.0 - sqrt(gX*gX + gY*gY))
        # end y for loop
      # end x for loop
    return phi_t
#===============================================================
cdef double[:,:] WENO_phi_t(double[:,:] phi):
    """
    WENO gradient based dphi/dt
    """
    # ==========================================================
    cdef:
        ssize_t  bdry=3
        ssize_t  x, nX = phi.shape[0]
        ssize_t  y, nY = phi.shape[1]
        double   gX, gXm, gXp, gY, gYm, gYp
        double   V1, V2, V3, V4, V5
        double   sgn
        #
        double[:,:] phi_t = np.empty((nX,nY), dtype=np.float64)
    # ==========================================================
    with nogil:
      for x in prange(0+bdry,nX-bdry):
        for y in range(0+bdry,nY-bdry):
          phi_t[x,y] = 0.0
          #=====================================================
          # CALCULATE THE 'SMEARED' SIGN OF PHI.  THE SMEARING
          # IS DONE WITH THE +1.0 IN THE SQRT AND IS DONE TO
          # SLOW THE RATE AT WHICH VALUES NEAR THE INTERFACE 
          # CHANGE.
          sgn = phi[x,y]/sqrt(phi[x,y]**2 + 1.0E-2)
          #=====================================================
          # BACKWARD DERIVATIVE IN X-DIRECTION
          V1 = phi[x-2, y] - phi[x-3, y]
          V2 = phi[x-1, y] - phi[x-2, y]
          V3 = phi[x+0, y] - phi[x-1, y]
          V4 = phi[x+1, y] - phi[x+0, y]
          V5 = phi[x+2, y] - phi[x+1, y]
          
          gXm = WENO(V1, V2, V3, V4, V5)
          #=====================================================
          # FORWARD DERIVATIVE IN X-DIRECTION
          V1 = phi[x+3, y] - phi[x+2, y]
          V2 = phi[x+2, y] - phi[x+1, y]
          V3 = phi[x+1, y] - phi[x+0, y]
          V4 = phi[x+0, y] - phi[x-1, y]
          V5 = phi[x-1, y] - phi[x-2, y]
          
          gXp = WENO(V1, V2, V3, V4, V5)
          #=====================================================
          # BACKWARD DERIVATIVE IN Y-DIRECTION
          V1 = phi[x, y-2] - phi[x, y-3]
          V2 = phi[x, y-1] - phi[x, y-2]
          V3 = phi[x, y+0] - phi[x, y-1]
          V4 = phi[x, y+1] - phi[x, y+0]
          V5 = phi[x, y+2] - phi[x, y+1]
          
          gYm = WENO(V1, V2, V3, V4, V5)
          #=====================================================
          # FORWARD DERIVATIVE IN Y-DIRECTION
          V1 = phi[x, y+3] - phi[x, y+2]
          V2 = phi[x, y+2] - phi[x, y+1]
          V3 = phi[x, y+1] - phi[x, y+0]
          V4 = phi[x, y+0] - phi[x, y-1]
          V5 = phi[x, y-1] - phi[x, y-2]
          
          gYp = WENO(V1, V2, V3, V4, V5)
          #=====================================================
          # USE ONLY UPWIND VALUES USING GODUNOV'S METHOD:
          if sgn > 0.0:
              gXm = d_max(+gXm, 0.0)
              gXp = d_max(-gXp, 0.0)
              gYm = d_max(+gYm, 0.0)
              gYp = d_max(-gYp, 0.0)
          else:
              gXm = d_max(-gXm, 0.0)
              gXp = d_max(+gXp, 0.0)
              gYm = d_max(-gYm, 0.0)
              gYp = d_max(+gYp, 0.0)
          gX = d_max(gXm, gXp)
          gY = d_max(gYm, gYp)
          phi_t[x,y] = sgn * (1.0 - sqrt(gX*gX + gY*gY))
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
cdef double[:,:] ApplyBCs_2D(double[:,:] phi, ssize_t bdry) nogil:
    #
    cdef:
        ssize_t i
        ssize_t x, nX=phi.shape[0]
        ssize_t y, nY=phi.shape[1]
    #
    for i in range(bdry):
        for x in prange(nX):
            phi[x,   0+i] = phi[x,   0+bdry]
            phi[x,nY-1-i] = phi[x,nY-1-bdry]
        for y in prange(nY):
            phi[   0+i,y] = phi[   0+bdry,y]
            phi[nX-1-i,y] = phi[nX-1-bdry,y]
    #
    return phi
# ==============================================================
