#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport fabs, sqrt, cos
from cython.parallel cimport prange
# ==============================================================
def imc(phi, H=None, eps=2.5, reinit=True):
    """
    INPUTS
    ======
      phi --> 2D or 3D numpy array, required
              Reference array.  The more accurate of the two
              input arrays.
              
      H ----> 2D or 3D numpy array, required
              
              
      eps --> float, optional (default=2.5)
              Half-width of the delta function (DD).  This  
              does not have much effect on the AID value - 
              eps values between 0.5 and 10.0 have been  
              tried without changing the value of AID.  
              Values above 10 do start to affect the 
              resulting AID.  
    
    OUTPUTS
    =======
      IMC --> float
              Value of the AID between the two arrays
    
    NOTES
    =====
      The input array
    """
    import curv
    import sdf
    import numpy as np
    # ==========================================================
    # 
    if reinit==True:
        phi = sdf.reinit(phi, tol=0.50, dt=0.25, verbose=0, max_it=100)
    if H==None:
        H=curv.H(phi, reinit=False)
    # 
    # ==========================================================
    # Create the delta function using the reference array 
    # values.  Zero the outer bounds because they don't have
    # much significance due to the boundary conditions applied
    # when creating the SDFs
    dd = np.zeros_like(phi)
    w = np.where(np.abs(phi) <= eps)
    dd[w] = 1./(2.*eps) + 1./(2.*eps)*np.cos(phi[w]*np.pi/eps)
    #
    dd[+0:+5, ...] = 0.0
    dd[-5:-1, ...] = 0.0
    dd[..., +0:+5] = 0.0
    dd[..., -5:-1] = 0.0
    #
    # ==========================================================
    # Calculate the integral mean curvature (IMC)
    #
    IMC = np.sum(dd * np.abs(H)) / np.sum(dd)
    #IMC_sq = np.sum(dd * H**2) / np.sum(dd)
    #IMC = IMC_sq**0.5
    #
    return IMC

