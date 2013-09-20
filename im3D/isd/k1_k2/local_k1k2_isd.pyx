#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

# cython: profile=False
# ==============================================================
import numpy as np
from libc.math cimport fabs, sqrt, floor, ceil, cos
from cython.parallel cimport prange
# ==============================================================
cdef inline ssize_t val2bin(double val, double bin_min, double bin_max, ssize_t nbins) nogil:
    return <ssize_t>floor(0.5+(val-bin_min)/(bin_max-bin_min)*<double>(nbins-1))
# ==============================================================
def calculate_isd(double[:,:,::1] phi, double[:,:,::1] k1, double[:,:,::1] k2,
    double[:,:,::1] P, double bin_min, double bin_max, int nbins, 
    double eps, double local_radius, double k1_val, double k2_val, 
    double dk):
    """
    INPUTS
    ======
        phi - 3D numpy array, required
              This must be a signed distance function
        
        k1 -- 3D numpy array, optional
              Principal curvature
              If this array is not given, it will be calculated.
              k1 < k2 is the typical convention
        
        k2 -- 3D numpy array, optional
              Principal curvature
              If this array is not given, it will be calculated
              k1 < k2 is the typical convention
        
        P --- 3D numpy array, optional
              Property to sum
              If this array is not given, ones will be used; i.e. an 
              area ISD will be calculated
        
        eps - float, optional, default=1.0
        
        
        nbins - int, optional, default=200
        
        
        bin_min - float, optional, default=-0.1
        
        
        bin_max - float, optional, default=+0.1
        
    
    OUTPUTS
    =======
        ISD(k1,k2) - 2D numpy array
              The interfacial shape distribution where k1 < k2
    
    
    EXAMPLE
    =======
    
    
    DETAILS
    =======
        - The ISD can be thought of as many surface integrals for a
          restricted range of curvatures. So for a particular ISD
          bin at k1=k1* and k2=k2*: 
          
                           /
            ISD(k1*,k2*) = | 1 dS*
                           /
            
            where S* is all surface patches with 
            (k1* - bin_size/2) < k1 < (k1* + bin_size/2) and 
            (k2* - bin_size/2) < k2 < (k2* + bin_size/2)
        - The surface integral of 1 will return the total surface
          area of all patches matching that particular k1*, k2*
        - To find the ISD of a property (e.g. interfacial velocity), P(x) is the property instead of 1
    
    IMPLEMENTATION
    ==============
        - Since it is much easier to do a volume integral than a
          surface integral, the following method that is presented
          in ... is used:  
          - If phi is a signed distance function, then the surface
            integral of property P(x) can be approximated by a 
            volume integral:
          
              /           /
              | P(x) dS = | D(phi(x)) P(x) dV
              /           /
              
              where D(phi(x)) is a smeared out delta function with
              width of eps and has the following form:
              
                         1   /     /phi*pi\\
              D(phi) = -----*|1+cos|------|| for -eps < phi < +eps
                       2*eps \     \ eps  //
              and 0 elsewhere
    """    
    # Typed values:
    cdef ssize_t  x, i, nx=phi.shape[0]
    cdef ssize_t  y, j, ny=phi.shape[1]
    cdef ssize_t  z, k, nz=phi.shape[2]
    cdef ssize_t  k1_bin, k2_bin
    cdef ssize_t  n = <int> ceil(local_radius)
    cdef double  delta, area_1, area_2
    cdef double  phi_x, phi_y, phi_z, grad
    cdef double  pi=3.141592653589793
    # Typed arrays:
    cdef double[:,:] ISD=np.zeros((nbins,nbins), dtype=np.float64)
    #
    with nogil:
      for x in range(1+n,nx-1-n):
        for y in range(1+n,ny-1-n):
          for z in range(1+n,nz-1-n):
            if fabs(phi[x,y,z]) <= eps:
              if (k1[x,y,z] > k1_val-dk) * (k1[x,y,z] < k1_val+dk):
                if (k2[x,y,z] > k2_val-dk) * (k2[x,y,z] < k2_val+dk):
                  phi_x = (phi[x+1,y,z] - phi[x-1,y,z])/2.0
                  phi_y = (phi[x,y+1,z] - phi[x,y-1,z])/2.0
                  phi_z = (phi[x,y,z+1] - phi[x,y,z-1])/2.0
                  #
                  grad = sqrt(phi_x**2 + phi_y**2 + phi_z**2)
                  #
                  delta = 1./(2.0*eps) * (1.0 + cos(phi[x,y,z]*pi/eps)) 
                  area_1 = delta * grad
                  for i in range(x-n, x+n):
                    for j in range(y-n, y+n):
                      for k in range(z-n, z+n):
                        if fabs(phi[i,j,k]) <= eps:
                          if sqrt((x-i)**2 + (y-j)**2 + (z-k)**2) < local_radius:
                            phi_x = (phi[i+1,j,k] - phi[i-1,j,k])/2.0
                            phi_y = (phi[i,j+1,k] - phi[i,j-1,k])/2.0
                            phi_z = (phi[i,j,k+1] - phi[i,j,k-1])/2.0
                            #
                            grad = sqrt(phi_x**2 + phi_y**2 + phi_z**2)
                            #
                            delta = 1./(2.0*eps) * (1.0 + cos(phi[i,j,k]*pi/eps)) 
                            area_2 = delta * grad
                            #
                            k1_bin = val2bin(k1[i,j,k], bin_min, bin_max, nbins)
                            k2_bin = val2bin(k2[i,j,k], bin_min, bin_max, nbins)
                            #
                            if (k1_bin >= 0) and (k1_bin < nbins) and \
                               (k2_bin >= 0) and (k2_bin < nbins):
                              ISD[k1_bin, k2_bin] += area_1 * area_2 * P[i,j,k]
                          # end bin if statement
                        # end phi<eps if statement
                      # end k for loop
                    # end j for loop
                  # end i for loop
                # end k2 if  statement
              # end k1 if statement
            # end phi<eps if statement
          # end z for loop
        # end y for loop
      # end x for loop
    # end nogil
    return np.asarray(ISD)

# ==============================================================
