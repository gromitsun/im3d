#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport fabs, sqrt, cos
from cython.parallel cimport prange
# ==============================================================
def aid(ref_arr, alt_arr, ref_thr=0.0, alt_thr=0.0, eps=2.5, 
        DS_it=20, DS_dt=0.1, SDF_dt=0.25, SDF_tol=0.10, SDF_band=3.0):
    """
    
    Average interfacial displacement (AID)
    ======================================
      For two arrays (ref_arr and alt_arr) with implicitly defined interfaces at the given threshold values, this code will calculate the average distance between the two sets of interfaces, normalized by linear pixel length (for 2D arrays) or pixel area (for 3D data).  AID is defined by the following equation:
      
             surface integral of the interfacial displacement
      AID = --------------------------------------------------
                         total surface area
      
      The surface integral is done using a procedure that is in "Level Sets and Dynamic Implicit Surfaces" by S. Osher and R. Fedkiw in which a signed distance function is calculated from the input array and used to determine the distance of any given voxel from the nearest interface.  With the distances known, a smeared delta function about the interface can be calculated.  This delta function has a value of zero far from the interface and finite values near the interface (in 1D, it is shaped like a gaussian distribution centered at the interface location).  When the delta function is multiplied by the property of interest and summed over all voxels, it gives the surface integral of that property.  To get the numerator in the above equation, the interfacial displacement is the summed-over property and to get the total surface area, the summed-over property is a constant value of 1.0.
      
      Or, in discrete form:
      
             sum( A_i * |ID_i| )
      AID = -------------------
                sum( A_i )
      
      where the sums are over all voxels (i), A_i is the surface area of voxel i and |ID_i| is absolute value of the interfacial displacement of voxel i.  The reference array is used for calculating the surface area
      
      
    INPUTS
    ======
      ref_arr --> 2D or 3D numpy array, required
                  Reference array.  The more accurate of the two
                  input arrays.
                  
      alt_arr --> 2D or 3D numpy array, required
                  Alternate array.  The array thats accuarcy is 
                  being tested
                  
      ref_thr --> float, optional (default=0.0)
                  Threshold value for the reference array
                  
      alt_thr --> float, optional (default=0.0)
                  Threshold value for the alternate array
                  
      eps ------> float, optional (default=2.5)
                  Half-width of the delta function (DD).  This  
                  does not have much effect on the AID value - 
                  eps values between 0.5 and 10.0 have been  
                  tried without changing the value of AID.  
                  Values above 10 do start to affect the 
                  resulting AID.  
                  
      DS_it ----> integer, optional (default=20)
                  Number of diffusion smoothing iterations after
                  the initial SDF calculation
                  
      DS_dt ----> float, optional (default==0.1)
                  Timestep for diffusion smoothing.  Timesteps 
                  of 0.1 are stable for most datasets but 
                  smaller timesteps might be required if  the 
                  initial data is very noisy.
                  
      SDF_dt ---> float, optional (default==0.25) 
                  Timestep for signed distance function (SDF)
                  calculation.  For data that starts with an
                  initial spatial gradient at the interface 
                  that is close to 1.0, timesteps up to 0.75 
                  and higher are stable.  For less well 
                  conditioned data, timesteps of 0.1 to 0.25 
                  should be used
                  
      SDF_tol --> float, optional (default==0.05)
                  Tolerance of the SDF.  How close the gradients
                  of the SDF have to be to 1.0.  Values of 0.01 
                  to 0.05 should be used for best accuracy
      
    OUTPUTS
    =======
      AID ------> float
                  Value of the AID between the two arrays
      
    NOTES
    =====
      Because this method uses signed distance functions to do 
      some of the comparisons between the two arrays, it can be
      helpful if the input arrays are scaled so that it is easy
      to calculate SDFs from them.  This means that they should
      have a spatial gradient across the interface of about 
      1.0.  This isn't necessary but it can make the process 
      faster.  Even if the gradients aren't close to 1.0, they
      should be similar to each other so that if there are any
      distortions due to the SDF calculation process, it is 
      roughly the same for both arrays.
      
      The only difference between the reference array and the
      alternate array is that the surface is calculated from 
      the reference array.  Switching the input order of the 
      two arrays will obviously switch which array is used to 
      calculate the surface area; doing so does not greatly 
      affect the resulting AID value but it is probably still a 
      good idea to use the more accurate of the two arrays for 
      the reference.
      
    """
    import sdf
    import smoothing
    import numpy as np
    # ==========================================================
    # Scale the two arrays to have the interfaces at arr=0
    SDF_ref = ref_arr - ref_thr
    SDF_alt = alt_arr - alt_thr
    # Create signed distance functions from the two arrays
    # SDF_ref = sdf.reinit(SDF_ref, tol=SDF_tol, dt=SDF_dt, band=SDF_band, verbose=0, max_it=100, use_weno=False)
    # SDF_alt = sdf.reinit(SDF_alt, tol=SDF_tol, dt=SDF_dt, band=SDF_band, verbose=0, max_it=100, use_weno=False)
    # ==========================================================
    # Diffusion smooth the two arrays to reduce the noise.  By
    # smoothing both, the larger differences are preserved but 
    # some of the noise can be removed
    SDF_ref = smoothing.ds(SDF_ref, DS_it, DS_dt)
    SDF_alt = smoothing.ds(SDF_alt, DS_it, DS_dt)
    # ==========================================================
    # Reinitialize the two signed distance functions to fix any
    # areas where they were altered by the diffusion smoothing
    SDF_ref = sdf.reinit(SDF_ref, tol=SDF_tol, dt=SDF_dt, band=SDF_band, verbose=0, max_it=100, use_weno=False)
    SDF_alt = sdf.reinit(SDF_alt, tol=SDF_tol, dt=SDF_dt, band=SDF_band, verbose=0, max_it=100, use_weno=False)
    # ==========================================================
    # Create the delta function using the reference array 
    # values.  Zero the outer bounds because they don't have
    # much significance due to the boundary conditions applied
    # when creating the SDFs
    d_ref = np.zeros_like(ref_arr)
    w_ref = np.where(np.abs(SDF_ref) <= eps)
    d_ref[w_ref] = 1./(2.*eps) + 1./(2.*eps)*np.cos(SDF_ref[w_ref]*np.pi/eps)
    #
    d_alt = np.zeros_like(alt_arr)
    w_alt = np.where(np.abs(SDF_alt) <= eps)
    d_alt[w_alt] = 1./(2.*eps) + 1./(2.*eps)*np.cos(SDF_alt[w_alt]*np.pi/eps)
    #
    dd = 0.5*(d_alt + d_ref)
    #dd = d_alt * d_ref
    dd[+0:+5, ...] = 0.0
    dd[-5:-1, ...] = 0.0
    dd[..., +0:+5] = 0.0
    dd[..., -5:-1] = 0.0
    #
    # ==========================================================
    # Calculate the interfacial displacement (ID) and the 
    # integral interfacial displacement (AID)
    ID = SDF_ref - SDF_alt
    #
    AID = np.sum(dd*np.abs(ID))/np.sum(dd)
    AID_RMS = np.sum(dd*ID**2)/np.sum(dd)
    AID_RMS = np.sqrt(AID_RMS)
    #
    return AID_RMS
    # ==========================================================
    

