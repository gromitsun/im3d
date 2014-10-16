from __future__ import absolute_import

def ds(arr, it=10, dt=0.25, D=None):
    """
    Diffusion smoothing
    
    INPUTS
    ======
      arr --> 1D, 2D, 3D or 4D numpy array
              Array that is to be smoothed
      it ---> integer, optional (default=10)
              number of iterations to do
      dt ---> float, optional (default=0.25)
              size of timestep
      D ----> numpy array with an entry for each axis in arr
              Diffusivity along each of the axes.  Default is 
              for diffusivity to be equal in all directions.
              Values are normalized so that this vector has a magnitude of one.
      
    OUTPUTS
    =======
      out --> Numpy array
              Smoothed array.  Has the same dimensions as the
              input array.
      
    DESCRIPTION
    ===========
      
    
    NOTES
    =====
     - Assumes a grid spacing of dX=1.0, dY=1.0 (dZ=1.0).
     - No flux boundary conditions
    """
    import numpy as np
    from im3D.smoothing import DS_1D_f32, DS_1D_f64
    from im3D.smoothing import DS_2D_f32, DS_2D_f64
    from im3D.smoothing import DS_3D_f32, DS_3D_f64
    from im3D.smoothing import DS_4D
    #=== Check inputs ==========================================================
    dtype = arr.dtype
    ndims = arr.ndim
    
    if dtype == np.float64:
        asfloat = np.float64
    else:
        asfloat = np.float32
        dtype = np.float32
    
    if ndims not in [1, 2, 3, 4]:
        raise ValueError("Array must be 1, 2, 3 or 4 dimensions")
    
    if D == None:
        D = np.ones((arr.ndim), dtype=dtype)
    else:
        D = np.array(D, dtype=dtype)
    
    D = D / np.sum(D)
    
    if arr.ndim != D.size:
        raise ValueError("D must have exactly one value for each axis in arr")
    #
    #if  bc == 'fixed':
    #    bc_type = 0
    #elif bc == 'mirror':
    #    bc_type = 1
    #else:
    #    raise ValueError("bc must have a value of 'mirror' or 'fixed'")
    #
    it  = int(it)
    dt  = asfloat(dt)
    arr = np.require(arr, dtype=dtype, requirements=['C_contiguous'])
    #
    if ndims == 1:
        Dx, = D[...]
    elif ndims == 2:
        Dx,Dy, = D[...]
    elif ndims == 3:
        Dx,Dy,Dz, = D[...]
    else:
        D0,D1,D2,D3, = D[...]
    
    if (ndims == 1) and (dtype == np.float32):
        result = DS_1D_f32.isotropic(arr, it, dt, Dx)
    elif (ndims == 1) and (dtype == np.float64):
        result = DS_1D_f64.isotropic(arr, it, dt, Dx)
    elif (ndims == 2) and (dtype == np.float32):
        result = DS_2D_f32.isotropic(arr, it, dt, Dx, Dy)
    elif (ndims == 2) and (dtype == np.float64):
        result = DS_2D_f64.isotropic(arr, it, dt, Dx, Dy)
    elif (ndims == 3) and (dtype == np.float32):
        result = DS_3D_f32.isotropic(arr, it, dt, Dx, Dy, Dz)
    elif (ndims == 3) and (dtype == np.float64):
        result = DS_3D_f64.isotropic(arr, it, dt, Dx, Dy, Dz)
    else:
        result = np.asarray(DS_4D.isotropic(arr, it, dt, D0, D1, D2, D3, bc_type))
    
    return result

# ==============================================================
def mmc(arr, it=25, dt=0.0025, scale=None):
    # ==========================================================
    #                            1D      2D      3D      4D  
    #                          ======  ======  ======  ======
    from im3D.smoothing import        MMC_2D, MMC_3D         
    # ==========================================================
    """
    Motion by mean curvature
    
    INPUTS
    ======
      arr --> 2D or 3D numpy array
              Array that is to be smoothed
      it ---> integer, optional (default=25)
              number of iterations to do
      dt ---> float, optional (default=0.005)
              size of timestep
      scale > float, optional (default=(arr.max-arr.min))
              Scaling factor to account for mean curvature
              not scaling linearly with the values of the input 
              array
      
    OUTPUTS
    =======
      out --> Numpy array
              Smoothed array.  Has the same dimensions as the
              input array.
      
    DESCRIPTION
    ===========
      
    
    NOTES
    =====
     - Assumes a grid spacing of dX=1.0, dY=1.0 (dZ=1.0).
     - No flux boundary conditions
    """
    #
    if scale == None:
        scale =  np.amax(arr) - np.amin(arr)
    it  = int(it)
    dt  = np.float64(dt) * scale
    eps = 1e-3 * scale
    arr = np.require(arr, dtype=np.float64, requirements=['ALIGNED', 'C_CONTIGUOUS'])
    if len(arr.shape) == 2:
        return np.asarray(MMC_2D.MMC_2D(arr, it, dt, eps))
    elif len(arr.shape) == 3:
        return np.asarray(MMC_3D.MMC_3D(arr, it, dt, eps))
    else:
        print("Array must be 2 or 3 dimensions")
    #

del absolute_import
