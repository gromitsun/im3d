from __future__ import absolute_import

def ds(arr, it=10, dt=0.10, D=None, bc='mirror'):
    """
    Diffusion smoothing
    
    INPUTS
    ======
      arr --> 1D, 2D, 3D or 4D numpy array
              Array that is to be smoothed
      it ---> integer, optional (default=10)
              number of iterations to do
      dt ---> float, optional (default=0.10)
              size of timestep
      D ----> numpy array with an entry for each axis in arr
              Diffusivity along each of the axes.  Default is 
              for diffusivity to be 1.0 in all directions.
      bc ---> string, optional, default='mirror'
              Specify the boundary condition
              'mirror' - gradients are computed as if there were
                         a mirror at the boundaries
              'fixed' -- the value at the boundary isn't allowed
                         to change
      
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
    # ==========================================================
    #                            1D      2D      3D      4D  
    #                          ======  ======  ======  ======
    from im3D.smoothing import DS_1D,  DS_2D,  DS_3D,  DS_4D
    # ==========================================================
    import numpy as np
    #
    if D == None:
        D = np.ones((arr.ndim))
    else:
        D = np.array(D, dtype=np.float64)
    if arr.ndim != D.size:
        print("D must have exactly one value for each axis in arr")
        return arr
    #
    if  bc == 'fixed':
        bc_type = 0
    elif bc == 'mirror':
        bc_type = 1
    else:
        print("bc must have a value of 'mirror' or 'fixed'")
        return arr
    #
    it  = int(it)
    dt  = np.float64(dt)
    arr = np.require(arr, dtype=np.float64, requirements=['ALIGNED', 'C_CONTIGUOUS'])
    #
    if len(arr.shape) == 1:
        Dx, = D[...]
        return DS_1D.isotropic(arr, it, dt, Dx, bc_type)
    elif len(arr.shape) == 2:
        Dx,Dy, = D[...]
        return np.asarray(DS_2D.isotropic(arr, it, dt, Dx, Dy, bc_type))
    elif len(arr.shape) == 3:
        Dx,Dy,Dz, = D[...]
        return np.asarray(DS_3D.isotropic(arr, it, dt, Dx, Dy, Dz, bc_type))
    elif len(arr.shape) == 4:
        D0,D1,D2,D3, = D[...]
        return np.asarray(DS_4D.isotropic(arr, it, dt, D0, D1, D2, D3, bc_type))
    else:
        print("Array must be 1, 2, 3 or 4 dimensions")
    #

# ==============================================================
def aniso_ds(arr, it=10, dt=0.10, bc='mirror'):
    """
    Modified diffusion smoothing
        phi_t = T(phi) dot laplacian(phi)
    
    INPUTS
    ======
      arr --> 2D or 3D numpy array
              Array that is to be smoothed
      it ---> integer, optional (default=10)
              number of iterations to do
      dt ---> float, optional (default=0.10)
              size of timestep
      bc ---> string, optional, default='mirror'
              Specify the boundary condition
              'mirror' - gradients are computed as if there were
                         a mirror at the boundaries
              'fixed' -- the value at the boundary isn't allowed
                         to change
      
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
    # ==========================================================
    #                             1D      2D      3D      4D  
    #                           ======  ======  ======  ======
    from im3D.smoothing import  DS_1D,  DS_2D,  DS_3D,  DS_4D
    # ==========================================================
    import numpy as np
    #
    if  bc == 'fixed':
        bc_type = 0
    elif bc == 'mirror':
        bc_type = 1
    else:
        print("bc must have a value of 'mirror' or 'fixed'")
        return arr
    #
    it  = int(it)
    dt  = np.float64(dt)
    arr = np.require(arr, dtype=np.float64, requirements=['ALIGNED', 'C_CONTIGUOUS'])
    #
    if len(arr.shape) == 2:
        return np.asarray(DS_2D.anisotropic(arr, it, dt, bc_type))
    elif len(arr.shape) == 3:
        return np.asarray(DS_3D.anisotropic(arr, it, dt, bc_type))
    else:
        print("Array must be 2 or 3 dimensions")
    #

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
    import numpy as np
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

