from __future__ import absolute_import
#=== Mean curvature ============================================================
def H(phi):
    """
    Mean curvature
    DESCRIPTION
    ===========
        Calculates the mean curvature from either a 2D or 3D  
        signed distance function.  In 2D:
        
            phi_x^2*phi_yy + phi_y^2*phi_xx - 2*phi_x*phi_y*phi_xy
        H = ------------------------------------------------------
                           sqrt(phi_x^2 + phi_y^2)
    
    INPUTS
    ======
        phi --> 2D or 3D numpy array
    
    
    OUTPUTS
    =======
        H ----> 2D or 3D numpy array
                This will have the same dimensions as phi
                It will also have the same datatype as phi (either 
                32- or 64-bit floats)
    
    """
    import numpy as np
    from im3D.curvature import curv_2D
    from im3D.curvature import curv_3D
    #=== check the number of dimensions ===
    ndim = phi.ndim
    if ndim not in [2, 3]:
        raise ValueError("Only 2D and 3D arrays supported")
    #=== check the datatype ===
    dtype = phi.dtype
    if dtype not in [np.float32, np.float64]:
        raise ValueError('Datatype must be 32 or 64 bit floats')
    #=== calculate the curvature ===
    arr_req = ['C_CONTIGUOUS', 'ALIGNED']
    if (ndim == 2) and (dtype == np.float32):
        phi = np.require(phi, dtype=np.float32, requirements=arr_req)
        return curv_2D.H_32(phi)
    elif (ndim == 2) and (dtype == np.float64):
        phi = np.require(phi, dtype=np.float64, requirements=arr_req)
        return curv_2D.H_64(phi)
    elif (ndim == 3) and (dtype == np.float32):
        phi = np.require(phi, dtype=np.float32, requirements=arr_req)
        return curv_3D.H_32(phi)
    elif (ndim == 3) and (dtype == np.float64):
        phi = np.require(phi, dtype=np.float64, requirements=arr_req)
        return curv_3D.H_64(phi)

#=== Gaussian curvature ========================================================
def K(phi):
    """
        Only works for 3D arrays because for a 2D array, the 
        mean and Gaussian curvatures are the same.
    """
    import numpy as np
    from im3D.curvature import curv_2D
    from im3D.curvature import curv_3D
    #=== check the number of dimensions ===
    ndim = phi.ndim
    if ndim not in [3,]:
        raise ValueError("Only 3D arrays supported for the Gaussian curvature")
    #=== check the datatype ===
    dtype = phi.dtype
    if dtype not in [np.float32, np.float64]:
        raise ValueError('Datatype must be 32 or 64 bit floats')
    #=== calculate the curvature ===
    arr_req = ['C_CONTIGUOUS', 'ALIGNED']
    if (ndim == 3) and (dtype == np.float32):
        phi = np.require(phi, dtype=np.float32, requirements=arr_req)
        return curv_3D.K_32(phi)
    elif (ndim == 3) and (dtype == np.float64):
        phi = np.require(phi, dtype=np.float64, requirements=arr_req)
        return curv_3D.K_64(phi)

#=== First (smaller) principal curvature =======================================
def k1(phi):
    """
    Principal curvature
    
    DESCRIPTION
    ===========
        Calculates one of the principal curvatures a 3D signed 
        distance function:
        
        k1 = H - sqrt(H^2-K)
        
    INPUTS
    ======
        phi --> 3D numpy array
                The signed distance function
    
    
    OUTPUTS
    =======
        k1 ---> 3D numpy array
                This will have the same dimensions as phi
    
    """
    import numpy as np
    from im3D.curvature import curv_2D
    from im3D.curvature import curv_3D
    #=== check the number of dimensions ===
    ndim = phi.ndim
    if ndim not in [3,]:
        raise ValueError("Only 3D arrays supported for the principal curvatures")
    #=== check the datatype ===
    dtype = phi.dtype
    if dtype not in [np.float32, np.float64]:
        raise ValueError('Datatype must be 32 or 64 bit floats')
    #=== calculate the curvature ===
    arr_req = ['C_CONTIGUOUS', 'ALIGNED']
    if (ndim == 3) and (dtype == np.float32):
        phi = np.require(phi, dtype=np.float32, requirements=arr_req)
        return curv_3D.k1_32(phi)
    elif (ndim == 3) and (dtype == np.float64):
        phi = np.require(phi, dtype=np.float64, requirements=arr_req)
        return curv_3D.k1_64(phi)

#=== Second (larger) principal curvature =======================================
def k2(phi):
    """
    Principal curvature
    
    DESCRIPTION
    ===========
        Calculates one of the principal curvatures a 3D signed 
        distance function:
        
        k2 = H + sqrt(H^2-K)
        
    INPUTS
    ======
        phi --> 3D numpy array
                The signed distance function
    
    
    OUTPUTS
    =======
        k2 ---> 3D numpy array
                This will have the same dimensions as phi
    
    """
    import numpy as np
    from im3D.curvature import curv_2D
    from im3D.curvature import curv_3D
    #=== check the number of dimensions ===
    ndim = phi.ndim
    if ndim not in [3,]:
        raise ValueError("Only 3D arrays supported for the principal curvatures")
    #=== check the datatype ===
    dtype = phi.dtype
    if dtype not in [np.float32, np.float64]:
        raise ValueError('Datatype must be 32 or 64 bit floats')
    #=== calculate the curvature ===
    arr_req = ['C_CONTIGUOUS', 'ALIGNED']
    if (ndim == 3) and (dtype == np.float32):
        phi = np.require(phi, dtype=np.float32, requirements=arr_req)
        return curv_3D.k2_32(phi)
    elif (ndim == 3) and (dtype == np.float64):
        phi = np.require(phi, dtype=np.float64, requirements=arr_req)
        return curv_3D.k2_64(phi)

