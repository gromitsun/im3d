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
    phi = np.require(phi, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
    #
    if phi.ndim == 2:
        phi = np.require(phi, dtype=np.float64)
        return curv_2D.H(phi)
    elif phi.ndim == 3:
        if phi.dtype == np.float32:
            return curv_3D.H_32(phi)
        elif phi.dtype == np.float64:
            return curv_3D.H_64(phi)
        else:
            raise ValueError('Datatype must be 32 or 64 bit floats')
    else:
        print("Only 2D and 3D arrays supported")
        return None
# ==============================================================
def K(phi):
    """
        Only works for 3D arrays because for a 2D array, the 
        mean and Gaussian curvatures are the same.
    """
    import numpy as np
    from im3D.curvature import curv_2D
    from im3D.curvature import curv_3D
    phi = np.require(phi, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
    #
    if phi.ndim == 3:
        if phi.dtype == np.float32:
            return curv_3D.K_32(phi)
        elif phi.dtype == np.float64:
            return curv_3D.K_64(phi)
        else:
            raise ValueError('Datatype must be 32 or 64 bit floats')
    else:
        print("Only 3D arrays supported")
        return None
# ==============================================================
def k1(phi):
    """
    Principal curvature
    
    DESCRIPTION
    ===========
        Calculates one of the principal curvatures a 3D signed 
        distance function:
        
        K1 = H - sqrt(H^2-K)
        
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
    phi = np.require(phi, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
    #
    if phi.ndim == 3:
        if phi.dtype == np.float32:
            return curv_3D.k1_32(phi)
        elif phi.dtype == np.float64:
            return curv_3D.k1_64(phi)
        else:
            raise ValueError('Datatype must be 32 or 64 bit floats')
    else:
        print("Only 3D arrays supported")
        return None
# ==============================================================
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
    phi = np.require(phi, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
    #
    if phi.ndim == 3:
        if phi.dtype == np.float32:
            return curv_3D.k2_32(phi)
        elif phi.dtype == np.float64:
            return curv_3D.k2_64(phi)
        else:
            raise ValueError('Datatype must be 32 or 64 bit floats')
    else:
        print("Only 3D arrays supported")
        return None
# ==============================================================
