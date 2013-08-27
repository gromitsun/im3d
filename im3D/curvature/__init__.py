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
    
    """
    import numpy as np
    import curv_2D
    import curv_3D
    phi = np.require(phi, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
    #
    if phi.ndim == 2:
        return curv_2D.H(phi)
    elif phi.ndim == 3:
        return curv_3D.H(phi)
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
    import curv_2D
    import curv_3D
    phi = np.require(phi, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
    #
    if phi.ndim == 3:
        return curv_3D.K(phi)
    else:
        print("Only 3D arrays supported")
        return None
# ==============================================================
def K1(phi):
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
        H ----> 3D numpy array
                This will have the same dimensions as phi
    
    """
    import numpy as np
    import curv_2D
    import curv_3D
    phi = np.require(phi, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
    #
    if phi.ndim == 3:
        return curv_3D.K1(phi)
    else:
        print("Only 3D arrays supported")
        return None
# ==============================================================
def K2(phi):
    """
    Principal curvature
    
    DESCRIPTION
    ===========
        Calculates one of the principal curvatures a 3D signed 
        distance function:
        
        K2 = H + sqrt(H^2-K)
        
    INPUTS
    ======
        phi --> 3D numpy array
                The signed distance function
    
    
    OUTPUTS
    =======
        H ----> 3D numpy array
                This will have the same dimensions as phi
    
    """
    import numpy as np
    import curv_2D
    import curv_3D
    phi = np.require(phi, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
    #
    if phi.ndim == 3:
        return curv_3D.K2(phi)
    else:
        print("Only 3D arrays supported")
        return None
# ==============================================================
