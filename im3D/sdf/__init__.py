# import im3D.sdf.test as test

def reinit(phi, dt=0.25, tol=0.25, band=5.0, verbose=1, 
           max_it=-1, use_weno=False):
    """
    INPUTS
    ======
        phi: 2D or 3D numpy array, required
            The initial conditions for creating the signed 
            distance function (SDF)
        dt: float, optional, default=0.25
            Timestep for SDF calculation
        tol: float, optional, default=0.2
            Tolerance for how close the gradient should be to 1
        band: float, optional, default=5.0
            How far to extend the SDF
        verbose: int, optional, default=1
            Flag for how much information should be printed
            at every iteration.
            0 -> No print out
            1 -> Print iteration number and number of 'active'
                 voxels
            2 -> Same as 1 plus max |phi_t| and max |phi|
                 NB: This is much slower than 0 or 1
    
    OUTPUTS
    =======
        phi: 2D or 3D numpy array
             The sigend distance function
    NOTES
    =====
        
    """
    import numpy as np
    #
    from im3D.sdf import reinit_2D
    from im3D.sdf import reinit_3D
    # ==============================================================
    phi = np.require(phi, dtype=np.float64, requirements=('C', 'A'))
    verbose = np.intc(verbose)
    use_weno = np.intc(use_weno)
    #
    if phi.ndim == 2:
        phi = reinit_2D.reinit(phi, dt, tol, band, verbose, max_it, use_weno)
    elif phi.ndim == 3:
        phi = reinit_3D.reinit(phi, dt, tol, band, verbose, max_it, use_weno)
    else:
        raise ValueError("phi must be 2 or 3 dimesnsions")
    #
    return phi
# ==============================================================
