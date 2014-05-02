from __future__ import absolute_import

# ==============================================================
def subcell(phi, dt=0.25, max_it=25):
    import numpy as np
    from im3D.sdf.f_reinit_3D_UW import reinit
    # ==============================================================
    phi = np.require(phi, dtype=np.float32, requirements=('C', 'A'))
    phi = reinit(phi, dt, max_it, 5.0, 1)
    return phi

# ==============================================================
def subcell_WENO(phi, dt=0.25, max_it=25):
    import numpy as np
    from im3D.sdf.f_reinit_3D_WENO import reinit
    # ==============================================================
    phi = np.require(phi, dtype=np.float32, requirements=('C', 'A'))
    phi = reinit(phi, dt, max_it, 5.0, 1)
    return phi

def reinit(phi, dt=0.25, tol=0.25, band=None, verbose=False, max_it=-1):
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
    ndim = phi.ndim
    dtype = phi.dtype
    if ndim not in [2, 3]:
        raise ValueError('This function only works for 2D or 3D arrays')
    if dtype not in [np.float32, np.float64]:
        raise ValueError('This function only works for 32-bit and 64-bit values')
    # 
    phi = np.require(phi, requirements=('C_CONTIGUOUS', 'ALIGNED'))
    verbose = np.intc(verbose)
    use_weno = np.intc(use_weno)
    if band == None:
        band = np.size(phi)
    #
    if   (phi.ndim == 2) and (np.dtype == np.float32):
        phi = reinit_2D.reinit(phi, dt, tol, band, verbose, max_it, use_weno)
    elif (phi.ndim == 2) and (np.dtype == np.float64):
        phi = reinit_2D.reinit(phi, dt, tol, band, verbose, max_it, use_weno)
    elif (phi.ndim == 3) and (np.dtype == np.float32):
        phi = reinit_3D.reinit(phi, dt, tol, band, verbose, max_it, use_weno)
    elif (phi.ndim == 3) and (np.dtype == np.float64):
        phi = reinit_3D.reinit(phi, dt, tol, band, verbose, max_it, use_weno)
    #
    return phi
# ==============================================================
