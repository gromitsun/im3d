from __future__ import absolute_import
#=== 2D non-subcell ============================================================
from im3D.sdf import reinit_2D_UW_float32
from im3D.sdf import reinit_2D_UW_float64
from im3D.sdf import reinit_2D_WENO_float32
from im3D.sdf import reinit_2D_WENO_float64
#=== 3D non-subcell ============================================================
from im3D.sdf import reinit_3D_UW_float32
from im3D.sdf import reinit_3D_UW_float64
from im3D.sdf import reinit_3D_WENO_float32
from im3D.sdf import reinit_3D_WENO_float64
#=== 2D subcell ================================================================
from im3D.sdf import subcell_2D_UW_float32
from im3D.sdf import subcell_2D_UW_float64
from im3D.sdf import subcell_2D_WENO_float32
from im3D.sdf import subcell_2D_WENO_float64
#=== 3D subcell ================================================================
from im3D.sdf import subcell_3D_UW_float32
from im3D.sdf import subcell_3D_UW_float64
from im3D.sdf import subcell_3D_WENO_float32
from im3D.sdf import subcell_3D_WENO_float64

def reinit(arr, dt=0.40, it=25, subcell=True, WENO=True, verbose=0, band=3.0):
    """
    
    """
    import numpy as np
    
    dtype = arr.dtype
    ndim = arr.ndim
    if ndim not in [2,3]:
        raise ValueError("The input array must be 2D or 3D")
    #--- check datatype ---
    if dtype == 'float32':
        f32 = True
        f64 = False
    elif dtype == 'float64':
        f32 = False
        f64 = True
    else:
        raise TypeError('Only 32-bit and 64-bit arrays are accepted')
    #--- use WENO or upwind ---
    if WENO == True:
        UW = False
        WENO = True
    else:
        UW = True
        WENO = False
    #--- subcell or regular reinit ---
    if subcell == True:
        regular = False
        subcell = True
    else:
        regular = True
        subcell = False
    #--- sanity checks for inputs ---
    if dt > 0.4:
        raise ValueError("You're fucking crazy! There's no way a timestep of %.1f is stable" % dt)
    if it < 0:
        raise ValueError("The number of iterations must be positive")
    if verbose == 1:
        verbose == int(1)
    else:
        verbose == int(0)
    
    arr = np.require(arr, dtype=dtype, requirements=('C_contiguous',))
    
    if (ndim == 2) and f32 and WENO and subcell:
        out = subcell_2D_WENO_float32.reinit(arr, dt, it, band, verbose)
    elif (ndim == 2) and f32 and WENO and regular:
        out = reinit_2D_WENO_float32.reinit(arr, dt, it, band, verbose)
    elif (ndim == 2) and f32 and UW and subcell:
        out = subcell_2D_UW_float32.reinit(arr, dt, it, band, verbose)
    elif (ndim == 2) and f32 and UW and regular:
        out = reinit_2D_UW_float32.reinit(arr, dt, it, band, verbose)
    elif (ndim == 2) and f64 and WENO and subcell:
        out = subcell_2D_WENO_float64.reinit(arr, dt, it, band, verbose)
    elif (ndim == 2) and f64 and WENO and regular:
        out = reinit_2D_WENO_float64.reinit(arr, dt, it, band, verbose)
    elif (ndim == 2) and f64 and UW and subcell:
        out = subcell_2D_UW_float64.reinit(arr, dt, it, band, verbose)
    elif (ndim == 2) and f64 and UW and regular:
        out = reinit_2D_UW_float64.reinit(arr, dt, it, band, verbose)
    elif (ndim == 3) and f32 and WENO and subcell:
        out = subcell_3D_WENO_float32.reinit(arr, dt, it, band, verbose)
    elif (ndim == 3) and f32 and WENO and regular:
        out = reinit_3D_WENO_float32.reinit(arr, dt, it, band, verbose)
    elif (ndim == 3) and f32 and UW and subcell:
        out = subcell_3D_UW_float32.reinit(arr, dt, it, band, verbose)
    elif (ndim == 3) and f32 and UW and regular:
        out = reinit_3D_UW_float32.reinit(arr, dt, it, band, verbose)
    elif (ndim == 3) and f64 and WENO and subcell:
        out = subcell_3D_WENO_float64.reinit(arr, dt, it, band, verbose)
    elif (ndim == 3) and f64 and WENO and regular:
        out = reinit_3D_WENO_float64.reinit(arr, dt, it, band, verbose)
    elif (ndim == 3) and f64 and UW and subcell:
        out = subcell_3D_UW_float64.reinit(arr, dt, it, band, verbose)
    elif (ndim == 3) and f64 and UW and regular:
        out = reinit_3D_UW_float64.reinit(arr, dt, it, band, verbose)
    
    return out


# ==============================================================
# def subcell(phi, dt=0.25, max_it=25):
#     import numpy as np
#     from im3D.sdf.f_reinit_3D_UW import reinit
#     # ==============================================================
#     phi = np.require(phi, dtype=np.float32, requirements=('C', 'A'))
#     phi = reinit(phi, dt, max_it, 5.0, 1)
#     return phi

# ==============================================================
# def subcell_WENO(phi, dt=0.25, max_it=25):
#     import numpy as np
#     from im3D.sdf.f_reinit_3D_WENO import reinit
#     # ==============================================================
#     phi = np.require(phi, dtype=np.float32, requirements=('C', 'A'))
#     phi = reinit(phi, dt, max_it, 5.0, 1)
#     return phi

# def reinit(phi, dt=0.25, tol=0.25, band=None, verbose=False, max_it=-1):
#     """
#     INPUTS
#     ======
#         phi: 2D or 3D numpy array, required
#             The initial conditions for creating the signed 
#             distance function (SDF)
#         dt: float, optional, default=0.25
#             Timestep for SDF calculation
#         tol: float, optional, default=0.2
#             Tolerance for how close the gradient should be to 1
#         band: float, optional, default=5.0
#             How far to extend the SDF
#         verbose: int, optional, default=1
#             Flag for how much information should be printed
#             at every iteration.
#             0 -> No print out
#             1 -> Print iteration number and number of 'active'
#                  voxels
#             2 -> Same as 1 plus max |phi_t| and max |phi|
#                  NB: This is much slower than 0 or 1
#     
#     OUTPUTS
#     =======
#         phi: 2D or 3D numpy array
#              The sigend distance function
#     NOTES
#     =====
#         
#     """
#     import numpy as np
#     #
#     from im3D.sdf import reinit_2D
#     from im3D.sdf import reinit_3D
#     # ==============================================================
#     ndim = phi.ndim
#     dtype = phi.dtype
#     if ndim not in [2, 3]:
#         raise ValueError('This function only works for 2D or 3D arrays')
#     if dtype not in [np.float32, np.float64]:
#         raise ValueError('This function only works for 32-bit and 64-bit values')
#     # 
#     phi = np.require(phi, requirements=('C_CONTIGUOUS', 'ALIGNED'))
#     verbose = np.intc(verbose)
#     use_weno = np.intc(use_weno)
#     if band == None:
#         band = np.size(phi)
#     #
#     if   (phi.ndim == 2) and (np.dtype == np.float32):
#         phi = reinit_2D.reinit(phi, dt, tol, band, verbose, max_it, use_weno)
#     elif (phi.ndim == 2) and (np.dtype == np.float64):
#         phi = reinit_2D.reinit(phi, dt, tol, band, verbose, max_it, use_weno)
#     elif (phi.ndim == 3) and (np.dtype == np.float32):
#         phi = reinit_3D.reinit(phi, dt, tol, band, verbose, max_it, use_weno)
#     elif (phi.ndim == 3) and (np.dtype == np.float64):
#         phi = reinit_3D.reinit(phi, dt, tol, band, verbose, max_it, use_weno)
#     #
#     return phi
# # ==============================================================
