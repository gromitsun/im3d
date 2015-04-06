from __future__ import absolute_import
#=== 2D non-subcell ============================================================
from im3D.sdf.inplace import reinit_2D_UW_float32
from im3D.sdf.inplace import reinit_2D_UW_float64
from im3D.sdf.inplace import reinit_2D_WENO_float32
from im3D.sdf.inplace import reinit_2D_WENO_float64
#=== 3D non-subcell ============================================================
from im3D.sdf.inplace import reinit_3D_UW_float32
from im3D.sdf.inplace import reinit_3D_UW_float64
from im3D.sdf.inplace import reinit_3D_WENO_float32
from im3D.sdf.inplace import reinit_3D_WENO_float64
#=== 2D subcell ================================================================
from im3D.sdf.inplace import subcell_2D_UW_float32
from im3D.sdf.inplace import subcell_2D_UW_float64
from im3D.sdf.inplace import subcell_2D_WENO_float32
from im3D.sdf.inplace import subcell_2D_WENO_float64
#=== 3D subcell ================================================================
from im3D.sdf.inplace import subcell_3D_UW_float32
from im3D.sdf.inplace import subcell_3D_UW_float64
from im3D.sdf.inplace import subcell_3D_WENO_float32
from im3D.sdf.inplace import subcell_3D_WENO_float64


def reinit(arr, out=None, dt=0.40, niter=25, subcell=True, WENO=True, verbose=0, band=3.0):
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
    if niter < 0:
        raise ValueError("The number of iterations must be positive")
    if verbose == 1:
        verbose == int(1)
    else:
        verbose == int(0)
    
    # arr = np.require(arr, dtype=dtype, requirements=('C_contiguous',))

    if out is None:
        out = np.empty_like(arr, dtype=dtype)
        ret = True
    else:
        ret = False
    
    if (ndim == 2) and f32 and WENO and subcell:
        subcell_2D_WENO_float32.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 2) and f32 and WENO and regular:
        reinit_2D_WENO_float32.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 2) and f32 and UW and subcell:
        subcell_2D_UW_float32.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 2) and f32 and UW and regular:
        reinit_2D_UW_float32.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 2) and f64 and WENO and subcell:
        subcell_2D_WENO_float64.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 2) and f64 and WENO and regular:
        reinit_2D_WENO_float64.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 2) and f64 and UW and subcell:
        subcell_2D_UW_float64.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 2) and f64 and UW and regular:
        reinit_2D_UW_float64.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 3) and f32 and WENO and subcell:
        subcell_3D_WENO_float32.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 3) and f32 and WENO and regular:
        reinit_3D_WENO_float32.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 3) and f32 and UW and subcell:
        subcell_3D_UW_float32.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 3) and f32 and UW and regular:
        reinit_3D_UW_float32.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 3) and f64 and WENO and subcell:
        subcell_3D_WENO_float64.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 3) and f64 and WENO and regular:
        reinit_3D_WENO_float64.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 3) and f64 and UW and subcell:
        subcell_3D_UW_float64.reinit(arr, out, dt, niter, band, verbose)
    elif (ndim == 3) and f64 and UW and regular:
        reinit_3D_UW_float64.reinit(arr, out, dt, niter, band, verbose)

    if ret:
        return out