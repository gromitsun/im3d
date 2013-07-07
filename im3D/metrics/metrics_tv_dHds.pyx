#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange
# ==============================================================
def tv_dHds(phi):
    import numpy as np
    import curv
    import grad
    
    # Delta function:
    eps = 2.5
    D = 1.0/(2.0*eps) * (1.0 + np.cos(phi*np.pi/eps))
    w = np.where(np.abs(phi) >= eps)
    D[w] = 0.0
    
    ndim = phi.ndim
    if ndim == 2:
        # Normal vector:
        mag_grad = grad.grad(phi, weno=False)
        N_x = grad.grad_x(phi, weno=False)/mag_grad
        N_y = grad.grad_y(phi, weno=False)/mag_grad
        # Tangent vector:
        T_x = + N_y
        T_y = - N_x
        # Mean curvature:
        H = curv.H(phi, reinit=False)
        H_x = grad.grad_x(H, weno=False)
        H_y = grad.grad_y(H, weno=False)
        # Curvature surface gradients:
        Hs_x = H_x * T_x
        Hs_y = H_y * T_y
        Hs = Hs_x + Hs_y
    elif ndim == 3:
        # Normal vector:
        mag_grad = grad.grad(phi, weno=False)
        N_x = grad.grad_x(phi, weno=False)/mag_grad
        N_y = grad.grad_y(phi, weno=False)/mag_grad
        N_z = grad.grad_z(phi, weno=False)/mag_grad
        # Tangent vector:
        T_x = + N_y - N_z
        T_y = - N_x + N_z
        T_z = + N_x - N_y
        # Mean curvature:
        H = curv.H(phi, reinit=False)
        H_x = grad.grad_x(H, weno=False)
        H_y = grad.grad_y(H, weno=False)
        H_z = grad.grad_z(H, weno=False)
        # Curvature surface gradients:
        Hs_x = H_x * T_x
        Hs_y = H_y * T_y
        Hs_z = H_z * T_z
        Hs = Hs_x + Hs_y + Hs_z
            
    tv_dH = np.nansum(D * Hs**2)
    area  = np.nansum(D)
    
    return np.sqrt(tv_dH/area)

