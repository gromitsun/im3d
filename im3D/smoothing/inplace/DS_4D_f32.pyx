#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from cython.parallel cimport prange
# ==============================================================
cpdef void isotropic(float[:,:,:,:] in_arr, float[:,:,:,:] out, int niter=10, float dt=0.25,
              float D0=1.0, float D1=1.0, float D2=1.0, float D3=1.0, int bc_type=1):
    # ==========================================================
    # Variables:
    # a# is an axis
    # n# is the number of elements in that axis
    # D# is the diffusivity along that axis
    # g# is the gradient (second derivative) along that axis
    cdef int  i
    cdef ssize_t  a0, n0=in_arr.shape[0]+2
    cdef ssize_t  a1, n1=in_arr.shape[1]+2
    cdef ssize_t  a2, n2=in_arr.shape[2]+2
    cdef ssize_t  a3, n3=in_arr.shape[3]+2
    cdef float  g0, g1, g2, g3
    # Arrays:
    cdef float[:,:,:,:] out1   = np.zeros((n0,n1,n2,n3), dtype=np.float32)
    cdef float[:,:,:,:] dA_dt = np.zeros((n0,n1,n2,n3), dtype=np.float32)
    # Fill in values for out; central values with input array
    # and perimeter values using BCs:
    out1[1:-1, 1:-1, 1:-1, 1:-1] = in_arr.copy()
    mirror_BC(out1, 1)
    # ==========================================================
    with nogil:
        for i in range(niter):
            # compute dA_dt:
            for a0 in prange(1, n0-1):
                for a1 in range(1, n1-1):
                    for a2 in range(1, n2-1):
                        for a3 in range(1, n3-1):
                            g0 = (+1.0*out1[a0-1, a1, a2, a3] \
                                  -2.0*out1[a0+0, a1, a2, a3] \
                                  +1.0*out1[a0+1, a1, a2, a3])/1.0
                            g1 = (+1.0*out1[a0, a1-1, a2, a3] \
                                  -2.0*out1[a0, a1+0, a2, a3] \
                                  +1.0*out1[a0, a1+1, a2, a3])/1.0
                            g2 = (+1.0*out1[a0, a1, a2-1, a3] \
                                  -2.0*out1[a0, a1, a2+0, a3] \
                                  +1.0*out1[a0, a1, a2+1, a3])/1.0
                            g3 = (+1.0*out1[a0, a1, a2, a3-1] \
                                  -2.0*out1[a0, a1, a2, a3+0] \
                                  +1.0*out1[a0, a1, a2, a3+1])/1.0
                            #
                            dA_dt[a0,a1,a2,a3] = D0*g0 + D1*g1 + D2*g2 + D3*g3
                        # end a3 for loop
                    # end a2 for loop
                # end a1 for loop
            # end a0 for loop
            # ======================================================
            # Update out:
            for a0 in prange(1, n0-1):
                for a1 in range(1, n1-1):
                    for a2 in range(1, n2-1):
                        for a3 in range(1, n3-1):
                            out1[a0,a1,a2,a3] = out1[a0,a1,a2,a3] + dt*dA_dt[a0,a1,a2,a3]
                        # end a3 for loop
                    # end a2 for loop
                # end a1 for loop
            # end a0 for loop
            # ======================================================
            # Apply boundary conditions:
            if bc_type==1:
                mirror_BC(out1, 1)
            else:
                fixed_BC(out1)
        # end iteration for loop
    # end nogil
    # ==========================================================
    # Return only the central portion of the smoothed array
    out[:, :, :, :] = out1[1:-1, 1:-1, 1:-1, 1:-1]

# ==============================================================
cdef inline void mirror_BC(float[:,:,:,:] arr, ssize_t bdry) nogil:
    # ==========================================================
    # Variables:
    cdef int  i
    cdef ssize_t  a0, n0=arr.shape[0]
    cdef ssize_t  a1, n1=arr.shape[1]
    cdef ssize_t  a2, n2=arr.shape[2]
    cdef ssize_t  a3, n3=arr.shape[3]
    cdef ssize_t  min_a0=0, max_a0=n0-1
    cdef ssize_t  min_a1=0, max_a1=n1-1
    cdef ssize_t  min_a2=0, max_a2=n2-1
    cdef ssize_t  min_a3=0, max_a3=n3-1
    # ==========================================================
    for i in range(bdry):
        # 
        # a0 BCs:
        for a1 in prange(n1):
            for a2 in range(n2):
                for a3 in range(n3):
                    arr[min_a0+i, a1, a2, a3] = arr[min_a0+bdry, a1, a2, a3]
                    arr[max_a0-i, a1, a2, a3] = arr[max_a0-bdry, a1, a2, a3]
                # end a3 for loop
            # end a2 for loop
        # end a1 for loop
        # 
        # a1 BCs:
        for a0 in prange(n0):
            for a2 in range(n2):
                for a3 in range(n3):
                    arr[a0, min_a1+i, a2, a3] = arr[a0, min_a1+bdry, a2, a3]
                    arr[a0, max_a1-i, a2, a3] = arr[a0, max_a1-bdry, a2, a3]
                # end a3 for loop
            # end a2 for loop
        # end a0 for loop
        # 
        # a2 BCs:
        for a0 in prange(n0):
            for a1 in range(n1):
                for a3 in range(n3):
                    arr[a0, a1, min_a2+i, a3] = arr[a0, a1, min_a2+bdry, a3]
                    arr[a0, a1, max_a2-i, a3] = arr[a0, a1, max_a2-bdry, a3]
                # end a3 for loop
            # end a1 for loop
        # end a0 for loop
        # 
        # a3 BCs:
        for a0 in prange(n0):
            for a1 in range(n1):
                for a2 in range(n2):
                    arr[a0, a1, a2, min_a3+i] = arr[a0, a1, a2, min_a3+bdry]
                    arr[a0, a1, a2, max_a3-i] = arr[a0, a1, a2, max_a3-bdry]
                # end a2 for loop
            # end a1 for loop
        # end a0 for loop
    # end for boundary width loop
    # return arr
# ==============================================================
cdef inline void fixed_BC(float[:,:,:,:] arr) nogil:
    # ==========================================================
    # Variables:
    cdef int  i
    cdef ssize_t  a0, n0=arr.shape[0]
    cdef ssize_t  a1, n1=arr.shape[1]
    cdef ssize_t  a2, n2=arr.shape[2]
    cdef ssize_t  a3, n3=arr.shape[3]
    cdef ssize_t  min_a0=0, max_a0=n0-1
    cdef ssize_t  min_a1=0, max_a1=n1-1
    cdef ssize_t  min_a2=0, max_a2=n2-1
    cdef ssize_t  min_a3=0, max_a3=n3-1
    # ==========================================================
    # a0 BCs:
    for a1 in prange(n1):
        for a2 in range(n2):
            for a3 in range(n3):
                arr[min_a0+1, a1, a2, a3] = arr[min_a0, a1, a2, a3]
                arr[max_a0-1, a1, a2, a3] = arr[max_a0, a1, a2, a3]
            # end a3 for loop
        # end a2 for loop
    # end a1 for loop
    # 
    # a1 BCs:
    for a0 in prange(n0):
        for a2 in range(n2):
            for a3 in range(n3):
                arr[a0, min_a1+1, a2, a3] = arr[a0, min_a1, a2, a3]
                arr[a0, max_a1-1, a2, a3] = arr[a0, max_a1, a2, a3]
            # end a3 for loop
        # end a2 for loop
    # end a0 for loop
    # 
    # a2 BCs:
    for a0 in prange(n0):
        for a1 in range(n1):
            for a3 in range(n3):
                arr[a0, a1, min_a2+1, a3] = arr[a0, a1, min_a2, a3]
                arr[a0, a1, max_a2-1, a3] = arr[a0, a1, max_a2, a3]
            # end a3 for loop
        # end a1 for loop
    # end a0 for loop
    # 
    # a3 BCs:
    for a0 in prange(n0):
        for a1 in range(n1):
            for a2 in range(n2):
                arr[a0, a1, a2, min_a3+1] = arr[a0, a1, a2, min_a3]
                arr[a0, a1, a2, max_a3-1] = arr[a0, a1, a2, max_a3]
            # end a2 for loop
        # end a1 for loop
    # end a0 for loop
    # return arr
