def grad(arr, order=1):
    """
    arr --- Numpy array; required
            The array on which to compute the gradient
    
    order - int; optional, default=1
            Order of the derivative.  First (order=1) and second
            (order=2) order derivatives are implemented
    """
    import numpy as np
    #
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    ndim = np.ndim(arr)
    if ndim == 2:
        from im3D.gradient import grad_2D
        #
        if order == 1:
            return grad_2D.order_1(arr)
        elif order == 2:
            return grad_2D.order_2(arr)
        else:
            raise ValueError('Only order=1 and order=2 are allowed')
    elif ndim == 3:
        from im3D.gradient import grad_3D
        #
        if order == 1:
            return grad_3D.order_1(arr)
        elif order == 2:
            return grad_3D.order_2(arr)
        else:
            raise ValueError('Only order=1 and order=2 are allowed')
    
