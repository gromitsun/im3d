cimport rotate_2D
cimport rotate_3D
cimport translate_2D
cimport translate_3D
# ==============================================================
def rotate(arr, theta, ctr=None):
    """
    INPUTS
    ======
        arr:   2D or 3D numpy array, required
               The array to be rotated
        theta: Float, required
               The amount of rotation in degrees
        ctr:   2D or 3D numpy array, optional
               Center of rotation
    
    OUTPUTS
    =======
        arr:   2D or 3D numpy array
               The rotated array
    NOTES
    =====
        
    """
    import numpy as np
    #
    if ctr == None:
        ctr = np.zeros((arr.ndim))
        ctr[...] = arr.shape
        ctr = (ctr-1.0)/2.0
    #
    arr = np.require(arr, dtype=np.float64, requirements=('C', 'A'))
    rot = np.array(theta, dtype=np.float64, ndmin=1)
    ctr = np.array(ctr, dtype=np.float64)
    rot *= np.pi/180.0
    #
    if arr.ndim != ctr.size:
        print("'ctr' must have exactly one value for each axis in arr")
        return
    #
    if arr.ndim == 2:
        z_rot, = rot[...]
        x_ctr, y_ctr = ctr[...]
        #
        arr2 = rotate_2D.rotate(arr, z_rot, x_ctr, y_ctr)
    elif arr.ndim == 3:
        x_rot, y_rot, z_rot = rot[...]
        x_ctr, y_ctr, z_ctr = ctr[...]
        #
        arr2 = rotate_3D.rotate(arr, x_rot, y_rot, z_rot, x_ctr, y_ctr, z_ctr)
    #
    return np.asarray(arr2)
# ==============================================================
def translate(arr, disp):
    """
    INPUTS
    ======
        arr:   2D or 3D numpy array, required
               The array to be rotated
        disp:  Float, list or np array, required
               The amount of displacement in each direction
    
    OUTPUTS
    =======
        arr:   2D or 3D numpy array
               The rotated array
    NOTES
    =====
        
    """
    import numpy as np
    #
    arr = np.require(arr, dtype=np.float64, requirements=('C', 'A'))
    disp = np.array(disp, dtype=np.float64, ndmin=1)
    #
    if arr.ndim == 2:
        dX, dY, = disp[...]
        arr2 = translate_2D.translate(arr, dX, dY)
    elif arr.ndim == 3:
        dX, dY, dZ = disp[...]
        arr2 = translate_3D.translate(arr, dX, dY, dZ)
    #
    return np.asarray(arr2)
# ==============================================================
