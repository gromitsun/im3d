from .base_classes import SC_isd

class isd(SC_isd):
    
    """
    USEAGE
    ======
        ...
    
    INPUTS
    ======
        phi - 3D numpy array, required
              This must be a signed distance function
    
        k1 -- 3D numpy array, required
              ...
    
        k2 -- 3D numpy array, optional
              ...
    
        P --- 3D numpy array, optional
              Property to sum
              If this array is not given, ones will be used; i.e. an 
              area ISD will be calculated
    
        eps - float, optional, default=1.0
              Width of the interface
    
        nbins - int, optional, default=200
              Number of bins in each direction
    
        C_max - float, optional, default=+0.1
              Max bin value in the 'C' direction.  The min value
              of C will always be 0.0 and S has limits of -1.0
              to +1.0

    OUTPUTS
    =======
        ISD(S,C) - 2D numpy array
              The interfacial shape distribution
        S --- 3D numpy array, required
              'Shape' curvature
                   1      2
              S = --- + ---- * atan2(k1, k2)
                   2     pi
    
        C --- 3D numpy array, optional
              'Curvedness' curvature
              C = sqrt(k1**2 + k2**2)
    


    EXAMPLE
    =======


    DETAILS
    =======
        - The ISD can be thought of as many surface integrals for a
          restricted range of curvatures. So for a particular ISD
          bin at k1=k1* and k2=k2*: 
      
                           /
            ISD(k1*,k2*) = | 1 dS*
                           /
        
            where S* is all surface patches with 
            (k1* - bin_size/2) < k1 < (k1* + bin_size/2) and 
            (k2* - bin_size/2) < k2 < (k2* + bin_size/2)
        - The surface integral of 1 will return the total surface
          area of all patches matching that particular k1*, k2*
        - To find the ISD of a property (e.g. interfacial velocity), P(x) is the property instead of 1

    IMPLEMENTATION
    ==============
        - Since it is much easier to do a volume integral than a
          surface integral, the following method that is presented
          in ... is used:  
          - If phi is a signed distance function, then the surface
            integral of property P(x) can be approximated by a 
            volume integral:
      
              /           /
              | P(x) dS = | D(phi(x)) P(x) dV
              /           /
          
              where D(phi(x)) is a smeared out delta function with
              width of eps and has the following form:
          
                         1   /     /phi*pi\\
              D(phi) = -----*|1+cos|------|| for -eps < phi < +eps
                       2*eps \     \ eps  //
              and 0 elsewhere
    """
    
    def __init__(self, phi, k1, k2, P=None, nbins=200, C_max=+10.0, eps=1.0):
        """
        add some info here about what is done in __init__
        """
        import numpy as np
        from .global_SC_isd import calculate_isd
        # 
        # Create an array of ones for P if nothing is provided:
        if P==None:
            P = np.ones_like(phi)
        # 
        # Ensure the arrays are the format that Cython expects:
        phi = np.require(phi, dtype=np.float64, requirements='C')
        k1  = np.require(k1,  dtype=np.float64, requirements='C')
        k2  = np.require(k2,  dtype=np.float64, requirements='C')
        P   = np.require(P,   dtype=np.float64, requirements='C')
        # 
        # Calculate the ISD:
        self.data = calculate_isd(phi, k1, k2, P, nbins, C_max, eps)
        # 
        # Save misc values for later use:
        self.S_min = -1.0
        self.S_max = +1.0
        self.C_min = +0.0
        self.C_max = C_max
        self.nbins = nbins
        self.eps = eps
