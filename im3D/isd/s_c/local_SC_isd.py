class __general_SC_isd__:
    
    #def __init__(self):
    #    self.data = make_isd(...)
    
    def plot(self, fig_num=None, scaled=True, cmap=None, 
             vmin=None, vmax=None, label=None, label_fmt='%.2f'):
        """
        ...
        """
        import matplotlib.pyplot as plot
        # 
        # Make the axis:
        ax = self.make_axes(fig_num, scaled)
        im = self.plot_isd(ax, cmap, vmin, vmax)
        cb = self.add_colorbar(im, label, label_fmt)
    
    def make_axes(self, fig_num=None, scaled=True):
        """
        ...
        """
        import matplotlib.pyplot as plot
        # 
        # Make the figure
        fig = plot.figure(fig_num)
        fig.clf()
        # 
        # Setup the axis as a subplot
        ax = plot.subplot2grid(shape=[1,1], loc=[0,0], rowspan=1, colspan=1)
        # 
        # Set axis limits
        ax.set_xlim(self.S_min, self.S_max)
        ax.set_ylim(self.C_min, self.C_max)
        # 
        # Draw lines
        lines = {'color': 'w', 'linestyle': '--', 'linewidth': 4}
        ax.plot((-0.5, -0.5), (self.C_min, self.C_max), **lines)
        ax.plot((+0.0, +0.0), (self.C_min, self.C_max), **lines)
        ax.plot((+0.5, +0.5), (self.C_min, self.C_max), **lines)
        # 
        # Add labels
        if scaled == True:
            ax.set_xlabel('$S$', fontsize=30)
            ax.set_ylabel('$C/S_v$', fontsize=30)
        else:
            ax.set_xlabel('$S$', fontsize=30)
            ax.set_ylabel('$C$', fontsize=30)
        # 
        # Force draw the figure and axes
        plot.draw()
        # 
        # Return the axis handle for later use
        return ax
        
    
    def plot_isd(self, ax, cmap=None, vmin=None, vmax=None, **kwargs):
        """
    
        """
        import matplotlib.pyplot as plot
        # 
        if cmap == None: 
            cmap = plot.cm.spectral
        # 
        ax_limits = (ax.get_xlim()[0], ax.get_xlim()[1], 
                     ax.get_ylim()[0], ax.get_ylim()[1])
        im = ax.imshow(self.data.T, extent=ax_limits, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', **kwargs)
        # 
        plot.draw()
        return im
        
    
    def add_colorbar(self, im, label=None, format='%.2f'):
        """
  
        """
        import matplotlib.pyplot as plot
        # 
        cbar = plot.colorbar(im, orientation='vertical', format=format)
        if label != None:
            cbar.set_label(label, fontsize=30)
        # 
        return cbar
        
    
    def add_contour(self, ax, isd, levels=None):
        """
    
        """
        import matplotlib.pyplot as plot
        # 
        ax_limits = (ax.get_xlim()[0], ax.get_xlim()[1], 
                     ax.get_ylim()[0], ax.get_ylim()[1])
        ct = plot.contour(isd.T, extent=ax_limits, levels=levels)
        #
        plot.draw()
        #
        return ct
        


class isd(__general_SC_isd__):
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
        add some info here
        """
        import numpy as np
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
        self.data = self.make_isd(phi, k1, k2, P, nbins, C_max, eps)
        # 
        # Save misc values for later use:
        self.S_min = -1.0
        self.S_max = +1.0
        self.C_min = +0.0
        self.C_max = C_max
        self.eps = eps
        self.nbins = nbins
    
    def make_isd(*args, **kwargs):
        import numpy as np
        nbins=200
        return np.random.random(size=(nbins, nbins))
#     def make_isd(self, double[:,:,:] phi, double[:,:,:] k1, double[:,:,:] k2
#                  double[:,:,:] P, int nbins, double C_max, double eps):
#         # Typed values:
#         cdef ssize_t  x, y, z, nx, ny, nz
#         cdef ssize_t  S_bin, C_bin
#         cdef double  C, C_min=+0.0
#         cdef double  S, S_min=-1.0, S_max=+1.0
#         cdef double  phi_x, phi_y, phi_z, grad
#         cdef double  delta, area, pi=3.141592653589793
#         # Typed arrays:
#         cdef double[:,:] ISD=np.zeros((nbins,nbins), dtype=np.float64)
#         #
#         nx, ny, nz=phi.shape
#         # 
#         with nogil:
#           for x in range(1,nx-1):
#             for y in range(1,ny-1):
#               for z in range(1,nz-1):
#                 if fabs(cy_phi[x,y,z]) <= eps:
#                   phi_x = (cy_phi[x+1,y,z] - cy_phi[x-1,y,z])/2.0
#                   phi_y = (cy_phi[x,y+1,z] - cy_phi[x,y-1,z])/2.0
#                   phi_z = (cy_phi[x,y,z+1] - cy_phi[x,y,z-1])/2.0
#                   #
#                   grad = sqrt(phi_x**2 + phi_y**2 + phi_z**2)
#                   #
#                   delta = 1./(2.0*eps) * (1.0 + cos(cy_phi[x,y,z]*pi/eps)) 
#                   area = delta * grad
#                   #
#                   C = sqrt(cy_k1[x,y,z]**2 + cy_k2[x,y,z]**2)
#                   S = 0.5 + 2/pi * atan2(cy_k1[x,y,z], cy_k2[x,y,z])
#                   #
#                   S_bin = val2bin(S, S_min, S_max, nbins)
#                   C_bin = val2bin(C, C_min, C_max, nbins)
#                   #
#                   if (S_bin >= 0) and (S_bin < nbins) and \
#                      (C_bin >= 0) and (C_bin < nbins):
#                     ISD[S_bin, C_bin] += area*cy_P[x,y,z]
#               # end z for loop
#             # end y for loop
#           # end x for loop
#         # end nogil
#         return np.asarray(ISD)
# 
#         
