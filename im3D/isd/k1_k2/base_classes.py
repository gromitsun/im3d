class k1k2_isd:
    
    """
    required data from __init__:
        self.k1_min
        self.k1_max
        self.k2_min
        self.k2_max
        
    USEAGE
    ======
        ...
    
    INPUTS
    ======
        phi - 3D numpy array, required
              This must be a signed distance function
    
        k1 -- 3D numpy array, required
              Principal curvature
              If this array is not given, it will be calculated.
              k1 < k2 is the typical convention
    
        k2 -- 3D numpy array, optional
              Principal curvature
              If this array is not given, it will be calculated
              k1 < k2 is the typical convention
    
        P --- 3D numpy array, optional
              Property to sum
              If this array is not given, ones will be used; i.e. an 
              area ISD will be calculated
    
        eps - float, optional, default=1.0
              Width of the interface
    
        nbins - int, optional, default=200
              Number of bins in each direction
    
        bin_min - float, optional, default=-0.1
    
        bin_max - float, optional, default=+0.1

    OUTPUTS
    =======
        ISD(k1,k2) - 2D numpy array
              The interfacial shape distribution where k1 < k2


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
    
    #def __init__(self):
    #    self.data = make_isd(...)
    
    def plot(self, fig_num=None, scaled=True, cmap=None, 
             vmin=None, vmax=None, label='$P(\\kappa_1,\\kappa_2)$', 
             label_fmt='%.2f'):
        """
        ...
        """
        import matplotlib.pyplot as plot
        # 
        # Make the axis:
        self.ax = self.make_axes(fig_num, scaled)
        self.im = self.draw_isd(self.ax, cmap, vmin, vmax)
        self.cb = self.add_colorbar(self.im, label, label_fmt)
    
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
        ax.set_xlim(self.k1_min, self.k1_max)
        ax.set_ylim(self.k2_min, self.k2_max)
        # 
        # Draw lines
        lines = {'color': 'w', 'linewidth': 4}
        ax.plot((self.k1_min, self.k1_max), (        0.0,         0.0), **lines)
        ax.plot((        0.0,         0.0), (self.k2_min, self.k2_max), **lines)
        ax.plot((self.k1_min, self.k1_max), (self.k2_min, self.k2_max), linestyle='dashed', **lines)
        ax.plot((self.k1_min,         0.0), (self.k2_max,         0.0), linestyle='dashed', **lines)
        # 
        # Add labels
        if scaled == True:
            ax.set_xlabel('$\\kappa_1/S_v$', fontsize=30)
            ax.set_ylabel('$\\kappa_2/S_v$', fontsize=30)
        else:
            ax.set_xlabel('$\\kappa_1$', fontsize=30)
            ax.set_ylabel('$\\kappa_2$', fontsize=30)
        # 
        # Force draw the figure and axes
        plot.draw()
        # 
        # Return the axis handle for later use
        return ax
        
    
    def draw_isd(self, ax, cmap=None, vmin=None, vmax=None, **kwargs):
        """
    
        """
        import matplotlib.pyplot as plot
        # 
        if cmap == None: 
            cmap = plot.cm.spectral
        # 
        aspect = (self.k1_max - self.k1_min) / (self.k2_max - self.k2_min)
        ax_limits = (self.k1_min, self.k1_max, 
                     self.k2_min, self.k2_max)
        im = ax.imshow(self.data.T, extent=ax_limits, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, origin='lower', **kwargs)
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
        ax_limits = (self.k1_min, self.k1_max, 
                     self.k2_min, self.k2_max)
        ct = plot.contour(isd.T, extent=ax_limits, levels=levels)
        #
        plot.draw()
        #
        return ct
        
