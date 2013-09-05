class SC_isd:
    
    """
    required data from __init__:
        self.S_min
        self.S_max
        self.C_min
        self.C_max
        
    """
    #def __init__(self):
    #    self.data = make_isd(...)
    
    def plot(self, fig_num=None, scaled=True, cmap=None, 
             vmin=None, vmax=None, label='$P(S,C)$', label_fmt='%.2f'):
        """
        ...
        """
        import matplotlib.pyplot as plot
        # 
        # Make the axis:
        self.ax = self.make_axes(fig_num, scaled)
        self.im = self.plot_isd(self.ax, cmap, vmin, vmax)
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
        aspect = (self.S_max - self.S_min) / (self.C_max - self.C_min)
        ax_limits = (self.S_min, self.S_max, 
                     self.C_min, self.C_max)
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
        ax_limits = (self.S_min, self.S_max, 
                     self.C_min, self.C_max)
        ct = plot.contour(isd.T, extent=ax_limits, levels=levels)
        #
        plot.draw()
        #
        return ct
        
