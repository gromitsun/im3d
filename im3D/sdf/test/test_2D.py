#! /opt/local/bin/py27-MacPorts -i
import im3D.sdf
import numpy as np
import matplotlib.pyplot as plot

d1 = 100
d2 = 35
a=np.ones((d1,d1))
a[d2:-d2,d2:-d2] = -1

b = im3D.sdf.reinit(a, band=50, verbose=0)

plot.imshow(b)
plot.contour(b, levels=np.linspace(-10,10,5))
