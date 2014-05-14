#! python -i
import im3D.sdf
import numpy as np
import matplotlib.pyplot as plot

d1 = 100
d2 = 35
a=np.ones((d1,d1))
a[d2:-d2,d2:-d2] = -1

fig_num = 0
for dtype in ('float32', 'float64'):
    for subcell in (False, True):
        for WENO in (False, True):
            fig_num += 1
            print('fig: %i | dtype: %s | subcell: %5s | WENO: %5s' %(fig_num, dtype, subcell, WENO))
            b = im3D.sdf.reinit(a.astype(dtype), dt=0.40, it=25, subcell=subcell, WENO=WENO, verbose=0)
            plot.figure(fig_num)
            plot.imshow(b, vmin=-10, vmax=+10)
            plot.contour(b, levels=np.linspace(-10,10,5), colors='r')

