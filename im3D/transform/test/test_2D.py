#! /opt/local/bin/py27-MacPorts -i
from DataIO import read_tiff
from transform import rotate, translate
import numpy as np
import matplotlib.pyplot as plot

im = read_tiff('/Users/johngibbs/Desktop/high-SNR.tif')
im = im[312:-312, 312:-312]
plot.imshow(rotate(im, 90))


dX = 0
dY = 1
plot.imshow(translate(im, [dX,dY]))

