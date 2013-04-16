#! /opt/local/bin/py27-MacPorts -i
from DataIO import *
from transform import rotate, translate
import numpy as np
import matplotlib.pyplot as plot
import pstats, cProfile

data_1 = np.load('/scratch/Al-24Cu_as-cast_ramp.npy')

x_rot = 10.0
y_rot = 0.0
z_rot = 10.0
rot = [x_rot,y_rot,z_rot]
data_2 = rotate(data_1,rot)
plot.imshow(data_2[50,:,:])

x_rot = 10.0
y_rot = 10.0
z_rot = 10.0
rot = [x_rot,y_rot,z_rot]
data_2 = rotate(data_1,rot)
plot.imshow(data_2[:,50,:])

x_rot = 0.0
y_rot = 0.0
z_rot = 10.0
rot = [x_rot,y_rot,z_rot]
data_2 = rotate(data_1,rot)
plot.imshow(data_2[:,:,50])


data_2 = translate(data_1, [10,10,0])
plot.imshow(data_2[...,200])
