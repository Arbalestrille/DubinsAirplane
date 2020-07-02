"""
This is the file to execute examples of the Dubins Airplane mode
that supports 16 cases of possible trajectories

@author: Kostas Alexis (konstantinos.alexis@mavt.ethz.ch)
"""

import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
from dubins_airplane.path import dubins

class Vehicle(object):
    """
        Vehicle Parameters
    """

    def __init__(self, bank_max, gamma_max):
        self.bank_max = np.deg2rad(bank_max)
        self.gamma_max = np.deg2rad(gamma_max)

plane = Vehicle(45, 30)

# Example set for the 16 cases Dubins Airplane paths
dubins_case = 1

if dubins_case > 16:
    print('Not a case')
elif dubins_case == 1: 
    Path_Type = 'short climb RSR'
    start_node = np.array([0, 0, -100, 0, 15])
    end_node = np.array([0, 200, -125, 270, 15])
elif dubins_case == 2: 
    Path_Type = 'short climb RSL'
    start_node = np.array([0, 0, -100, -70, 15])
    end_node = np.array([100, 100, -125, -70, 15])
elif dubins_case == 3:
    Path_Type = 'short climb LSR'
    start_node = np.array([0, 0, -100, 70, 15])
    end_node = np.array([100, -100, -125, 70, 15])
elif dubins_case == 4:
    Path_Type = 'short climb LSL'
    start_node = np.array([0, 0, -100, 70, 15])
    end_node = np.array([100, -100, -125, -135, 15])
elif dubins_case == 5:
    Path_Type = 'long climb RSR'
    start_node = np.array([0, 0, -100, 0, 15])
    end_node = np.array([0, 200, -250, 270, 15])
elif dubins_case == 6:
    Path_Type = 'long climb RSL'
    start_node = np.array([0, 0, -100, -70, 15])
    end_node = np.array([100, 100, -350, -70, 15])
elif dubins_case == 7:
    Path_Type = 'long climb LSR'
    start_node = np.array([0, 0, -350, 70, 15])
    end_node = np.array([100, -100, -100, 70, 15])
elif dubins_case == 8:
    Path_Type = 'long climb LSL'
    start_node = np.array([0, 0, -350, 70, 15])
    end_node = np.array([100, -100, -100, -135, 15])
elif dubins_case == 9: 
    Path_Type = 'intermediate climb RLSR (climb at beginning)'
    start_node = np.array([0, 0, -100, 0, 15])
    end_node = np.array([0, 200, -200, 270, 15])
elif dubins_case == 10: 
    Path_Type = 'intermediate climb RLSL (climb at beginning)'
    start_node = np.array([0, 0, -100, 0, 15])
    end_node = np.array([100, 100, -200, -90, 15])
elif dubins_case == 11:
    Path_Type = 'intermediate climb LRSR (climb at beginning)'
    start_node = np.array([0, 0, -100, 0, 15])
    end_node = np.array([100, -100, -200, 90, 15])
elif dubins_case == 12: 
    Path_Type = 'intermediate climb LRSL (climb at beginning)'
    start_node = np.array([0, 0, -100, 0, 15])
    end_node = np.array([100, -100, -200, -90, 15])
elif dubins_case == 13:  
    Path_Type = 'intermediate climb RSLR (descend at end)'
    start_node = np.array([0, 0, -200, 0, 15])
    end_node = np.array([100, 100, -100, 90, 15])
elif dubins_case == 14:
    Path_Type = 'intermediate climb RSRL (descend at end)'
    start_node = np.array([0, 0, -200, 0, 15])
    end_node = np.array([100, 100, -100, -90, 15])
elif dubins_case == 15:
    Path_Type = 'intermediate climb LSLR (descend at end)'
    start_node = np.array([0, 0, -200, 70, 15])
    end_node = np.array([100, -100, -100, 90, 15])
elif dubins_case == 16:
    Path_Type = 'intermediate climb LSRL (descend at end)'
    start_node = np.array([0, 0, -150, 0, 15])
    end_node = np.array([100, -100, -100, -90, 15])
elif dubins_case == 0:
    Path_Type = 'for fixing errors'
    start_node = np.array([0, 0, 0, 0, 15])
    end_node = np.array([40, -140, 100, 3.84, 15])

path = dubins(start_node, end_node, plane)               
    
pylab.ion()
fig = pylab.figure()
ax = Axes3D(fig)
ax.invert_xaxis()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.invert_zaxis()
ax.set_zlabel('z (m)')
ax.set_title(Path_Type)
ax.plot(path.X, path.Y, path.Z)
fig.show()

