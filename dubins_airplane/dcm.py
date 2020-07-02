"""
Created on Thu Jul  2 15:51:40 2020

@author: Kostas Alexis (konstantinos.alexis@mavt.ethz.ch)
"""
import numpy as np

def roty(theta=None):
    # Rotation around y
    R = np.array([  [np.cos(theta),    0,      np.sin(theta)],
                    [0,                1,      0],
                    [-np.sin(theta),   0,      np.cos(theta)]])
    return R


def rotz(theta=None):
    # Rotation around z
    R = np.array([  [np.cos(theta),    -np.sin(theta),    0],
                    [np.sin(theta),    np.cos(theta),     0],
                    [0,                0,                 1]])
    return R