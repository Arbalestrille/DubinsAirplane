"""
The functions here implement the 3D Dubins Airplane model with totally
16 cases of possible trajectories

@author: Kostas Alexis (konstantinos.alexis@mavt.ethz.ch)
"""

import numpy as np
import pandas as pd
from math import tan, sin, cos, atan2, fmod, acos, asin, fabs, atan
from dubins_airplane import dcm
from dubins_airplane import compute

pi = np.pi

def _generate(init_conf, final_conf, Vehicle):
    # Generate the Dubins Airplane path
    
    gamma_max = Vehicle.gamma_max
    bank_max = Vehicle.bank_max
    
    # start
    zs = (init_conf[0:3]).T
    Vairspeed = init_conf[4]
    anglstart = np.deg2rad(init_conf[3])
    
    # end
    ze = (final_conf[0:3]).T
    anglend = np.deg2rad(final_conf[3])
                         
    R_min = compute.MinTurnRadius(Vairspeed, bank_max)    
    # Check if start and end node are too close. Since spiral-spiral-spiral
    # (or curve-curve-curve) paths are not considered, the optimal path may
    # not be found... (see Shkel, Lumelsky, 2001, Classification of the
    # Dubins set, Prop. 5/6). Below is a conservative bound, which seems
    # (by experiments) to assure a unproblematical computation of the
    # dubins path.    
    dist = np.linalg.norm(init_conf[0:2] - final_conf[0:2], ord=2)
    assert dist > 6 * R_min, 'Start and end pose are close together'  
        
    df = dict()
    df['p_s'] = zs
    df['angl_s'] = anglstart
    df['p_e'] = ze
    df['angl_e'] = anglend

    crs = zs + R_min*np.dot(dcm.rotz(pi/2), np.array([cos(anglstart), sin(anglstart), 0]).T)
    cls = zs + R_min*np.dot(dcm.rotz(-pi/2),np.array([cos(anglstart), sin(anglstart), 0]).T)
    cre = ze + R_min*np.dot(dcm.rotz(pi/2),np.array([cos(anglend), sin(anglend), 0]).T)
    cle = ze + R_min*np.dot(dcm.rotz(-pi/2),np.array([cos(anglend), sin(anglend), 0]).T)

    # compute L1, L2, L3, L4
    L1 = compute.RSR(R_min, crs, cre, anglstart, anglend)
    L2 = compute.RSL(R_min, crs, cle, anglstart, anglend)
    L3 = compute.LSR(R_min, cls, cre, anglstart, anglend)
    L4 = compute.LSL(R_min, cls, cle, anglstart, anglend)

    # L is the minimum distance
    L = np.amin(np.array([L1, L2, L3, L4]))
    idx = np.where(np.array([L1,L2,L3,L4])==L)[0][0] + 1

    hdist = -(ze[2] - zs[2])
    if fabs(hdist) <= L*tan(gamma_max):
        gam = atan(hdist/L)
        df['case'] = 1
        df['R'] = R_min
        df['gamma'] = gam
        df['L'] = L/cos(gam)
        df['k_s'] = 0
        df['k_e'] = 0
    elif fabs(hdist) >= (L+2*pi*R_min)*tan(gamma_max):

        k = np.floor( (fabs(hdist)/tan(gamma_max) - L)/(2*pi*R_min))

        if hdist >= 0:

            df['k_s'] = k
            df['k_e'] = 0
        else:
            df['k_s'] = 0
            df['k_e'] = k

        # find optimal turning radius
        R = compute.OptimalRadius(zs, anglstart, ze, anglend, R_min, gamma_max, idx, k, hdist)

        # recompute the centers of spirals and Dubins path length with new R
        crs = zs + R*np.dot(dcm.rotz(pi/2), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
        cls = zs + R*np.dot(dcm.rotz(-pi/2), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
        cre = ze + R*np.dot(dcm.rotz(pi/2), np.array( [cos(anglend), sin(anglend), 0] ).T )
        cle = ze + R*np.dot(dcm.rotz(-pi/2), np.array( [cos(anglend), sin(anglend), 0] ).T )

        if idx == 1:
            L = compute.RSR( R, crs, cre, anglstart, anglend )
        elif idx == 2:
            L = compute.RSL( R, crs, cle, anglstart, anglend )
        elif idx == 3:
            L = compute.LSR( R, cls, cre, anglstart, anglend )
        elif idx == 4:
            L = compute.LSL( R, cls, cle, anglstart, anglend )

        df['case'] = 1
        df['R'] = R
        gam = np.sign( hdist ) * gamma_max
        df['gamma'] = gam
        df['L'] = ( L + 2 * pi * k * R ) / cos( gamma_max )

    else:

        gam = np.sign( hdist ) * gamma_max

        if hdist > 0:
            zi, chii, L, ci, psii = compute.addSpiralBeginning( zs, anglstart, ze, anglend, R_min, gam, idx, hdist )
            df['case'] = 2
        else:
            zi, chii, L, ci, psii = compute.addSpiralEnd( zs, anglstart, ze, anglend, R_min, gam, idx, hdist )
            df['case'] = 3

        df['R'] = R_min
        df['gamma'] = gam
        df['L'] = L / cos( gamma_max )

    e1 = np.array( [1, 0, 0] ).T
    R = df['R']

    if np.isscalar(df['case']):
        pass
    else:
        print( '### Error' )



    if df['case'] == 1: # spiral-line-spiral
        if idx == 1: # right-straight-right
            theta = atan2( cre[1]-crs[1], cre[0]-crs[0])
            dist1 = R*fmod(2*pi+fmod(theta-pi/2,2*pi)-fmod(anglstart-pi/2,2*pi),2*pi) + 2*pi*R*df['k_s']
            dist2 = R*fmod(2*pi+fmod(anglend-pi/2,2*pi)-fmod(theta-pi/2,2*pi),2*pi) + 2*pi*R*df['k_e']
            w1 = crs + df['R']*np.dot(dcm.rotz(theta-pi/2),e1.T).T + np.array([0,0,-dist1*tan(gam)]).T
            w2 = cre + df['R']*np.dot(dcm.rotz(theta-pi/2),e1.T).T - np.array([0,0,-dist2*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2) # direction of line

            df['c_s']   = crs
            df['psi_s'] = anglstart-pi/2
            df['lamda_s'] = 1
            # end spiral
            df['c_e']   = cre-np.array([0,0,-dist2*tan(gam)])
            df['psi_e'] = theta-pi/2
            df['lamda_e'] = 1
            # hyperplane H_s: switch from first spiral to line
            df['w_s']   = w1
            df['q_s']   = q1
            # hyperplane H_l: switch from line to last spiral
            df['w_l']   = w2
            df['q_l']   = q1
            # hyperplane H_e: end of Dubins path
            df['w_e']   = ze
            df['q_e']   = np.dot(dcm.rotz(anglend), np.array([1,0,0]).T)
            
        elif idx == 2: # right-straight-left
            ell = np.linalg.norm(cle[0:2] - crs[0:2],ord=2)
            theta = atan2(cle[1]-crs[1], cle[0]-crs[0])
            theta2 = theta - pi/2 + asin(2*R/ell)
            dist1 = R*fmod(2*pi+fmod(theta2,2*pi)-fmod(anglstart-pi/2,2*pi),2*pi) + 2*pi*R*df['k_s']
            dist2 = R*fmod(2*pi+fmod(theta2+pi,2*pi)-fmod(anglend+pi/2,2*pi),2*pi) + 2*pi*R*df['k_e']
            w1 = crs + R*np.dot(dcm.rotz(theta2), e1.T).T + np.array([0, 0, -dist1*tan(gam)]).T
            w2 = cle + R*np.dot(dcm.rotz(theta2+pi),e1.T).T - np.array([0,0,-dist2*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2)
            # start spiral
            df['c_s']   = crs
            df['psi_s'] = anglstart-pi/2
            df['lamda_s'] = 1
            # end spiral
            df['c_e']   = cle - np.array([0,0,-dist2*tan(gam)]).T
            df['psi_e'] = theta2+pi
            df['lamda_e'] = -1
            # hyperplane H_s: switch from first spiral to line
            df['w_s']   = w1
            df['q_s']   = q1
            # hyperplane H_l: switch from line to end spiral
            df['w_l']   = w2
            df['q_l']   = q1
            # hyperplane H_e: end of Dubins path
            df['w_e'] = ze
            df['q_e']   = np.dot(dcm.rotz(anglend),np.array([1,0,0]).T)
            
        elif idx == 3: # left-straight-right
            ell = np.linalg.norm(cre[0:2]-cls[0:2],ord=2)
            theta = atan2( cre[1]-cls[1],cre[0]-cls[0])
            theta2 = acos(2*R/ell)
            dist1 = R*fmod(2*pi-fmod(theta+theta2,2*pi) + fmod(anglstart+pi/2,2*pi),2*pi) + 2*pi*R*df['k_s']
            dist2 = R*fmod(2*pi-fmod(theta+theta2-pi,2*pi)+fmod(anglend-pi/2,2*pi),2*pi) + 2*pi*R*df['k_e']
            w1 = cls + R*np.dot(dcm.rotz(theta+theta2),e1.T).T + np.array([0, 0, -dist1*tan(gam)]).T
            w2 = cre + R*np.dot(dcm.rotz(-pi+theta+theta2),e1.T).T - np.array([0, 0, -dist2*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2)

            # start spiral
            df['c_s']   = cls
            df['psi_s'] = anglstart+pi/2
            df['lamda_s'] = -1
            # end spiral
            df['c_e']   = cre - np.array([0,0,-dist2*tan(gam)]).T
            df['psi_e'] = fmod(theta+theta2-pi,2*pi)
            df['lamda_e'] = 1
            # hyperplane H_s: switch from first spiral to line
            df['w_s']   = w1
            df['q_s']   = q1
            # hyperplane H_l: switch from line to end spiral
            df['w_l']   = w2
            df['q_l']   = q1
            # hyperplane H_e: end of Dubins path
            df['w_e']   = ze
            df['q_e']   = np.dot(dcm.rotz(anglend), np.array([1, 0, 0]).T)
            
        elif idx ==4: # left-straight-left
            theta = atan2(cle[1] -cls[1], cle[0] - cls[0])
            dist1 = R*fmod(2*pi-fmod(theta+pi/2,2*pi)+fmod(anglstart+pi/2,2*pi),2*pi) + 2*pi*R*df['k_s']
            dist2 = R*fmod(2*pi-fmod(anglend+pi/2,2*pi)+fmod(theta+pi/2,2*pi),2*pi) + 2*pi*R*df['k_e']
            w1 = cls + df['R']*np.dot(dcm.rotz(theta+pi/2),e1.T).T + np.array([0,0,-dist1*tan(gam)]).T
            w2 = cle + df['R']*np.dot(dcm.rotz(theta+pi/2),e1.T).T - np.array([0,0,-dist2*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2)

            # start spiral
            df['c_s']   = cls
            df['psi_s'] = anglstart+pi/2
            df['lamda_s'] = -1
            # end spiral
            df['c_e']   = cle - np.array([0,0,-dist2*tan(gam)]).T
            df['psi_e'] = theta+pi/2
            df['lamda_e'] = -1
            # hyperplane H_s: switch from first spiral to line
            df['w_s']   = w1
            df['q_s']   = q1
            # hyperplane H_l: switch from line to end spiral
            df['w_l']   = w2
            df['q_l']   = q1
            # hyperplane H_e: end of Dubins path
            df['w_e']   = ze
            df['q_e']   = np.dot(dcm.rotz(anglend), np.array([1,0,0]).T)
            
    elif df['case'] == 2:
        if idx == 1: # right-left-straight-right
            # start spiral
            df['c_s'] = crs
            df['psi_s'] = anglstart-pi/2
            df['lamda_s'] = 1
            df['k_s']   = 0
            ell = np.linalg.norm(cre[0:2]-ci[0:2],ord=2)
            theta = atan2(cre[1] - ci[1], cre[0] - ci[0])
            theta2 = acos(2*R/ell)
            dist1 = R_min*psii + R*fmod(2*pi-fmod(theta+theta2,2*pi) + fmod(chii+pi/2,2*pi),2*pi)
            dist2 = R*fmod(2*pi-fmod(theta+theta2-pi,2*pi)+fmod(anglend-pi/2,2*pi),2*pi)
            w1 = ci + R*np.dot(dcm.rotz(theta+theta2),e1.T).T + np.array([0, 0, -dist1*tan(gam)]).T
            w2 = cre + R*np.dot(dcm.rotz(-pi+theta+theta2),e1.T).T - np.array([0, 0, -dist2*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2)
            # intermediate-start spiral
            df['c_si']   = ci + np.array([0, 0, -R_min*psii*tan(gam)]).T
            df['psi_si'] = chii + pi/2
            df['lamda_si'] = -1
            df['k_si']   = 0
            # end spiral
            df['c_e']   = cre - np.array([0,0,-dist2*tan(gam)]).T
            df['psi_e'] = fmod(theta+theta2-pi,2*pi)
            df['lamda_e'] = 1
            df['k_e']   = 0
            # hyperplane H_s: switch from first to second spiral
            df['w_s']   = zi - np.array([0, 0, -psii*R_min*tan(gam)]).T
            df['q_s']   = np.array([cos(chii), sin(chii), 0]).T
            # hyperplane H_si: switch from second spiral to straight line
            df['w_si']  = w1
            df['q_si']  = q1
            # hyperplane H_l: switch from straight-line to end spiral
            df['w_l']   = w2
            df['q_l']   = q1
            # hyperplane H_e: end of Dubins path
            df['w_e']   = ze
            df['q_e']   = np.dot(dcm.rotz(anglend), np.array([1,0,0]).T)

        elif idx == 2: # right-left-straight-left
            theta = atan2(cle[1]-ci[1],cle[0]-ci[0])
            dist1 = R*fmod(2*pi-fmod(theta+pi/2,2*pi)+fmod(chii+pi/2,2*pi),2*pi)
            dist2 = psii*R
            dist3 = R*fmod(2*pi-fmod(anglend+pi/2,2*pi)+fmod(theta+pi/2,2*pi),2*pi)
            w1 = ci + df['R']*np.dot(dcm.rotz(theta+pi/2),e1.T).T + np.array([0, 0, -(dist1+dist2)*tan(gam)]).T
            w2 = cle + df['R']*np.dot(dcm.rotz(theta+pi/2),e1.T).T - np.array([0,0,-dist3*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2) # direction of line

            # start spiral
            df['c_s']   = crs
            df['psi_s'] = anglstart-pi/2
            df['lamda_s'] = 1
            df['k_s']   = 0
            # intermediate-start spiral
            df['c_si'] = ci + np.array([0, 0, -dist2*tan(gam)]).T
            df['psi_si'] = chii+pi/2
            df['lamda_si'] = -1
            df['k_si']   = 0
            # end spiral
            df['c_e'] = cle - np.array([0, 0, -dist3*tan(gam)]).T
            df['psi_e'] = theta+pi/2
            df['lamda_e'] = -1
            df['k_e']   = 0
            # hyperplane H_s: switch from first to second spiral
            df['w_s'] = zi - np.array([0, 0, -dist2*tan(gam)]).T
            df['q_s'] = np.array([cos(chii), sin(chii), 0]).T
            # hyperplane H_si: switch from second spiral to straight line
            df['w_si'] = w1
            df['q_si'] = q1
            # hyperplane H_l: switch from straight-line to end spiral
            df['w_l'] = w2
            df['q_l'] = q1
            # hyperplane H_e: end of Dubins path
            df['w_e'] = ze
            df['q_e'] = np.dot(dcm.rotz(anglend), np.array([1, 0, 0]).T)

        elif idx == 3: # left-right-straight-right
            theta = atan2(cre[1]-ci[1], cre[0] - ci[0])
            dist1 = R*fmod(2*pi+fmod(theta-pi/2,2*pi)-fmod(chii-pi/2,2*pi),2*pi)
            dist2 = psii*R
            dist3 = R*fmod(2*pi+fmod(anglend-pi/2,2*pi)-fmod(theta-pi/2,2*pi),2*pi)
            w1 = ci + df['R']*np.dot(dcm.rotz(theta-pi/2),e1.T).T + np.array([0, 0, -(dist1+dist2)*tan(gam)]).T
            w2 = cre + df['R']*np.dot(dcm.rotz(theta-pi/2),e1.T).T - np.array([0, 0, -dist3*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2) # direction of line
            # start spiral
            df['c_s']   = cls
            df['psi_s'] = anglstart+pi/2
            df['lamda_s'] = -1
            df['k_s'] = 0
            # intermediate-start spiral
            df['c_si'] = ci + np.array([0, 0, -dist2*tan(gam)]).T
            df['psi_si'] = chii-pi/2
            df['lamda_si'] = 1
            df['k_si'] = 0
            # end spiral
            df['c_e'] = cre - np.array([0, 0, -dist3*tan(gam)]).T
            df['psi_e'] = theta-pi/2
            df['lamda_e'] = 1
            df['k_e']   = 0
            # hyperplane H_s: switch from first to second spiral
            df['w_s'] = zi - np.array([0, 0, -dist2*tan(gam)]).T
            df['q_s'] = np.array([cos(chii), sin(chii), 0]).T
            # hyperplane H_si: switch from second spiral to straight line
            df['w_si'] = w1
            df['q_si'] = q1
            # hyperplane H_l: switch from straight-line to end spiral
            df['w_l'] = w2
            df['q_l'] = q1
            # hyperplane H_e: end of Dubins path
            df['w_e'] = ze
            df['q_e'] = np.dot(dcm.rotz(anglend), np.array([1,0,0]).T)

        elif idx == 4: # left-right-straight-left
            ell = np.linalg.norm(cle[0:2]-ci[0:2],ord=2)
            theta = atan2(cle[1] - ci[1], cle[0] - ci[0])
            theta2 = theta - pi/2 + asin(2*R/ell)
            dist1 = R*fmod(2*pi+fmod(theta2,2*pi) - fmod(chii-pi/2,2*pi),2*pi)
            dist2 = R*psii
            dist3 = R*fmod(2*pi+fmod(theta2+pi,2*pi) - fmod(anglend+pi/2,2*pi),2*pi)
            w1 = ci + R*np.dot(dcm.rotz(theta2),e1.T).T + np.array([0, 0, -(dist1+dist2)*tan(gam)]).T
            w2 = cle + R*np.dot(dcm.rotz(theta2+pi),e1.T).T - np.array([0, 0, -dist3*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2)

            # start spiral
            df['c_s'] = cls
            df['psi_s'] = anglstart+pi/2
            df['lamda_s'] = -1
            df['k_s'] = 0
            # intermediate-start spiral
            df['c_si'] = ci + np.array([0, 0, -dist2*tan(gam)]).T
            df['psi_si'] = chii-pi/2
            df['lamda_si'] = 1
            df['k_si']   = 0
            # end spiral
            df['c_e'] = cle - np.array([0, 0, -dist3*tan(gam)]).T
            df['psi_e'] = theta2+pi
            df['lamda_e'] = -1
            df['k_e'] = 0
            # hyperplane H_s: switch from first to second spiral
            df['w_s'] = zi - np.array([0, 0, -dist2*tan(gam)]).T
            df['q_s'] = np.array([cos(chii), sin(chii), 0]).T
            # hyperplane H_si: switch from second spiral to straight line
            df['w_si'] = w1
            df['q_si'] = q1
            # hyperplane H_l: switch from straight-line to end spiral
            df['w_l'] = w2
            df['q_l'] = q1
            # hyperplane H_e: end of Dubins path
            df['w_e'] = ze
            df['q_e'] = np.dot(dcm.rotz(anglend), np.array([1, 0, 0]).T)

    elif df['case'] == 3:
        if idx == 1: # right-straight-left-right
            # path specific calculations
            ell = np.linalg.norm(ci[0:2] - crs[0:2],ord=2)
            theta = atan2(ci[1] - crs[1], ci[0] - crs[0])
            theta2 = theta-pi/2 + asin(2*R/ell)
            dist1 = R*fmod(2*pi+fmod(theta2,2*pi) - fmod(anglstart-pi/2,2*pi),2*pi)
            dist2 = R*fmod(2*pi+fmod(theta2+pi,2*pi)-fmod(chii+pi/2,2*pi),2*pi)
            dist3 = fabs(R_min*psii)
            w1 = crs + R*np.dot(dcm.rotz(theta2),e1.T).T + np.array([0, 0, -dist1*tan(gam)]).T
            w2 = ci + R*np.dot(dcm.rotz(theta2+pi),e1.T).T - np.array([0, 0, -(dist2+dist3)*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2)
            # start spiral
            df['c_s']   = crs
            df['psi_s'] = anglstart-pi/2
            df['lamda_s'] = 1
            df['k_s']   = 0
            # intermediate-end spiral
            df['c_ei'] = ci - np.array([0, 0, -(dist2+dist3)*tan(gam)]).T
            df['psi_ei'] = theta2+pi
            df['lamda_ei'] = -1
            df['k_ei'] = 0
            # end spiral
            df['c_e'] = cre - np.array([0, 0, -dist3*tan(gam)]).T
            df['psi_e'] = anglend-pi/2-psii
            df['lamda_e'] = 1
            df['k_e'] = 0
            # hyperplane H_s: switch from first to second spiral
            df['w_s'] = w1
            df['q_s'] = q1
            # hyperplane H_l: switch from straight-line to intermediate spiral
            df['w_l'] = w2
            df['q_l'] = q1
            # hyperplane H_ei: switch from intermediate spiral to
            # end spiral
            df['w_ei']  = zi - np.array([0, 0, -dist3*tan(gam)]).T
            df['q_ei']  = np.array([cos(chii), sin(chii), 0]).T
            # hyperplane H_e: end of Dubins path
            df['w_e'] = ze
            df['q_e'] = np.dot(dcm.rotz(anglend),np.array([1,0,0]).T)

        elif idx == 2: # right-straight-right-left
            # path specific calculations
            theta = atan2(ci[1] - crs[1], ci[0] - crs[0])
            dist1 = R*fmod(2*pi+fmod(theta-pi/2,2*pi) - fmod(anglstart-pi/2,2*pi),2*pi)
            dist2 = R*fmod(2*pi+fmod(chii-pi/2,2*pi) - fmod(theta-pi/2,2*pi),2*pi)
            dist3 = fabs(R_min*psii)
            w1 = crs + R*np.dot(dcm.rotz(theta-pi/2),e1.T).T + np.array([0, 0, -dist1*tan(gam)]).T
            w2 = ci  + R*np.dot(dcm.rotz(theta-pi/2),e1.T).T - np.array([0, 0, -(dist2+dist3)*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2)
            # start spiral
            df['c_s'] = crs
            df['psi_s'] = anglstart-pi/2
            df['lamda_s'] = 1
            df['k_s'] = 0
            # intermediate-end spiral
            df['c_ei'] = ci - np.array([0, 0, -(dist2+dist3)*tan(gam)]).T
            df['psi_ei'] = theta - pi/2
            df['lamda_ei'] = 1
            df['k_ei'] = 0
            # end spiral
            df['c_e'] = cle - np.array([0, 0, -dist3*tan(gam)]).T
            df['psi_e'] = anglend+pi/2+psii
            df['lamda_e'] = -1
            df['k_e'] = 0
            # hyperplane H_s: switch from first to second spiral
            df['w_s'] = w1
            df['q_s'] = q1
            # hyperplane H_l: switch from straight-line to intermediate spiral
            df['w_l'] = w2
            df['q_l'] = q1
            # hyperplane H_ei: switch from intermediate spiral to
            # end spiral
            df['w_ei'] = zi - np.array([0, 0, -dist3*tan(gam)]).T
            df['q_ei'] = np.array([cos(chii), sin(chii), 0]).T
            # hyperplane H_e: end of Dubins path
            df['w_e'] = ze
            df['q_e'] = np.dot(dcm.rotz(anglend), np.array([1, 0, 0]).T)
            
        elif idx == 3: # left-straight-left-right
            # path specific calculations
            theta = atan2(ci[1]-cls[1],ci[0]-cls[0])
            dist1 = R*fmod(2*pi-fmod(theta+pi/2,2*pi)+fmod(anglstart+pi/2,2*pi),2*pi)
            dist2 = R*fmod(2*pi-fmod(chii+pi/2,2*pi)+fmod(theta+pi/2,2*pi),2*pi)
            dist3 = fabs(R_min*psii)
            w1 = cls + df['R']*np.dot(dcm.rotz(theta+pi/2),e1.T).T + np.array([0, 0, -dist1*tan(gam)]).T
            w2 = ci + df['R']*np.dot(dcm.rotz(theta+pi/2),e1.T).T - np.array([0, 0, -(dist2+dist3)*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2) # direction of line

            # start spiral
            df['c_s'] = cls
            df['psi_s'] = anglstart+pi/2
            df['lamda_s'] = -1
            df['k_s'] = 0
            # intermediate-end spiral
            df['c_ei'] = ci - np.array([0, 0, -(dist2+dist3)*tan(gam)]).T
            df['psi_ei'] = theta+pi/2
            df['lamda_ei'] = -1
            df['k_ei'] = 0
            # end spiral
            df['c_e'] = cre - np.array([0, 0, -dist3*tan(gam)]).T
            df['psi_e'] = anglend-pi/2-psii
            df['lamda_e'] = 1
            df['k_e'] = 0
            # hyperplane H_s: switch from first to second spiral
            df['w_s'] = w1
            df['q_s'] = q1
            # hyperplane H_l: switch from straight-line to intermediate spiral
            df['w_l'] = w2
            df['q_l'] = q1
            # hyperplane H_ei: switch from intermediate spiral to
            # end spiral
            df['w_ei'] = zi - np.array([0, 0, -dist3*tan(gam)]).T
            df['q_ei'] = np.array([cos(chii), sin(chii), 0]).T
            # hyperplane H_e: end of Dubins path
            df['w_e'] = ze
            df['q_e'] = np.dot(dcm.rotz(anglend),np.array([1,0,0]).T)

        elif idx == 4: # left-straight-right-left
            # path specific calculations
            ell = np.linalg.norm(ci[0:2] - cls[0:2],ord=2)
            theta = atan2( ci[1] - cls[1], ci[0] - cls[0])
            theta2 = acos(2*R/ell)
            dist1 = R*fmod(2*pi-fmod(theta+theta2,2*pi) + fmod(anglstart+pi/2,2*pi),2*pi)
            dist2 = R*fmod(2*pi-fmod(theta+theta2-pi,2*pi)+fmod(chii-pi/2,2*pi),2*pi)
            dist3 = fabs(R_min*psii)
            w1 = cls + R*np.dot(dcm.rotz(theta+theta2),e1.T).T + np.array([0, 0, -dist1*tan(gam)]).T
            w2 = ci + R*np.dot(dcm.rotz(-pi+theta+theta2),e1.T).T - np.array([0, 0, -(dist2+dist3)*tan(gam)]).T
            q1 = (w2-w1)/np.linalg.norm(w2-w1,ord=2)

            # start spiral
            df['c_s'] = cls
            df['psi_s'] = anglstart+pi/2
            df['lamda_s'] = -1
            df['k_s'] = 0
            # intermediate-end spiral
            df['c_ei'] = ci - np.array([0, 0, -(dist2+dist3)*tan(gam)]).T
            df['psi_ei'] = fmod(theta+theta2-pi,2*pi)
            df['lamda_ei'] = 1
            df['k_ei'] = 0
            # end spiral
            df['c_e'] = cle - np.array([0, 0, -dist3*tan(gam)]).T
            df['psi_e'] = anglend+pi/2+psii
            df['lamda_e'] = -1
            df['k_e'] = 0
            # hyperplane H_s: switch from first to second spiral
            df['w_s'] = w1
            df['q_s'] = q1
            # hyperplane H_l: switch from straight-line to intermediate spiral
            df['w_l'] = w2
            df['q_l'] = q1
            # hyperplane H_ei: switch from intermediate spiral to
            # end spiral
            df['w_ei'] = zi - np.array([0, 0, -dist3*tan(gam)]).T
            df['q_ei'] = np.array([cos(chii), sin(chii), 0]).T
            # hyperplane H_e: end of Dubins path
            df['w_e'] = ze
            df['q_e'] = np.dot(dcm.rotz(anglend), np.array([1, 0, 0]).T)

    if df['case'] == 4:
        print( '### Not Implemented Case')

    return df

def dubins(init_conf, final_conf, Vehicle, step=0.01):
    # Generate the Dubins Airplane path
    dfs = _generate(init_conf, final_conf, Vehicle)
    
    # Extract the Dubins Airplane Solution in vector form
    if dfs['case'] == 1: # spiral - line - spiral
        r1 = compute.spiral(dfs['R'],dfs['gamma'], dfs['c_s'], dfs['psi_s'], dfs['lamda_s'], dfs['k_s'], dfs['w_s'], dfs['q_s'],step)
        r2 = compute.line(dfs['w_s'], dfs['q_s'], dfs['w_l'], dfs['q_l'], step)
        r3 = compute.spiral(dfs['R'], dfs['gamma'], dfs['c_e'], dfs['psi_e'], dfs['lamda_e'], dfs['k_e'], dfs['w_e'], dfs['q_e'], step)
        r = np.hstack((r1,r2))
        r = np.hstack((r,r3))
        
    elif dfs['case'] == 2: # spiral - spiral - line -spiral
        r1 = compute.spiral(dfs['R'],dfs['gamma'],dfs['c_s'],dfs['psi_s'],dfs['lamda_s'],dfs['k_s'],dfs['w_s'],dfs['q_s'],step)
        r2 = compute.spiral(dfs['R'],dfs['gamma'],dfs['c_si'],dfs['psi_si'],dfs['lamda_si'],dfs['k_si'],dfs['w_si'],dfs['q_si'],step)
        r3 = compute.line(dfs['w_si'],dfs['q_si'],dfs['w_l'],dfs['q_l'],step)
        r4 = compute.spiral(dfs['R'],dfs['gamma'],dfs['c_e'],dfs['psi_e'],dfs['lamda_e'],dfs['k_e'],dfs['w_e'],dfs['q_e'],step)
        r = np.hstack((r1,r2))
        r = np.hstack((r,r3))
        r = np.hstack((r,r4))

    elif dfs['case'] == 3: # spiral - line - spiral - spiral
        r1 = compute.spiral(dfs['R'],dfs['gamma'],dfs['c_s'],dfs['psi_s'],dfs['lamda_s'],dfs['k_s'],dfs['w_s'],dfs['q_s'],step)
        r2 = compute.line(dfs['w_s'],dfs['q_s'],dfs['w_l'],dfs['q_l'],step)
        r3 = compute.spiral(dfs['R'],dfs['gamma'],dfs['c_ei'],dfs['psi_ei'],dfs['lamda_ei'],dfs['k_ei'],dfs['w_ei'],dfs['q_ei'],step)
        r4 = compute.spiral(dfs['R'],dfs['gamma'],dfs['c_e'],dfs['psi_e'],dfs['lamda_e'],dfs['k_e'],dfs['w_e'],dfs['q_e'],step)
        r = np.hstack((r1,r2))
        r = np.hstack((r,r3))
        r = np.hstack((r,r4))
    
    d = {'X':r[0,:], 'Y':r[1,:], 'Z':r[2,:]} 
    path = pd.DataFrame(data=d)

    return path

