"""
Created on Thu Jul  2 15:36:42 2020

@author:  Kostas Alexis (konstantinos.alexis@mavt.ethz.ch)
"""
import numpy as np
from math import tan, sin, cos, atan2, fmod, acos, asin, pow, sqrt, fabs
from dubins_airplane import dcm

pi = np.pi

# define a minimum error to prevent the algo from getting stuck
epsilon = 0.0000000001

def RSR(R=None, crs=None, cre=None, anglstart=None,anglend=None):
    # Compute Dubins RSR
    theta = atan2( cre[1]-crs[1], cre[0]-crs[0] )
    L = np.linalg.norm( crs[0:2]-cre[0:2],ord=2 ) + R* fmod( 2*pi+fmod(theta-pi/2,2*pi )-fmod( anglstart-pi/2,2*pi),2*pi ) + R*fmod( 2*pi+fmod(anglend-pi/2,2*pi )-fmod( theta-pi/2,2*pi),2*pi )
    return L


def LSL(R=None, cls=None, cle=None, anglstart=None, anglend=None):
    # Compute Dubins LSL
    theta = atan2( cle[1]-cls[1], cle[0]-cls[0] )
    L = np.linalg.norm(cls[0:2]-cle[0:2]) + R*fmod( 2*pi-fmod(theta+pi/2,2*pi)+fmod(anglstart+pi/2,2*pi),2*pi ) + R*fmod( 2*pi-fmod(anglend+pi/2,2*pi)+fmod(theta+pi/2,2*pi),2*pi )
    return L

def LSR(R=None, cls=None, cre=None, anglstart=None, anglend=None):
    # Compute Dubins LSR
    ell = np.linalg.norm( cre[0:2]-cls[0:2],ord=2 )
    theta = atan2( cre[1]-cls[1],cre[0]-cls[0] )
    acos_value = 2 * R / ell
    if fabs( acos_value ) > 1:
        flag_zero = 1
    else:
        flag_zero = 0

    acos_value = max( acos_value,-1 )
    acos_value = min( acos_value,1 )
    if ell == 0:
        theta2 = 0
    else:
        theta2 = acos( acos_value )

    if flag_zero == 1:
        theta2 = 0

    if theta2 == 0:
        L = pow(10.0,8)
    else:
        L = sqrt( pow(ell,2) - 4*pow(R,2) ) + R*fmod( 2*pi-fmod(theta+theta2,2*pi ) + fmod( anglstart+pi/2,2*pi),2*pi ) + R*fmod( 2*pi-fmod(theta+theta2-pi,2*pi)+fmod(anglend-pi/2,2*pi),2*pi )
    return L

def RSL(R=None, crs=None, cle=None, anglstart=None, anglend=None):
    # Compute Dubins RSL
    ell = np.linalg.norm( cle[0:2]-crs[0:2],ord=2 )
    theta = atan2( cle[1]-crs[1],cle[0]-crs[0] )
    asin_value = 2 * R / ell
    if fabs( asin_value ) > 1:
        flag_zero = 1
    else:
        flag_zero = 0
    asin_value = max( asin_value, -1 )
    asin_value = min( asin_value, 1 )
    if ell == 0:
        theta2 = 0
    else:
        theta2 = theta - pi/2 + asin( asin_value )

    if theta2 == 0:
        L = pow( 10.0, 8 )
    else:
        L = sqrt( fabs(pow(ell,2)-4 * pow(R,2)) ) + R * fmod( 2 * pi + fmod( theta2, 2 * pi )-fmod( anglstart-pi/2, 2 * pi ), 2 * pi ) + R * fmod( 2 * pi + fmod( theta2 + pi, 2 * pi )-fmod( anglend + pi / 2, 2 * pi ), 2 * pi )
    return L

def OptimalRadius(zs=None, anglstart=None, ze=None, anglend=None, R_min=None, gamma_max=None, idx=None, k=None, hdist=None):
    # Compute Optimal Radius
    R1 = R_min
    R2 = 2 * R_min
    R = ( R1 + R2 ) / 2

    if idx == 1:
        error = 1
        while fabs( error ) > 0.1:
            crs = zs + R * np.dot( dcm.rotz( pi / 2 ),np.array( [cos(anglstart), sin(anglstart), 0] ).T )
            cre = ze + R * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
            L = RSR(R, crs, cre, anglstart, anglend)
            error = ( L + 2 * pi * k * R ) - fabs( hdist ) / tan( gamma_max )
            if error > 0:
                R2 = R
            else:
                R1 = R
            R= ( R1 + R2 ) / 2
    elif idx == 2:
        error = 1
        while fabs( error ) > 0.1:
            crs = zs + R * np.dot( dcm.rotz( pi / 2 ),np.array( [cos(anglstart), sin(anglstart), 0] ).T )
            cle = ze + R * np.dot( dcm.rotz( -pi/2 ),np.array( [cos(anglend), sin(anglend), 0] ).T )
            L = RSL( R, crs, cle, anglstart, anglend )
            error = ( L + 2 * pi * k * R ) * tan( gamma_max ) - fabs( hdist )
            if error > 0:
                R2 = R
            else:
                R1 = R
            R = ( R1 + R2 ) / 2
    elif idx == 3:
        error = 1
        while fabs( error ) > 0.1:
            cls = zs + R * np.dot( dcm.rotz( -pi / 2 ), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
            cre = ze + R * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
            L = LSR( R, cls, cre, anglstart, anglend )
            error = ( L + 2 * pi * k * R ) * tan( gamma_max ) - fabs( hdist )
            if error > 0:
                R2 = R
            else:
                R1 = R
            R = ( R1 + R2 ) / 2
    elif idx == 4:
        error = 1
        while fabs( error ) > 0.1:
            cls = zs + R * np.dot( dcm.rotz( -pi / 2 ), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
            cle = ze + R * np.dot( dcm.rotz( -pi / 2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
            L = LSL( R, cls, cle, anglstart, anglend )
            error = ( L + 2 * pi * k * R ) * tan( gamma_max ) - fabs( hdist )
            if error > 0:
                R2 = R
            else:
                R1 = R
            R = ( R1 + R2 ) / 2
    return R

def MinTurnRadius(V=None,phi_max=None):
    # Compute Minimum Turning Radius
    g = 9.8065
    Rmin = pow( V,2 ) / (g * tan( phi_max ) )
    return Rmin

def line(w1=None, q1=None, w2=None, q2=None, step=None):
    # extract line path
    r = w1
    # propagate line until cross half plane
    s = 0

    NrNc = r.shape
    if len(NrNc) == 1:
        NrNc_ind = NrNc[0]
        last_col = r[:]
    else:
        NrNc_ind = NrNc[1]
        last_col = r[:,NrNc[1]-1]

    r.shape = (3,1)
    while np.dot( (last_col - w2).T,q2 ) <= 0:
        s = s + step
        w1.shape = (3,1)
        q1.shape = (3,1)
        new_col = w1+s*q1
        new_col.shape = (3,1)
        r = np.hstack( (r,  new_col) )
        NrNc = r.shape
        if len(NrNc) == 1:
            NrNc_ind = NrNc[0]
            last_col = r[:]
        else:
            NrNc_ind = NrNc[1]
            last_col = r[:,NrNc[1]-1]

    return r

def spiral(R=None, gam=None, c=None, psi=None, lam=None, k=None, w=None, q=None, step=None):
    # extract spiral path
    r = np.zeros((1,1))
    r = c.T + R*np.array( [cos(psi), sin(psi), 0] ).T
    r = r.T
    # determine number of required crossings of half plane
    NrNc = r.shape
    if len(NrNc) ==1 :
        NrNc_ind = NrNc[0]
        halfplane = np.dot( (r[0:2]-w[0:2].T),q[0:2] )
    else:
        NrNc_ind = NrNc[1]
        halfplane = np.dot( (r[0:2,NrNc_ind-1]-w[0:2].T),q[0:2] )


    if (halfplane > 0).all() :
        required_crossing = 2 * ( k + 1 )
    else:
        required_crossing = 2 * k + 1

    # propagate spiral until cross half plane the right number of times
    s = 0
    r.shape = (3,1)
    while ( required_crossing > 0 ) or ( (halfplane <= 0).all() ):
        s = s +step
        new_col = (c + R * np.array( [ cos(lam*s+psi), sin(lam*s+psi), -s*tan(gam)] ).T )
        new_col.shape = (3,1)
        r = np.hstack( (r, new_col) )

        NrNc = r.shape
        if len(NrNc)==1 :
            NrNc_ind = NrNc[0]
            if np.sign( halfplane ) != np.sign( np.dot((r[0:2]-w[0:2].T),q[0:2]) ):
                halfplane = np.dot( ( r[0:2] - w[0:2].T ), q[0:2] )
                required_crossing = required_crossing - 1
        else:
            NrNc_ind = NrNc[1]
            if np.sign(halfplane) != np.sign( np.dot( (r[0:2,NrNc_ind-1]-w[0:2].T ), q[0:2]) ):
                halfplane = np.dot( (r[0:2,NrNc_ind-1] - w[0:2] ).T, q[0:2] )
                required_crossing = required_crossing - 1
    return r

def addSpiralBeginning(zs=None, anglstart=None, ze=None, anglend=None, R_min=None, gamma_max=None, idx=None, hdist=None):
    # Add Spiral in the Dubins Airplane Path beginning
    cli = np.zeros((3,1))
    cri = np.zeros((3,1))
    zi = np.zeros((3,1))
    anglinter = 0
    L = 0
    ci = np.zeros((3,1))
    psii = 0
    psi1 = 0
    psi2 = 2 * pi
    psi = ( psi1 + psi2 ) / 2

    if idx == 1: # RLSR
        crs = zs + R_min * np.dot( dcm.rotz( pi/2 ), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
        cre = ze + R_min * np.dot( dcm.rotz( pi/2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
        L = RSR( R_min, crs, cre, anglstart, anglend )
        error = L - fabs( hdist / tan( gamma_max ) )
        while fabs( error ) > 0.001:
            zi = crs + np.dot( dcm.rotz( psi ),( zs-crs ) )
            anglinter = anglstart + psi
            cli = zi + R_min * np.dot( dcm.rotz( -pi/2 ), np.array( [cos(anglinter), sin(anglinter), 0] ).T )
            L = LSR( R_min, cli, cre, anglinter, anglend )
            error = ( L + fabs( psi ) * R_min ) - fabs( hdist / tan( gamma_max ) )

            if error > 0:
                psi2 = (179*psi2+psi)/180
            else:
                psi1 = (179*psi1+psi)/180

            if fabs(psi1 - psi2) < epsilon:
                print( 'unable to minimize error futher -- going with what we got...' )
                break

            psi = ( psi1 + psi2 ) / 2

        zi = crs + np.dot( dcm.rotz( psi ), ( zs-crs ) )
        anglinter = anglstart + psi
        L = L + fabs( psi ) * R_min
        ci = cli
        psii = psi

    elif idx == 2: # RLSL
        crs = zs + R_min * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
        cle = ze + R_min * np.dot( dcm.rotz( -pi / 2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
        L = RSL( R_min, crs, cle, anglstart, anglend )
        error = L - fabs( hdist / tan( gamma_max ) )
        while fabs( error ) > 0.001:
            zi = crs + np.dot( dcm.rotz( psi ), ( zs-crs ) )
            anglinter = anglstart + psi
            cli = zi + R_min * np.dot( dcm.rotz( -pi / 2 ), np.array( [cos(anglinter), sin(anglinter), 0] ).T )
            L = LSL( R_min, cli, cle, anglinter, anglend )
            error = ( L + fabs( psi ) * R_min ) - fabs( hdist / tan( gamma_max ) )

            if error > 0:
                psi2 = (179*psi2+psi)/180
            else:
                psi1 = (179*psi1+psi)/180

            if fabs(psi1 - psi2) < epsilon:
                print( 'unable to minimize error futher -- going with what we got...' )
                break

            psi = ( psi1 + psi2 ) / 2

        zi   = crs + np.dot( dcm.rotz( psi ), ( zs-crs ) )
        anglinter = anglstart + psi
        L = L + fabs( psi ) * R_min
        ci = cli
        psii = psi

    elif idx == 3: # LRSR
        cls = zs + R_min * np.dot( dcm.rotz( -pi/2 ), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
        cre = ze + R_min * np.dot( dcm.rotz( pi/2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
        L = LSR( R_min, cls, cre, anglstart, anglend )
        error = L - fabs( hdist / tan( gamma_max ) )

        while fabs( error ) > 0.001:
            zi = cls + np.dot( dcm.rotz( -psi ), ( zs-cls ) )
            anglinter = anglstart - psi
            cri = zi + R_min * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglinter), sin(anglinter), 0] ).T )
            L = RSR( R_min, cri, cre, anglinter, anglend )
            error = ( L + fabs( psi ) * R_min ) - fabs( hdist / tan( gamma_max ) )

            if error > 0:
                psi2 = (179*psi2+psi)/180
            else:
                psi1 = (179*psi1+psi)/180

            if fabs(psi1 - psi2) < epsilon:
                print( 'unable to minimize error futher -- going with what we got...' )
                break

            psi = ( psi1 + psi2 ) / 2

        zi   = cls + np.dot( dcm.rotz( -psi ), ( zs-cls ) )
        anglinter = anglstart - psi
        L = L + fabs( psi ) * R_min
        ci = cri
        psii = psi

    elif idx == 4: # LRSL
        cls = zs + R_min * np.dot( dcm.rotz( -pi/2 ), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
        cle = ze + R_min * np.dot( dcm.rotz( -pi/2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
        # above modified by liucz 2015-10-12, fix spell mistake cre -> cle
        # origin is "cre = ze + R_min * np.dot( dcm.rotz( -pi/2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )"
        L = LSL( R_min, cls, cle, anglstart, anglend )
        error = L - fabs( hdist / tan( gamma_max ) )

        while fabs( error ) > 0.001:
            zi = cls + np.dot( dcm.rotz( -psi ), ( zs-cls ) )
            anglinter = anglstart - psi
            cri = zi + R_min * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglinter), sin(anglinter), 0 ] ).T )
            # above is modified by licz 2015-10-12, fix written mistake np.array -> np.dot
            # origin is "cri = zi + R_min * np.array( dcm.rotz( pi / 2 ), np.array( [cos(anglinter), sin(anglinter), 0 ] ).T )"
            L = RSL( R_min, cri, cle, anglinter, anglend )
            error = ( L + fabs( psi ) * R_min ) - fabs( hdist / tan( gamma_max) )

            if error > 0:
                psi2 = (179*psi2+psi)/180
            else:
                psi1 = (179*psi1+psi)/180

            if fabs(psi1 - psi2) < epsilon:
                print( 'unable to minimize error futher -- going with what we got...' )
                break

            psi = ( psi1 + psi2 ) / 2

        zi   = cls + np.dot( dcm.rotz( -psi ), ( zs-cls ) )
        anglinter = anglstart - psi
        L = L + fabs( psi ) * R_min
        ci = cri
        psii = psi

    return zi, anglinter, L, ci, psii



def addSpiralEnd(zs=None, anglstart=None, ze=None, anglend=None, R_min=None, gamma_max=None, idx=None, hdist=None):
    # Add Spiral at the end of the Dubins Airplane path
    cli = np.zeros((3,1))
    cri = np.zeros((3,1))
    zi = np.zeros((3,1))
    anglinter = 0
    L = 0
    ci = np.zeros((3,1))
    psii = 0
    psi1 = 0
    psi2 = 2 * pi
    psi = ( psi1 + psi2 ) / 2

    if idx == 1: # RSLR
        crs = zs + R_min * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
        cre = ze + R_min * np.dot( dcm.rotz( pi/2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
        L = RSR( R_min, crs, cre, anglstart, anglend )
        error = L - fabs( hdist / tan( gamma_max ) )

        while fabs( error ) > 0.001:
            zi = cre + np.dot( dcm.rotz( -psi ), ( ze-cre ) )
            anglinter = anglend - psi
            cli = zi + R_min * np.dot( dcm.rotz( -pi / 2 ), np.array( [cos(anglinter), sin(anglinter), 0] ).T )
            L = RSL( R_min, crs, cli, anglstart, anglinter )
            error = ( L + fabs( psi ) * R_min ) - fabs( hdist / tan( gamma_max ) )

            if error > 0:
                psi2 = (179*psi2+psi)/180
            else:
                psi1 = (179*psi1+psi)/180

            if fabs(psi1 - psi2) < epsilon:
                print( 'unable to minimize error futher -- going with what we got...' )
                break

            psi = ( psi1 + psi2 ) / 2

        zi   = cre + np.dot( dcm.rotz( -psi ), ( ze-cre ) )
        anglinter = anglend - psi
        L = L + abs( psi ) * R_min
        ci = cli
        psii = psi

    elif idx == 2: # RSRL
        crs = zs + R_min * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
        cle = ze + R_min * np.dot( dcm.rotz( -pi/2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
        L = RSL( R_min, crs, cle, anglstart, anglend )
        error = L - fabs( hdist / tan( gamma_max ) )

        while fabs( error ) > 0.001:
            zi = cle + np.dot( dcm.rotz( psi ), ( ze-cle ) )
            anglinter = anglend + psi
            cri = zi + R_min * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglinter), sin(anglinter), 0] ).T )
            L = RSR( R_min, crs, cri, anglstart, anglinter )
            error = ( L + fabs( psi ) * R_min ) - fabs( hdist / tan( gamma_max ) )

            if error > 0:
                psi2 = (179*psi2+psi)/180
            else:
                psi1 = (179*psi1+psi)/180

            if fabs(psi1 - psi2) < epsilon:
                print( 'unable to minimize error futher -- going with what we got...' )
                break

            psi = ( psi1 + psi2 ) / 2

        zi = cle + np.dot( dcm.rotz( psi ), ( ze-cle ) )
        anglinter = anglend + psi
        cri  = zi + R_min * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglinter), sin(anglinter), 0] ).T )
        L = L + fabs( psi ) * R_min
        ci = cri
        psii = psi

    elif idx == 3: # LSLR
        cls = zs + R_min * np.dot( dcm.rotz( -pi / 2 ), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
        cre = ze + R_min * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
        L = LSR( R_min, cls, cre, anglstart, anglend )
        error = L - fabs( hdist / tan( gamma_max ))

        while fabs( error ) > 0.001:
            zi = cre + np.dot( dcm.rotz( -psi ), ( ze-cre ) )
            anglinter = anglend - psi
            cli = zi + R_min * np.dot( dcm.rotz( -pi / 2 ), np.array( [cos(anglinter), sin(anglinter), 0] ).T )
            L = LSL( R_min, cls, cli, anglstart, anglinter )
            error = ( L + fabs( psi ) * R_min ) - fabs( hdist / tan( gamma_max ) )

            if error > 0:
                psi2 = (179*psi2+psi)/180
            else:
                psi1 = (179*psi1+psi)/180

            if fabs(psi1 - psi2) < epsilon:
                print( 'unable to minimize error futher -- going with what we got...' )
                break

            psi = ( psi1 + psi2 ) / 2

        zi   = cre + np.dot( dcm.rotz( -psi ), ( ze-cre ) )
        anglinter = anglend - psi
        L = L + fabs( psi ) * R_min
        ci = cli
        psii = psi

    elif idx == 4:

        cls = zs + R_min * np.dot( dcm.rotz( -pi/2 ), np.array( [cos(anglstart), sin(anglstart), 0] ).T )
        cle = ze + R_min * np.dot( dcm.rotz( -pi/2 ), np.array( [cos(anglend), sin(anglend), 0] ).T )
        L = LSL( R_min, cls, cle, anglstart, anglend )
        error = L - fabs( hdist / tan( gamma_max ) )

        while fabs( error ) > 0.001:
            zi = cle + np.dot( dcm.rotz( psi ), ( ze-cle ) )
            anglinter = anglend + psi
            cri = zi + R_min * np.dot( dcm.rotz( pi / 2 ), np.array( [cos(anglinter), sin(anglinter), 0] ).T )
            L = LSR( R_min, cls, cri, anglstart, anglinter )
            error = ( L + fabs( psi ) * R_min ) - fabs( hdist / tan( gamma_max ) )

            if error > 0:
                psi2 = (179*psi2+psi)/180
            else:
                psi1 = (179*psi1+psi)/180

            if fabs(psi1 - psi2) < epsilon:
                print( 'unable to minimize error futher -- going with what we got...' )
                break

            psi = ( psi1 + psi2 ) / 2

        zi = cle + np.dot( dcm.rotz( psi ), ( ze-cle ))
        anglinter = anglend + psi
        L = L + fabs( psi ) * R_min
        ci = cri
        psii = psi

    return zi, anglinter, L, ci, psii

