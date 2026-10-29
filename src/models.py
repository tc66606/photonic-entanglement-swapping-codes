"""Analytical formulas for the resource and filtering comparisons.
Probabilities are per supplied trial unless explicitly labeled otherwise."""
from math import cos, sin, sqrt, log, exp, pi, acos
import numpy as np
from scipy.optimize import brentq


def source_densities(p,c,helper='midpoint'):
    """Symmetric qubit sources in the midpoint basis; p is Bloch length."""
    if not (0<=p<=1 and 0<=c<=1) or helper not in ('midpoint','opposite'):
        raise ValueError('Require p,c in [0,1] and midpoint/opposite helper.')
    off=p*sqrt(1-c*c)/2
    A=np.array([[(1+p*c)/2,off],[off,(1-p*c)/2]],complex)
    B=A.copy();B[0,1]*=-1;B[1,0]*=-1
    H=np.diag([1.,0.]) if helper=='midpoint' else np.diag([0.,1.])
    return np.array([A,B,H],complex)


def bare_probability(tau,target):
    if target > (1+tau)/2+1e-13:return 0.
    if target <= .5+tau/4:return 1.
    return tau/(4*(target-.5))


def four_point(c):
    den = 2-c+2*c*c
    return dict(P=den/72, F=(1+c+c*c)/den)


def gaussian_target(c, target):
    """Optimum: equal widths, independently adjustable centers, complete bare policy.

    Targets lie above the bare ceiling. Intensity variance = 1;
    input means +/-mu and filter centers +/-nu.
    """
    if not (0 < c < 1 and (1+c*c)/2 < target <= 1):
        raise ValueError('Require 0<c<1 and (1+c^2)/2 < target <= 1.')
    a = -2*log(c)
    t = 2*target-1
    b = sqrt(max(0., -log(t)/a))
    residual = lambda q: (1-q*q)**2-a*q*(q-b)*(1-b*q)
    q = brentq(residual, b, 1., xtol=5e-15)
    r = q*q
    mu = sqrt(a)
    shift = 1-b/q
    nu = -shift*r*mu/(1-r)
    total_transmission = r*exp(-a*(1-b*q)**2/(1-r))
    return dict(c=c, target_F=target, q=q, r=r, mu=mu, nu=nu,
                kappa=sqrt(r/(1-r)), transmitted_mean=r*mu+(1-r)*nu,
                P=total_transmission/2, F=target,
                scope='equal-width independent-center Gaussian filters')


def efficiency(n, layers, sA=.8, sB=.8, sH=.7, path=.95, detector=.9, layer=.98):
    """Uniform padded layers, equal independent helper supply probabilities."""
    return sA*sB*sH**(n-2)*(path*detector)**n*layer**(n*layers)


def calibration_amplitudes(offsets):
    d02,d13,d14,d01 = offsets
    al,be,ga,de = pi/4+d02, pi/4+d13, acos(1/sqrt(3))+d14, pi/4+d01
    return np.array([
        -cos(ga)*cos(de)*sin(de)*cos(al)*sin(al),
        cos(2*de)*cos(al)*sin(be)*cos(2*ga)/2,
        cos(2*de)*sin(al)*cos(be)*cos(2*ga)/2,
        cos(de)*sin(de)*cos(be)*sin(be)*cos(ga)*(3*cos(ga)**2-2)])


def symmetry_weights(A,B,H):
    tau = np.trace(A@B).real
    qa,qb = np.trace(A@H).real,np.trace(B@H).real
    w = np.trace(H@A@B)
    s = (1+tau+qa+qb+2*w.real)/6
    a = (1-tau-qa-qb+2*w.real)/6
    return dict(s=float(s), a=float(a), Phi_plus=float(s/24), Psi_minus=float(a/8),
                cyclic_trace_real=float(w.real), cyclic_trace_imag=float(w.imag))


def pure_sources(c):
    a = np.array([1.,0.])
    b = np.array([c,sqrt(1-c*c)])
    return a,b,(a+b)/sqrt(2*(1+c)),(a-b)/sqrt(2*(1-c))
