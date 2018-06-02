import numpy as np
import matplotlib.pyplot as plt
from numericalMethods import derivative, initialValueSolution


def metricPolar():
    def metricFunc(rad, theta):
        return np.array([[1, 0], [0, rad**2]])
    return metricFunc


# 2 Sphere surface
def metric2Sphere(r):
    def metricFunc(theta, phi):
        return np.array([[r**2, 0], [0, r**2*np.sin(theta)**2]])
    return metricFunc


# Schwarzschild with G = c = 1
# rs=2m is the Schwarzschild radius
def metricSchwarzschild(rs):
    def metricFunc(t, rad, theta, phi):
        return np.diag([-(1-rs/rad), 1/(1-rs/rad), rad**2, rad**2*np.sin(theta)**2])
    return metricFunc


# Schwarzschild metric with Gullstrand–Painlevé coordinates with G = c = 1
def metricSGP(rs):
    def metricFunc(t, rad, theta, phi):
        print("Unsure if metric is correct after trying to switch signature.")
        m = np.diag([-(1-rs/rad), 1, rad**2, rad**2*np.sin(theta)**2])
        m[0][1] = 1*np.sqrt(rs/rad)
        m[1][0] = 1*np.sqrt(rs/rad)
        return m
    return metricFunc


# a<rs/2
# a=Kerr metric measure of rotation
def metricKerr(rs, a):
    def metricFunc(t, rad, theta, phi):
        assert a < rs/2
        Sigma = rad**2 + a**2*np.cos(theta)**2
        Delta = rad**2 - rad*rs + a**2

        m = np.diag([-(1-rs*rad/Sigma), Sigma/Delta, Sigma, (rad**2+a**2+(rs*rad*a**2/Sigma)*np.sin(theta)**2)*np.sin(theta)**2])
        m[0][3] = -(rs*rad*a*np.sin(theta)**2)/Sigma
        m[3][0] = -(rs*rad*a*np.sin(theta)**2)/Sigma
        return m
    return metricFunc


global metric
metric = None


def compute_christoffel(*args):
    d = len(args)  # number of dimensions

    csymbols = np.zeros((d, d, d))  # first index is the upper index

    dg = []  # First index is the derivative!
    for i in range(d):
        dg.append(derivative(metric, i, *args))

    ginverse = np.linalg.inv(metric(*args))

    for mu in range(d):
        for nu in range(d):
            for sigma in range(d):
                for dummy in range(d):
                    term1 = dg[nu][mu][dummy]
                    term2 = dg[mu][nu][dummy]
                    term3 = dg[dummy][mu][nu]

                    csymbols[sigma][mu][nu] += .5*ginverse[sigma][dummy]*(term1+term2-term3)

    return csymbols


def compute_riemann_christoffel(*args):
    d = len(args)  # number of dimensions

    riemann = np.zeros((d, d, d, d))  # first index is the upper index

    gamma = compute_christoffel
    gammaEval = gamma(*args)
    dgamma = []  # First index is the derivative!
    for i in range(d):
        dgamma.append(derivative(gamma, i, *args))

    for epsilon in range(d):
        for mu in range(d):
            for nu in range(d):
                for sigma in range(d):
                    tot = 0

                    for alpha in range(d):
                        tot += gammaEval[alpha][mu][sigma]*gammaEval[epsilon][alpha][nu]
                        tot -= gammaEval[alpha][mu][nu]*gammaEval[epsilon][alpha][sigma]

                    tot += dgamma[nu][epsilon][mu][sigma]
                    tot -= dgamma[sigma][epsilon][mu][nu]

                    riemann[epsilon][mu][nu][sigma] = tot

    return riemann


# Defined as the contraction of the first and last indices of the Riemann tensor
def compute_riccitensor(*args):
    d = len(args)  # number of dimensions

    riccitensor = np.zeros((d, d))

    riemann = compute_riemann_christoffel(*args)

    for mu in range(d):
        for nu in range(d):
            tot = 0
            for sigma in range(d):
                tot += riemann[sigma][mu][sigma][nu]

            riccitensor[mu][nu] = tot

    return riccitensor


def compute_ricciscalar(*args):
    d = len(args)  # number of dimensions

    ginverse = np.linalg.inv(metric(*args))
    riccitensor = compute_riccitensor(*args)

    ricciscalar = 0
    for mu in range(d):
        for nu in range(d):
            ricciscalar += ginverse[mu][nu]*riccitensor[mu][nu]

    return ricciscalar


# y should be an array with each coordinate value followed directly by its
# derivative with respect to s. This function is intended to be used to
# calculate the magnitude of a velocity vector at a given point. This is useful
# for testing if a velocity vector is null for example.
def velocityMagnitude(y):
    assert len(y) % 2 == 0
    d = len(y)//2  # number of dimensions
    g = metric(*y[::2])

    tot = 0
    for mu in range(d):
        for nu in range(d):
            tot += g[mu][nu]*y[2*mu+1]*y[2*nu+1]

    return tot


# y should be a vector corresponding to each coordinate with its derivative
# following (x, dx, y, dy, ...)
# stop is a functions taking in s, y, N which returns true when a stop condition is met.
# N is the number of steps taken.
def compute_geodesic(s_0, y_0, stop, tol=1.0E-5):
    assert len(y_0) % 2 == 0
    d = len(y_0)//2

    y_0 = np.array(y_0).astype(float)

    def dyds(s, y):
        csymbols = compute_christoffel(*y[::2])
        f = []
        for i in range(d):
            f.append(y[2*i+1])

            secondDeriv = 0
            for alpha in range(d):
                for nu in range(d):
                    secondDeriv += -1*csymbols[i][alpha][nu]*y[2*alpha+1]*y[2*nu+1]

            f.append(secondDeriv)

        return np.array(f)

    return initialValueSolution(s_0, y_0, dyds, stop, tol=tol)


# wrapper for compute_geodesic
def compute_null_geodesic(s_0, y_0, stop, tol=1.0E-5):
    assert abs(velocityMagnitude(y_0)) < 10**-10

    nullCheck = 1
    while nullCheck > 10**-2:
        tvals, yvals = compute_geodesic(s_0, y_0, stop, tol=tol)

        nullCheck = abs(velocityMagnitude(yvals[-1]))
        tol /= 10

    return tvals, yvals


# wrapper for compute_geodesic
def compute_timelike_geodesic(s_0, y_0, stop, tol=1.0E-5):
    allowedError = .1

    v = -1
    assert v-10**-10 < velocityMagnitude(y_0) < v+10**-10

    Check = False
    while not Check:
        tvals, yvals = compute_geodesic(s_0, y_0, stop, tol=tol)

        velM = velocityMagnitude(yvals[-1])
        Check = v-allowedError < velM < v+allowedError
        tol /= 10
        if not Check:
            print("dang")

    return tvals, yvals


def polarToRectangular(rad, phi):
    x = np.array(rad)*np.cos(phi)
    y = np.array(rad)*np.sin(phi)

    return x, y
