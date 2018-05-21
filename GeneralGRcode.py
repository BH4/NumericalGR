import numpy as np
import matplotlib.pyplot as plt
from numericalMethods import derivative, initialValueSolution

# define the global values which are used in any metric
global r
global rs
r = 1.0
rs = 2.0


def metricPolar(rad, theta):
    return np.array([[1, 0], [0, rad**2]])


# 2 Sphere surface
def metric2Sphere(theta, phi):
    return np.array([[r**2, 0], [0, r**2*np.sin(theta)**2]])


# Schwarzschild with G = c = 1
def metricSchwarzschild(t, rad, theta, phi):
    return np.diag([1-rs/rad, -1/(1-rs/rad), -rad**2, -rad**2*np.sin(theta)**2])


# Schwarzschild metric with Gullstrand–Painlevé coordinates with G = c = 1
def metricSGP(t, rad, theta, phi):
    m = np.diag([1-rs/rad, -1, -rad**2, -rad**2*np.sin(theta)**2])
    m[0][1] = -1*np.sqrt(rs/rad)
    m[1][0] = -1*np.sqrt(rs/rad)
    return m


global metric
metric = metricSchwarzschild


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
                tot += riemann[sigma][mu][nu][sigma]

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
# derivative with respect to s. This function is intended to be used to test
# if a particular set of coordinates and derivatives corresponds to a null
# geodesic.
def nullTest(y):
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
    assert abs(nullTest(y_0)) < 10**-10

    nullCheck = 1
    while nullCheck > 10**-10:
        tvals, yvals, numStepsTaken = compute_geodesic(s_0, y_0, stop, tol=tol)

        nullCheck = abs(nullTest(yvals[-1]))
        tol /= 10

    return tvals, yvals, numStepsTaken


if __name__ == "__main__":
    
    args = (0, 3, np.pi/2, 0)
    g = metric(*args)
    # value of dt/ds = sqrt(-(g[i][i]*vel[i])/g[0][0]) for i=1,2,3,... for a DIAGONAL metric
    dtds = np.sqrt(-(g[3][3])/g[0][0])
    y_0 = [args[0], dtds, args[1], 0, args[2], 0, args[3], 1]
    print(nullTest(y_0))

    # Schwarschild coordinates photosphere
    tvals, yvals = compute_geodesic(0, y_0, lambda s, y, N: s > 50 or y[2] < 1.1*rs or y[2] > 10, tol=1.0E-10)
    print(nullTest(yvals[len(yvals)//2]))
    print(nullTest(yvals[-1]))
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='polar')
    #c = ax.scatter(yvals[:, 6], yvals[:, 2])
    plt.polar(yvals[:, 6], yvals[:, 2])
    plt.show()
