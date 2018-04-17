import numpy as np
import matplotlib.pyplot as plt
from numericalMethods import derivative, initialValueSolution

# define the global values which are used in the metric

"""
# 2 Sphere surface
global r
r = 1


def metric(theta, phi):
    return np.array([[r**2, 0], [0, r**2*np.sin(theta)**2]])
"""

# Schwarzschild with G = c = 1
global rs
rs = 2.0


def metric(t, r, theta, phi):
    return np.diag([1-rs/r, -1/(1-rs/r), -r**2, -r**2*np.sin(theta)**2])

"""
# Schwarzschild metric with Gullstrand–Painlevé coordinates with G = c = 1
global rs
rs = 2


def metric(t, r, theta, phi):
    m = np.diag([1-rs/r, -1, -r**2, -r**2*np.sin(theta)**2])
    m[0][1] = -1*np.sqrt(rs/r)
    m[1][0] = -1*np.sqrt(rs/r)
    return m
"""

def compute_christoffel(*args):
    d = len(args)  # number of dimensions

    csymbols = np.zeros((d, d, d))  # first index is the upper index

    dg = []
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
    dgamma = []
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


# y should be a vector corresponding to each coordinate with its derivative
# following (x, dx, y, dy, ...)
def compute_geodesic(s_0, y_0, stop):
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

    return initialValueSolution(s_0, y_0, dyds, stop)


if __name__ == "__main__":
    t = 0
    r = 5
    theta = np.pi/5
    phi = 0
    args = [t, r, theta, phi]
    """
    print(compute_christoffel(*args))
    print(compute_riemann_christoffel(*args))
    print(compute_riccitensor(*args))
    print(compute_ricciscalar(*args))
    """

    """
    for i in range(5):
        r = 2**i
        plt.scatter(1.0/r**2, compute_ricciscalar(np.pi/2, 0))

    plt.ylim(-3, 1)
    plt.show()
    """
    """
    tvals, yvals = compute_geodesic(0, [np.pi/2, 1, 0, 0], lambda t, y: y[0] > 3*np.pi/2)
    plt.plot(yvals[:, 2], yvals[:, 0])
    plt.xlim(-np.pi, np.pi)
    plt.ylim(np.pi/2, 3*np.pi/2)
    plt.show()
    """
    """
    tvals, yvals = compute_geodesic(0, [.00001, 0, 0, 1], lambda t, y: y[2] > 2*np.pi)
    plt.plot(yvals[:, 2], yvals[:, 0])
    #plt.xlim(0, 2*np.pi)
    #plt.ylim(0, np.pi)
    plt.show()
    """
    """
    error1 = []
    error2 = []
    error3 = []
    for i in np.linspace(np.pi-1, np.pi+1, 1000):
        csymbols = compute_christoffel(i, 542165)

        error1.append(np.cos(i)/np.sin(i) - csymbols[1][1][0])
        error2.append(np.cos(i)/np.sin(i) - csymbols[1][0][1])
        error3.append(-np.sin(i)*np.cos(i) - csymbols[0][1][1])

    print(max(abs(np.array(error1))))
    print(max(abs(np.array(error2))))
    print(max(abs(np.array(error3))))
    """

    # Schwarschild coordinates orbit BH
    tvals, yvals = compute_geodesic(0, [0, 1, 5, -.5, np.pi/2, 0, 0, .140275], lambda s, y: y[2] < 1.1*rs or y[2] > 10)
    plt.polar(yvals[:, 6], yvals[:, 2])
    plt.show()

    """
    # GP coordinates falling into BH
    tvals, yvals = compute_geodesic(0, [0, 0, 3, -.01, np.pi/2, 0, 0, .00981], lambda s, y: y[2] < 0.1*rs or y[2] > 10)
    plt.polar(yvals[:, 6], yvals[:, 2])
    plt.show()
    """
