"""
Procession of the perihelion of mercury
aphelion = 23.64 million times rs for the sun
perihelion = 15.58 million times rs for the sun

Code starts with planet at phi=0 at the aphelion and I found a value for
dphi/ds which gives a perihelion near the actual value.

When using an aphelion closer to rs or an orbit with a much larger eccentricity
the procession effect is much easier to see. Here it is mostly not observable.
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import GeneralGRcode as GR


if __name__ == "__main__":
    rs = 10**-6
    GR.metric = GR.metricSchwarzschild(rs)

    args = (0, 23.64, np.pi/2, 0)
    g = GR.metric(*args)
    y = .000005483
    dtds = np.sqrt((-1-g[3][3]*y**2)/g[0][0])
    y_0 = [args[0], dtds, args[1], 0, args[2], 0, args[3], y]
    numOrbits = 10
    tvals, yvals = GR.compute_timelike_geodesic(0, y_0, lambda s, y, N: y[6]/(2*np.pi) > numOrbits, tol=1.0E-5)

    rad = yvals[:, 2]
    phi = yvals[:, 6]
    # numOrbits = phi[-1]/(2*np.pi)
    # print("Number of orbits:", numOrbits)
    print("Perihelion", min(yvals[:, 2]))

    n = []
    for i in range(numOrbits):
        n.append(len([x for x in phi if x < (i+1)*2*np.pi]))
    print(n)

    # plt.polar(yvals[:, 6], yvals[:, 2])
    # plt.show()
    x, y = GR.polarToRectangular(rad, phi)
    prev = 0
    for ni in n:
        plt.plot(x[prev:ni+1], y[prev:ni+1])
        prev = ni
    plt.show()
