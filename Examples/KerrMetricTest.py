import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import GeneralGRcode as GR


def SchwarschildLimit():
    metric = GR.metricKerr
    GR.rs = 1.0
    GR.a = 0.0

    args = (0, (3/2)*GR.rs, np.pi/2, 0)
    g = metric(*args)
    dtds = np.sqrt(-(g[3][3])/g[0][0])
    y_0 = [args[0], dtds, args[1], 0, args[2], 0, args[3], 1]

    tvals, yvals = GR.compute_null_geodesic(0, y_0, lambda s, y, N: s > 10 or y[2] < 1.1*GR.rs or y[2] > 10, tol=1.0E-14)

    rad = yvals[:, 2]
    phi = yvals[:, 6]
    x, y = GR.polarToRectangular(rad, phi)

    fig = plt.Figure()
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 5
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size

    plt.plot(x, y)
    plt.show()





if __name__ == "__main__":
    GR.metric = GR.metricKerr
    GR.rs = 3.0
    GR.a = 1.0

    args = (0, (10)*GR.rs, np.pi/2, 0)
    y_0 = [args[0], 1, args[1], 0, args[2], 0, args[3], 0]

    tvals, yvals = GR.compute_geodesic(0, y_0, lambda s, y, N: y[2] < 1.1*GR.rs or y[2] > 20*GR.rs, tol=1.0E-10)

    rad = yvals[:, 2]
    phi = yvals[:, 6]
    x, y = GR.polarToRectangular(rad, phi)

    fig = plt.Figure()
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 5
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size

    plt.plot(x, y)
    plt.show()
