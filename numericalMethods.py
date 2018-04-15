import numpy as np
import matplotlib.pyplot as plt


def derivative(f, comp, *args, h=1.0E-3):
    coef = [-1.0/12, 2.0/3, 0, -2.0/3, 1.0/12]

    m = (len(coef)-1)//2
    diff = np.zeros(len(args))
    diff[comp] = h

    tot = 0
    for i in range(m):
        tot += coef[i]*f(*(np.array(args)+(m-i)*diff))
        tot += coef[-1-i]*f(*(np.array(args)-(m-i)*diff))

    return tot/h


def testDeriv(f, df, comp, *args):
    hvals = np.logspace(-12, 0, 1000)

    diff = []
    for h in hvals:
        numDf = derivative(f, comp, *args, h=h)
        diff.append(abs(df(*args) - numDf))

    plt.loglog(hvals, diff)
    plt.show()


# y is the function we would like to approximate (can be a vector) and f is the
# function which given t and y calculates the derivative of y with respect to t
# This function takes one step of size h using the RK4 method.
def RK4(t_0, y_0, f, h=1.0E-3):
    t = t_0
    y = y_0

    k1 = f(t, y)
    k2 = f(t+h/2.0, y+h*k1/2.0)
    k3 = f(t+h/2.0, y+h*k2/2.0)
    k4 = f(t+h, y+h*k3)

    t += h
    y += (h/6.0)*(k1+2*k2+2*k3+k4)

    return t, y


# Given initial value problem returns the value of y evolved from times t_0
# until stop(t, y) return True
# May not use constant time steps.
def initialValueSolution(t_0, y_0, f, stop):
    t = t_0
    y = y_0

    h = 1.0E-3

    tvals = [t]
    yvals = [np.copy(y)]
    while not stop(t, y):
        t, y = RK4(t, y, f, h=h)
        tvals.append(t)
        yvals.append(np.copy(y))

    tvals = np.array(tvals)
    yvals = np.array(yvals)

    return tvals, yvals


if __name__ == '__main__':
    t = 0
    y = np.array([27.5, -5.3])
    f = lambda t, y: np.array([y[1], 9.8])

    tvals, yvals = initialValueSolution(t, y, f, lambda t, y: t > 10)
    print(tvals[-1])
    print(yvals[-1])

    plt.plot(tvals, yvals[:, 0])
    plt.show()
