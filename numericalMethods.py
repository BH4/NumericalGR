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
    y = np.copy(y_0)

    k1 = f(t, y)
    k2 = f(t+h/2.0, y+h*k1/2.0)
    k3 = f(t+h/2.0, y+h*k2/2.0)
    k4 = f(t+h, y+h*k3)

    t += h
    y += (h/6.0)*(k1+2*k2+2*k3+k4)

    return t, y


# Given initial value problem returns the value of y evolved from times t_0
# until stop(t, y) return True.
# Uses adaptive time steps (Taken from Wikipedia page on adaptive step size)
def initialValueSolution(t_0, y_0, f, stop):
    t = t_0
    y = np.array(y_0).astype(float)

    tol = 1.0E-5
    h = 1.0E-2

    tvals = [t]
    yvals = [y]
    while not stop(t, y):
        tFull, yFull = RK4(t, y, f, h=h)
        tHalf, yHalf = RK4(t, y, f, h=h/2.0)
        tHalf, yHalf = RK4(tHalf, yHalf, f, h=h/2.0)

        error = max(abs(yFull-yHalf))
        if error < tol:
            tvals.append(tFull)
            yvals.append(yFull)
            t = tFull
            y = yFull

        if error == 0:
            print("0 error?")
            h *= 2
        else:
            # .9 is safety factor to make sure we get desired accuracy,
            # .3 is minimum decrease in h, 2 is maximum increase
            h = .9*h*min(max(tol/error, .3), 2)

    tvals = np.array(tvals)
    yvals = np.array(yvals)

    return tvals, yvals


if __name__ == '__main__':
    t = 0
    y = np.array([0, 1])
    f = lambda t, y: np.array([np.cos(t), -1*np.sin(t)])

    tvals, yvals = initialValueSolution(t, y, f, lambda t, y: t > 100)

    print(max(abs(np.sin(tvals) - yvals[:, 0])))
