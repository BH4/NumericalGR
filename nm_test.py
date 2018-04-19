import numpy as np
from numericalMethods import derivative, RK4, initialValueSolution
import unittest


class TestDerivative(unittest.TestCase):

    def test_simple(self):
        f = lambda x: np.sin(x)
        xvals = np.linspace(0, 2*np.pi, 1000)
        solution = np.cos(xvals)
        returnValue = [derivative(f, 0, x) for x in xvals]
        maxError = max(returnValue - solution)

        self.assertTrue(maxError < 10**-10)

    def test_complex(self):
        f = lambda x: np.exp(np.cos(1.0/x))/np.tan(x)
        xvals = np.linspace(.1, np.pi/2, 1000)
        df = lambda x: (np.sin(1.0/x)*np.exp(np.cos(1.0/x))/np.tan(x))/x**2 - np.exp(np.cos(1.0/x))/np.sin(x)**2
        solution = df(xvals)
        returnValue = [derivative(f, 0, x) for x in xvals]
        maxError = max(returnValue - solution)

        self.assertTrue(maxError < 10**-2)

    def test_multiArgument(self):
        f = lambda w, x, y, z: w**2*np.sin(x)*np.cos(y)*np.exp(z)
        wvals = np.linspace(-20, 20, 1000)
        x = 5
        y = 21
        z = -35

        solution = 2*wvals*np.sin(x)*np.cos(y)*np.exp(z)
        returnValue = [derivative(f, 0, w, x, y, z) for w in wvals]
        maxError = max(returnValue - solution)

        self.assertTrue(maxError < 10**-15)


class TestRK4(unittest.TestCase):

    def test_simple(self):
        f = lambda t, y: np.cos(t)
        tvals = np.linspace(0, 2*np.pi, 5)
        h = 1.0E-3

        returnValue = [RK4(t_0, np.sin(t_0), f, h=h)[1] for t_0 in tvals]
        solution = np.sin(tvals + h)
        maxError = max(returnValue - solution)

        self.assertTrue(maxError < 10**-10)


class TestInitValueSolver(unittest.TestCase):

    def test_simple(self):
        t = 0
        y = np.array([0, 1])
        f = lambda t, y: np.array([np.cos(t), -1*np.sin(t)])
        tol = 1.0E-6

        tvals, yvals = initialValueSolution(t, y, f, lambda t, y: t > 100, tol=tol)

        maxError = max(abs(np.sin(tvals) - yvals[:, 0]))

        # The worst error can be a little larger than the tolerance the way it
        # is currently written.
        self.assertTrue(maxError < 10*tol)


if __name__ == '__main__':
    unittest.main()
