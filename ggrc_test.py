import numpy as np
import GeneralGRcode as GR
import unittest
from random import uniform


class TestChristoffel(unittest.TestCase):

    def test_2Sphere(self):
        """
        Gamma(1, 1, 0) = Gamma(1, 0, 1) = cos(theta)/sin(theta)
        Gamma(0, 1, 1) = -sin(theta)*cos(theta)
        All other Christoffel symbols are zero.
        """
        GR.metric = GR.metric2Sphere

        maxError = 0.0

        # Accuracy lost near the poles
        for theta in np.linspace(.001, np.pi-.001, 1000):
            csymbols = GR.compute_christoffel(theta, 0)

            e1 = 1.0/np.tan(theta) - csymbols[1][1][0]
            e2 = 1.0/np.tan(theta) - csymbols[1][0][1]
            e3 = -np.sin(theta)*np.cos(theta) - csymbols[0][1][1]

            maxError = max([maxError, e1, e2, e3])

        self.assertTrue(maxError < 10**-9)


class TestRicciScalar(unittest.TestCase):

    def test_rand_ricciscalar(self):
        """
        R = -2/r**2
        """
        GR.metric = GR.metric2Sphere

        for i in np.linspace(.1, 10, 100):
            GR.r = i
            # accuracy is lost near the poles.
            for theta in [.25, 1, np.pi-.25]:
                v = GR.compute_ricciscalar(theta, 0)

            self.assertEqual(round(v, 5), round(2/GR.r**2, 5))


if __name__ == '__main__':
    unittest.main()
