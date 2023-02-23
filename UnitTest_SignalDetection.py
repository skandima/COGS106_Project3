import unittest
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from scipy.stats import norm
from SignalDetection_Refactored import SignalDetection

class TestSignalDetection(unittest.TestCase):
    def test_d_prime_zero(self):
        sd   = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_d_prime_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_criterion_zero(self):
        sd   = SignalDetection(5, 5, 5, 5)
        # Calculate expected criterion        
        expected = 0
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_criterion_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        # Calculate expected criterion        
        expected = -0.463918426665941
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_attempt_corruption(self):
        sd = SignalDetection(10, 5, 2, 15)

        
        sd._SignalDetection__hits = 15
        sd._SignalDetection__misses = 10
        sd._SignalDetection__falseAlarms = 15
        sd._SignalDetection__correctRejections = 5

        assert sd.hit_rate() == 15 / (15 + 10)
        assert sd.falseAlarm_rate() == 15 / (15 + 5)
        assert abs(sd.d_prime() + 0.421142647060282) < 0.001
        assert abs(sd.criterion() + 0.463918426665941) < 0.001

    

if __name__ == '__main__':
    unittest.main()