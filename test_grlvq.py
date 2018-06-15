import unittest
import numpy as np
from grlvq import Grlvq

class TestGrlvq(unittest.TestCase):
    def test_relevance_dist(self):
        self.grlvq = Grlvq()
        self.grlvq.relevance = np.array([0.8,0.2])
        self.assertEqual(self.grlvq.relevance_dist(np.array([1,1]),np.array([2,3])),np.array([1.6]))

if __name__ == '__main__':
    unittest.main()