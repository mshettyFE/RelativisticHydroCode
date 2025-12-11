import unittest
from HydroCore import SimulationState
from MHD import SodShockInitialization

class TestNR(unittest.TestCase):

    def test_conversion(self):
        state = SodShockInitialization(1.0,0.0,1.0, 0.1, 0.0, 0.125, N_cells=1000, t_max=0.2) 
        self.assertEqual(self.state.current_time, 0)

if __name__ == '__main__':
    unittest.main()