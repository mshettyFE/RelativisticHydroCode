import unittest
from HydroCore import SimulationState
from MHD import SodShockInitialization, ImplosionInitialization
from HelperFunctions import WhichVar, WhichRegime
from numpy.testing import assert_allclose

class TestNR(unittest.TestCase):

    def test_conversion_newtonian(self):
        state = SodShockInitialization(1.0,0.0,1.0, 0.1, 0.0, 0.125, N_cells=1000, t_max=0.2, relativistic=WhichRegime.NEWTONIAN)
        U = state.primitive_to_conservative(state.primitive_previous)
        U_padded = state.pad_unweighted_array(U, WhichVar.CONSERVATIVE)
        W = state.conservative_to_primitive(U_padded)
        assert_allclose(state.primitive_previous, W)

    def test_conversion_SR(self):
        state = SodShockInitialization(1.0,0.0,1.0, 0.1, 0.0, 0.125, N_cells=1000, t_max=0.2, relativistic=WhichRegime.RELATIVITY)
        U = state.primitive_to_conservative(state.primitive_previous)
        U_padded = state.pad_unweighted_array(U, WhichVar.CONSERVATIVE)
        W = state.conservative_to_primitive(U_padded)
        assert_allclose(state.primitive_previous, W)

    def test_conversion_SR2D(self):
        state =  ImplosionInitialization(t_max=3, regime=WhichRegime.RELATIVITY)
        U = state.primitive_to_conservative(state.primitive_previous)
        U_padded = state.pad_unweighted_array(U, WhichVar.CONSERVATIVE)
        W = state.conservative_to_primitive(U_padded)
        assert_allclose(state.primitive_previous, W)

if __name__ == '__main__':
    unittest.main()