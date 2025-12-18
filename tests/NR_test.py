import unittest
from HydroCore import SimulationState
from MHD import SodShockInitialization, ImplosionInitialization
from HelperFunctions import WhichVar, WhichRegime
from numpy.testing import assert_allclose
import pickle as pkl
import os

class TestNR(unittest.TestCase):

    # def test_conversion_newtonian(self):
    #     state = SodShockInitialization(1.0,0.0,1.0, 0.1, 0.0, 0.125, N_cells=1000, t_max=0.2, relativistic=WhichRegime.NEWTONIAN)
    #     U = state.primitive_to_conservative(state.primitive_previous)
    #     U_padded = state.pad_unweighted_array(U, WhichVar.CONSERVATIVE)
    #     W = state.conservative_to_primitive(U_padded)
    #     assert_allclose(state.primitive_previous, W)

    # def test_conversion_SR(self):
    #     state = SodShockInitialization(1.0,0.0,1.0, 0.1, 0.0, 0.125, N_cells=1000, t_max=0.2, relativistic=WhichRegime.RELATIVITY)
    #     U = state.primitive_to_conservative(state.primitive_previous)
    #     U_padded = state.pad_unweighted_array(U, WhichVar.CONSERVATIVE)
    #     W = state.conservative_to_primitive(U_padded)
    #     assert_allclose(state.primitive_previous, W)

    # def test_conversion_SR2D(self):
    #     state =  ImplosionInitialization(t_max=3, regime=WhichRegime.RELATIVITY)
    #     U = state.primitive_to_conservative(state.primitive_previous)
    #     U_padded = state.pad_unweighted_array(U, WhichVar.CONSERVATIVE)
    #     W = state.conservative_to_primitive(U_padded)
    #     assert_allclose(state.primitive_previous, W)

    def test_conversion_SR(self):
        print(os.getcwd())
        state = SodShockInitialization(1.0,0.0,1.0, 0.1, 0.0, 0.125, N_cells=5, t_max=0.2, relativistic=WhichRegime.RELATIVITY)
        f = open("./tests/Prim.pkl", "wb")
        pkl.dump(state.primitive_previous, f )
        f.close()
        U = state.primitive_to_conservative(state.primitive_previous)
        state.update()
        print(U,state.U)
        f = open("./tests/Cons.pkl", "wb")
        pkl.dump(U, f)
        f.close()

if __name__ == '__main__':
    unittest.main()