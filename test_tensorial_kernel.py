"""Unit testing of tensorial_kernel.py.
"""

import numpy as np
import tensorial_kernel
import unittest


class UnfoldingTest(unittest.TestCase):

    def test_unfolding_size(self):
        """Test size of all the unfoldings of a tensor.
        """
        A = np.zeros((2,3,4))
        A0 = tensorial_kernel.unfold(A, mode=0)
        assert(A0.shape==(2,12))
        A1 = tensorial_kernel.unfold(A, mode=1)
        assert(A1.shape==(3,8))
        A2 = tensorial_kernel.unfold(A, mode=2)
        assert(A2.shape==(4,6))

    def test_unfolding_order(self):
        # A = np.arange(np.prod((2,3,4))).reshape(2,3,4)
        A = np.array([[[ 0,  1,  2,  3],
                       [ 4,  5,  6,  7],
                       [ 8,  9, 10, 11]],
                      [[12, 13, 14, 15],
                       [16, 17, 18, 19],
                       [20, 21, 22, 23]]])

        A0 = tensorial_kernel.unfold(A, mode=0)
        A0_expected = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
                                [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
        assert((A0 == A0_expected).all())
        A1 = tensorial_kernel.unfold(A, mode=1)
        A1_expected = np.array([[ 0,  1,  2,  3, 12, 13, 14, 15],
                                [ 4,  5,  6,  7, 16, 17, 18, 19],
                                [ 8,  9, 10, 11, 20, 21, 22, 23]])
        assert((A1 == A1_expected).all())
        A2 = tensorial_kernel.unfold(A, mode=2)
        A2_expected = np.array([[ 0, 12,  4, 16,  8, 20],
                                [ 1, 13,  5, 17,  9, 21],
                                [ 2, 14,  6, 18, 10, 22],
                                [ 3, 15,  7, 19, 11, 23]])
        assert((A2 == A2_expected).all())


class TestMultilinearSVD(unittest.TestCase):
    def test_MLSVD_shape(self):
        """Given a tensor of dimension D compute multilinear_SVD and
        check the size of each part of the result.
        """
        D = (2,3,4)
        A = np.arange(np.prod(D)).reshape(*D)
        U_s_Vh_list = tensorial_kernel.multilinear_SVD(A)
        assert(len(U_s_Vh_list) == len(D)) # check number of unfoldings
        for i in range(len(D)):
            assert(len(U_s_Vh_list[i]) == 3) # check number of results of each SVD
            U, s, Vh = U_s_Vh_list[i]
            assert(U.shape == (D[i], D[i])) # check shape of U
            assert(s.shape == (D[i],)) # check shape of s
            assert(Vh.shape == (D[i], np.prod(D[:i]+D[i+1:]))) # check shape of Vh

if __name__=='__main__':

    unittest.main()
    
