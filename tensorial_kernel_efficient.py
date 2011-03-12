"""Compute tensorial kernel. More efficient version.

Copyright Emanuele Olivetti & Marco Signoretto, 2010.

Based on:

@unpublished{signoretto2010kernel,
    author = {Signoretto, Marco and De Lathauwer, Lieven and Suykens, Johan A. K.},
    citeulike-article-id = {8248002},
    keywords = {tensor},
    posted-at = {2010-11-15 14:15:37},
    priority = {2},
    title = {{A Kernel-based Framework to Tensorial Data Analysis}},
    year = {2010}
}
"""

import numpy as np
import sys

def unfold(A, mode=0):
    """Unfold tensor A along 'mode' axis.
    """
    assert(mode < len(A.shape))
    A = A.swapaxes(0,mode) # swap first and mode-th axis to prepare tensor for mode-'mode' unfolding
    return A.reshape(A.shape[0], np.prod(A.shape[1:]))


def precompute_multilinearSVD(A):
    """Compute multilinear SVD.
    """
    precomputed_mSVD_dot = []
    for i in range(len(A.shape)):
        B = unfold(A, mode=i)
        U, s, Vh = np.linalg.linalg.svd(B, full_matrices=False)
        W = Vh.T
        if U.shape[0] > Vh.shape[1]:
            W = U
        precomputed_mSVD_dot.append(W)
    return precomputed_mSVD_dot


def compute_projection_frobenius_norm_efficient(precomputed_mSVD_X, precomputed_mSVD_Y):
    """Compute the projection Frobenius norm between two tensors.
    Scalable implementation.
    """
    projection_frobenius_norm = 0.0
    for i in range(len(precomputed_mSVD_X)):
        # ||A||_F = sqrt(dot(A.T,A)) . Substituting
        # precomputed_mSVD_X[i], precomputed_mSVD_Y[i] after some
        # steps we get: (note that abs() is used to handle numerical
        # instabilities leading to sqrt() of tiny negative numbers.
        Z = np.atleast_2d(np.dot(precomputed_mSVD_X[i].T, precomputed_mSVD_Y[i]))
        projection_frobenius_norm += 2.0 * (Z.shape[0] - np.dot(Z, Z.T).trace())
    return projection_frobenius_norm


def compute_projection_frobenius_norm_train_efficient(X):
    """Compute projection Frobenius norm matrix between among all
    pairs of tensors within trainset X.

    X = trainset, one example (tensor) per row.
    """
    # Pre-compute the list of SVD decomposition for each element of
    # the training set. The result is stored into a list of lists.
    precomputed_mSVD_X = []
    for i in range(X.shape[0]):
        precomputed_mSVD_X.append(precompute_multilinearSVD(X[i]))

    projection_frobenius_norm = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        # print i,
        # sys.stdout.flush()
        for j in range(i+1, X.shape[0]):
            projection_frobenius_norm[i,j] += compute_projection_frobenius_norm_efficient(precomputed_mSVD_X[i], precomputed_mSVD_X[j])
    # copy upper diag on lower diag because matrix must be symmetric
    projection_frobenius_norm += projection_frobenius_norm.T
    return projection_frobenius_norm


def compute_projection_frobenius_norm_test_efficient(X, Y):
    """Compute projection Frobenius norm matrix between among all
    pairs of tensors within trainset X and testset Y.

    X = trainset, one example (tensor) per row.
    Y = testset.
    """
    # Pre-compute the list of SVD decompositions for each element of
    # the training set. The result is stored into a list of lists.
    precomputed_mSVD_X = []
    for i in range(X.shape[0]):
        precomputed_mSVD_X.append(precompute_multilinearSVD(X[i]))

    precomputed_mSVD_Y = []
    for i in range(Y.shape[0]):
        precomputed_mSVD_Y.append(precompute_multilinearSVD(Y[i]))

    projection_frobenius_norm = np.zeros((X.shape[0],Y.shape[0]))
    for i in range(X.shape[0]):
        # print i,
        # print sys.stdout.flush()
        for j in range(Y.shape[0]):
            projection_frobenius_norm[i,j] += compute_projection_frobenius_norm_efficient(precomputed_mSVD_X[i], precomputed_mSVD_Y[j])
    return projection_frobenius_norm


def compute_tensorial_kernel(X, Y=None, sigma2=1.0):
    """Compute tensorial RBF kernel between two datasets.
    """
    if Y is None:
        projection_frobenius_norm_matrix = compute_projection_frobenius_norm_train_efficient(X)
    else:
        projection_frobenius_norm_matrix = compute_projection_frobenius_norm_test_efficient(X, Y)
    return np.exp(-1.0/sigma2 * projection_frobenius_norm_matrix*projection_frobenius_norm_matrix)

                    
if __name__=='__main__':

    D = [4,3,2]
    D_X = [5] + D
    X = np.arange(np.prod(D_X)).reshape(*D_X)
    D_Y = [6] + D
    Y = np.arange(np.prod(D_Y)).reshape(*D_Y) + 1

    print "X[0]:"
    print X[0]
    print "shape:", X[0].shape
    print
    for i in range(len(X[0].shape)):
        print "Unfolding X[0] (mode-%i):" % i
        tmp = unfold(X[0], mode=i)
        print tmp
        print "shape:", tmp.shape
        print
    print "Multilinear SVD of X[0]:"
    mlSVD = precompute_multilinearSVD(X[0])
    print mlSVD
    print "length:", len(mlSVD)
    print
    print "Projection Frobenius norm of ||X[0] - X[0]|| =",
    print compute_projection_frobenius_norm_efficient(mlSVD, mlSVD)
    print
    mlSVD1 = precompute_multilinearSVD(X[1])
    print "Projection Frobenius norm of ||X[0] - X[1]|| =",
    print compute_projection_frobenius_norm_efficient(mlSVD1, mlSVD)
    print
    print "Projection Frobenius norm of all trainset X against X:"
    tmp = compute_projection_frobenius_norm_train_efficient(X)
    print tmp
    print "shape:", tmp.shape
    print
    print "Projection Frobenius norm of all trainset X against testset Y:"
    tmp = compute_projection_frobenius_norm_test_efficient(X, Y)
    print tmp
    print "shape:", tmp.shape
    print
    print "Tensorial Kernel matrix K(X,X):"
    K_X_X = compute_tensorial_kernel(X)
    print K_X_X
    print "shape:", K_X_X.shape
    print
    print "Tensorial Kernel matrix K(X,Y):"
    K_X_Y = compute_tensorial_kernel(X, Y)
    print K_X_Y
    print "shape:", K_X_Y.shape
