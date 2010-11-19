"""Compute tensorial kernel.

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


def unfold(A):
    """Unfold tensor A along first axis.
    """
    return A.reshape(A.shape[0], np.prod(A.shape[1:]))


def multilinear_SVD(A):
    """Compute multilinear SVD.
    """
    # assert(len(A.shape)<=3) # No-one worked out unfolding for higher that 3 dim, it should work but... TEST!
    U_s_Vh_list = []
    for i in range(len(A.shape)):
        B = A.swapaxes(0,i) # swap first and i-th axis to prepare tensor for mode-i unfolding
        B = unfold(B)
        U_s_Vh_list.append(np.linalg.linalg.svd(B, full_matrices=False))
    return U_s_Vh_list


def compute_projection_frobenius_norm(U_s_Vh_list_X, U_s_Vh_list_Y):
    """Compute the projection Frobenius norm between two tensors.
    """
    projection_frobenius_norm = 0.0
    for i in range(len(U_s_Vh_list_X)):
        U_X, s_X, Vh_X = U_s_Vh_list_X[i]
        U_Y, s_Y, Vh_Y = U_s_Vh_list_Y[i]
        if U_X.shape[0] > Vh_X.shape[1]:
            W_X = U_X
            W_Y = W_X
        else:
            W_X = Vh_X.T
            W_Y = Vh_Y.T
        projection_frobenius_norm += np.linalg.norm(np.dot(W_X,W_X.T)-np.dot(W_Y,W_Y.T) , ord='fro')
    return projection_frobenius_norm


def compute_projection_frobenius_norm_train(X):
    """Compute projection Frobenius norm matrix between among all
    pairs of tensors within trainset X.

    X = trainset, one example (tensor) per row.
    """
    # Pre-compute the list of SVD decomposition for each element of
    # the training set. The result is stored into a list of lists.
    U_s_Vh_list_list = []
    for i in range(X.shape[0]):
        U_s_Vh_list_list.append(multilinear_SVD(X[i]))

    projection_frobenius_norm = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            projection_frobenius_norm[i,j] += compute_projection_frobenius_norm(U_s_Vh_list_list[i], U_s_Vh_list_list[j])
    # copy diag sup on dig inf
    projection_frobenius_norm += projection_frobenius_norm.T
    return projection_frobenius_norm


def compute_projection_frobenius_norm_test(X, Y):
    """Compute projection Frobenius norm matrix between among all
    pairs of tensors within trainset X and testset Y.

    X = trainset, one example (tensor) per row.
    X = testset.
    """
    # Pre-compute the list of SVD decomposition for each element of
    # the training set. The result is stored into a list of lists.
    U_s_Vh_list_list_X = []
    for i in range(X.shape[0]):
        U_s_Vh_list_list_X.append(multilinear_SVD(X[i]))

    U_s_Vh_list_list_Y = []
    for i in range(Y.shape[0]):
        U_s_Vh_list_list_Y.append(multilinear_SVD(Y[i]))

    projection_frobenius_norm = np.zeros((X.shape[0],Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            projection_frobenius_norm[i,j] += compute_projection_frobenius_norm(U_s_Vh_list_list_X[i], U_s_Vh_list_list_Y[j])
    return projection_frobenius_norm


def compute_tensorial_kernel(X, Y=None, sigma2=1.0):
    """Compute tensorial RBF kernel between two datasets.
    """
    if Y is None:
        projection_frobenius_norm_matrix = compute_projection_frobenius_norm_train(X)
    else:
        projection_frobenius_norm_matrix = compute_projection_frobenius_norm_test(X, Y)
    return np.exp(-1.0/sigma2 * projection_frobenius_norm_matrix)

                    
if __name__=='__main__':

    D = [5,4,3,2]
    D_X = [10] + D
    X = np.arange(np.prod(D_X)).reshape(*D_X)
    D_Y = [5] + D
    Y = np.arange(np.prod(D_Y)).reshape(*D_Y) + 1

    print "X[0]:"
    print X[0]
    print "shape:", X[0].shape
    print
    print "Unfolding X[0] (mode-0):"
    tmp = unfold(X[0])
    print tmp
    print "shape:", tmp.shape
    print
    print "Multilinear SVD of X[0]:"
    mlSVD = multilinear_SVD(X[0])
    print mlSVD
    print "length:", len(mlSVD)
    print
    print "Projection Frobenius norm of ||X[0] - X[0]|| =",
    print compute_projection_frobenius_norm(mlSVD, mlSVD)
    print
    mlSVD1 = multilinear_SVD(X[1])
    print "Projection Frobenius norm of ||X[0] - X[1]|| =",
    print compute_projection_frobenius_norm(mlSVD1, mlSVD)
    print
    print "Projection Frobenius norm of all trainset X against X:"
    tmp = compute_projection_frobenius_norm_train(X)
    print tmp
    print "shape:", tmp.shape
    print
    print "Projection Frobenius norm of all trainset X against testset Y:"
    tmp = compute_projection_frobenius_norm_test(X, Y)
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
