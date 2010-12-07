import tensorial_kernel

if __name__=='__main__':

    import numpy as np

    D = [64, 64, 34, 4]
    # D = [276,10,10]
    D_X = [5] + D
    X = np.arange(np.prod(D_X)).reshape(*D_X)
    D_Y = [6] + D
    Y = np.arange(np.prod(D_Y)).reshape(*D_Y) + 1

    mlSVD = tensorial_kernel.multilinear_SVD(X[0])
    print "Projection Frobenius norm of ||X[0] - X[0]|| =",
    print tensorial_kernel.compute_projection_frobenius_norm(mlSVD, mlSVD)
    print
    
    mlSVD1 = tensorial_kernel.multilinear_SVD(X[1])
    print "Projection Frobenius norm of ||X[0] - X[1]|| =",
    print tensorial_kernel.compute_projection_frobenius_norm(mlSVD1, mlSVD)
    print
