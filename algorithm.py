import numpy as np



def aliverti_x_til(X, k, Z):

    v, sigma, uT = np.linalg.svd(X, full_matrices=False)

    # By Eckhart-Young, an aproximation matrix with minimized F-norm can be
    # found from the sigma matrix of a rank-k SVD by keeping the top k values
    # in sigma and setting the rest to zero (hard thresholding)

    # this also takes care of the rank-reduction in x_tilde ;)

    sigma = np.diag(sigma)
    if k < len(sigma):
        for i in range(k, len(sigma)):
            sigma[i][i] = 0

    ztz = np.dot(Z.transpose(), Z)
    ztzInv = 1/ztz
    temp = np.dot(Z, ztzInv)
    pZ = np.dot(temp, Z.transpose())

    n = v.shape[0]
    i = np.eye(n)

    xTilde = np.subtract(i, pZ) @ v @ sigma @ uT

    return xTilde



def new_x_til(X, k, Z):

    print("Under Construction")
