import numpy as np
import numpy.linalg as LA
from numpy.random import multivariate_normal as rmvn

# Author: Aviv Navon
# Email: avivnav@gmail.com
#
# Utility functions for running and evaluating the simulations presented in the paper
# "Capturing Between-Tasks Covariance and Similarities Using Multivariate Linear Mixed Models"


def get_c_matrix(q, rho=.5):
    """
    Generate equicorrelation matrix

    :param q: dimension
    :param rho: correlation parameter in [0,1)
    :return:
    """
    assert((rho >= 0) & (rho < 1))
    return np.matrix(np.eye(q) + rho * (np.ones((q, q)) - np.eye(q)))


def get_u_matrix(q):
    """
    Generate orthogonal matrix U s.t C=UDU^T and D is diag.

    :param q: dimension
    :return:
    """
    C = get_c_matrix(q)
    U, D, V = LA.svd(C)
    return np.matrix(U)


def ar_cov(dim, corr):
    """
    Generate AR(1) covariance matrix

    :param dim: dimension
    :param corr: correlation parameter in [0,1)
    """
    idxs = np.arange(1, dim + 1)
    AR = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            AR[i,j] = np.power(corr, np.abs(idxs[i] - idxs[j]))
    return AR


def fgn_cov(dim, H=.95):
    """
    Generate Fractional Gaussian Noise covariance matrix

    :param dim: dimension
    :param H: Hurst parameter in (0,1)
    """
    cov = np.empty((dim, dim))
    H2 = 2*H

    for i in range(dim):
        for j in range(dim):
            if i == j:
                cov[i, j] = 1
            else:
                abs_diff = np.abs(i-j)
                cov[i, j] = .5 * (((abs_diff + 1) ** H2) - (2 * abs_diff ** H2) + ((abs_diff-1) ** H2))
    return cov


def model_error(B, B_hat, Sigma_X):
    """
    Compute Model Error, given by tr[(B-B_hat)^T*Sigma_X*(B-B_hat)]

    :param B: true coefficient matrix
    :param B_hat: estimated coefficient matrix
    :param Sigma_X: covariace matrix for predictors
    """
    return np.trace((B_hat - B).T * Sigma_X * (B_hat - B))


def generate_err_cov(q, err_cov_type, corr=None):
    """
    Generate covariance matrix for error terms

    :param q : dim.
    :param err_cov_type: error covariance type. One of ['ar', 'equi', 'fgn', 'identity']
    :param corr: correlation coefficient, for ['ar', 'equi'] only.
    """
    assert err_cov_type in ['ar', 'equi', 'fgn', 'identity']

    if err_cov_type == 'ar':
        Sigma = ar_cov(q, corr)

    elif err_cov_type == 'equi':
        Sigma = np.matrix(np.eye(q) + corr * (np.ones((q, q)) - np.eye(q)))

    elif err_cov_type == 'fgn':
        Sigma = fgn_cov(q)

    else:
        Sigma = np.eye(q)

    return Sigma


def generate_data(
        err_cov_type,
        rho,
        n,
        p,
        q,
        sigma,
        corr_x,
        sigma_err,
        g_sparse_level,
        sparse_level,
        err_corr=None
):
    """
    Generate data for simulations

    :param err_cov_type: error covariance type. One of ['ar', 'equi', 'fgn', 'identity']
    :param rho: correlation coefficient for parameters
    :param n: observation dim.
    :param p: predictors dim.
    :param q: tasks dim.
    :param sigma: sd for coefficients
    :param corr_x: correlation between the X matrix row elements
    :param sigma_err: sd for errors
    :param g_sparse_level: group sparse level, in [0,1]
    :param sparse_level: sparse level, in [0,1]
    :param err_corr: correlation coefficient, for ['ar', 'equi'] only.
    """
    Sigma_X = ar_cov(p, corr_x)                                                 # Sigma_X
    mu_X = np.zeros(p)                                                          # mu_X
    X = np.matrix([rmvn(mu_X, Sigma_X) for _ in range(n)])                      # realization of X

    Sigma_B = (sigma ** 2) * (np.eye(q) + rho * (np.ones((q, q)) - np.eye(q)))  # this is sigma^2 * C
    mu_B = np.zeros(q)                                                          # mu_B
    B = np.matrix([rmvn(mu_B, Sigma_B) for i in range(p)])                      # realization of B
    B = np.diag(np.random.binomial(1, 1 - g_sparse_level, size=p)) * B          # generate group sparsity
    K = np.random.binomial(1, 1 - sparse_level, size=p * q).reshape((p, q))     # for element-wise sparsity
    B = np.matrix(np.multiply(B, K))                                            # generate element-wise sparsity

    mu = np.zeros(q)  # mu_E
    U = get_u_matrix(q)  # U s.t C=UDU^T and D is diag.
    Sigma = (sigma_err**2) * generate_err_cov(q, err_cov_type=err_cov_type, corr=err_corr)  # Sigma_E
    Sigma = U * Sigma * U.transpose()  # transformed covariance matrix
    E = np.matrix([rmvn(mu, Sigma) for _ in range(n)])  # realization of E

    Y = X * B + E  # realization of Y
    return X, Y, B, Sigma, Sigma_X


def generate_data_(
        err_cov_type,
        n,
        p,
        q,
        corr_x,
        sigma_err,
        g_sparse_level,
        sparse_level,
        err_corr=None,
        **kwargs
):
    """
    Generate data for simulations

    :param err_cov_type: error covariance type. One of ['ar', 'equi', 'fgn', 'identity']
    :param rho: correlation coefficient for parameters
    :param n: observation dim.
    :param p: predictors dim.
    :param q: tasks dim.
    :param sigma: sd for coefficients
    :param corr_x: correlation between the X matrix row elements
    :param sigma_err: sd for errors
    :param g_sparse_level: group sparse level, in [0,1]
    :param sparse_level: sparse level, in [0,1]
    :param err_corr: correlation coefficient, for ['ar', 'equi'] only.
    """
    Sigma_X = ar_cov(p, corr_x)                                                 # Sigma_X
    mu_X = np.zeros(p)                                                          # mu_X
    X = np.matrix([rmvn(mu_X, Sigma_X) for _ in range(n)])                      # realization of X

    # Sigma_B = (sigma ** 2) * (np.eye(q) + rho * (np.ones((q, q)) - np.eye(q)))  # this is sigma^2 * C
    # mu_B = np.zeros(q)                                                          # mu_B
    # B = np.matrix([rmvn(mu_B, Sigma_B) for i in range(p)])                      # realization of B
    B = np.matrix(np.random.uniform(-1, 1, (p, q)))
    B = np.diag(np.random.binomial(1, 1 - g_sparse_level, size=p)) * B          # generate group sparsity
    K = np.random.binomial(1, 1 - sparse_level, size=p * q).reshape((p, q))     # for element-wise sparsity
    B = np.matrix(np.multiply(B, K))                                            # generate element-wise sparsity

    mu = np.zeros(q)  # mu_E
    U = get_u_matrix(q)  # U s.t C=UDU^T and D is diag.
    Sigma = (sigma_err**2) * generate_err_cov(q, err_cov_type=err_cov_type, corr=err_corr)  # Sigma_E
    Sigma = U * Sigma * U.transpose()  # transformed covariance matrix
    E = np.matrix([rmvn(mu, Sigma) for _ in range(n)])  # realization of E

    Y = X * B + E  # realization of Y
    return X, Y, B, Sigma, Sigma_X
