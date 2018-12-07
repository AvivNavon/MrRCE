# Author: Aviv Navon
# Email: avivnav@gmail.com
#
# This is the implementation for the MrRCE algorithm, described in
# the paper "Capturing Between-Tasks Covariance and Similarities Using Multivariate Linear Mixed Models"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from sklearn.covariance import graph_lasso, empirical_covariance, GraphLassoCV
from inverse_covariance import QuicGraphLassoCV, QuicGraphLasso
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import mahalanobis
from itertools import product

class MrRCE(object):
    """
    Multivariate random Regression with Covariance Estimation (MrRCE)
    
    Note that the random regression eq. is Y=XB+E here, and not Y=Z*Gamma+E as in the paper.
    
    Parametes
    ---------
    max_iter: maximal number of iterations for the MrRCE algorithm.
    lam : regularization parameter for the Graphical Lasso algorithm.
            Only used if self.cv set to False.
    tol_glasso : tolerence value for the Glasso estimation, e.g. we
                 we will keep iterating if the sum of squared changes in Omega
                 in two successive iterations is larger than the tolerance value.
    tol : same as tol_glasso but for (sigma, rho).
    init_B : Initial value for B, one of ['zero', 'ols']
    gl_method One of 'gl' - regular GL alg., and 'quic' for 
             Quadratic Approximation for Sparse Inverse Covariance Estimation
             Hsieh, C. J., Sustik, M. A., Dhillon, I. S., & Ravikumar, P. (2014). 
             QUIC: quadratic approximation for sparse inverse covariance estimation. 
             The Journal of Machine Learning Research, 15(1), 2911-2947.
    cv : boolean. Whether to use CV during the model fitting process 
         (e.g for the selection of the regularization parameter for 
         the graphical lasso step)
    init_guess : Initial guess for parameters (sigma, rho)
    bounds : tuple of two 2-tupples. Bounds for parameters (sigma, rho).
    rhos : grid of values of rho for exhaustive search. Only used if 
           self.exhaustive_search is True.  
    sigmas : grid of values of sigma for exhaustive search. Only used if 
           self.exhaustive_search is True.
    exhaustive_search : whether to do exhaustive search for minimazing with 
                        respect to (sigma, rho)
    assume_centered : Boolean. If True, data are not centered before computation of
                      empirical covariance. If False, data are centered before computation.
                      (see sklearn.covariance.empirical_covariance)
    glasso_max_iter : maximal number of iterations for each iteration over
                      the Graphical Lasso alg.
    n_lams : see `alphas` in sklearn.covariance.GraphLassoCV or `lams`
             in inverse_covariance.QuicGraphLassoCV
    n_refinements : The number of times the grid is refined.
    n_folds : Number of CV folds.
    verbose : Boolean. Whether to print progress and messages
    """
    def __init__(self, max_iter = 40, cv = True, lam = 1e-1, tol_glasso = 1e-4,
                 tol = 1e-5, init_B = 'zero', gl_method = 'quic', 
                 init_guess = [1, .5], bounds = None, rhos = None, sigmas = None,
                 exhaustive_search = False, assume_centered = False, glasso_max_iter = int(1e3),
                 n_lams = 10, n_refinements = 5, n_folds = 3, verbose = False):
        
        self.max_iter = max_iter
        self.cv = cv
        self.lam = lam
        self.tol_glasso = tol_glasso
        self.tol = tol
        self.init_B = init_B
        if gl_method not in ['quic', 'gl']:
            raise ValueError("gl_method must be one of ['quic', 'gl'].")
        self.gl_method = gl_method
        self.init_guess = init_guess
        self.bounds = ((0 + 1e-10, 5), (.0+1e-10, 1-1e-10)) if bounds is None else bounds
        self.rhos = np.linspace(.001, .999, 20) if rhos is None else rhos
        self.sigmas = np.linspace(.01, 2, 20) if sigmas is None else sigmas
        self.exhaustive_search = exhaustive_search
        self.assume_centered = assume_centered
        self.glasso_max_iter = glasso_max_iter
        self.n_lams = n_lams
        self.n_refinements = n_refinements
        self.n_folds = n_folds
        self.verbose = verbose
        
        # set during fitting process
        self.p = None
        self.q = None
        self.n = None
        self.X = None
        self.Y = None
        self.X_trans = None
        self.Y_trans = None
        self.y_trans = None
        self.L = None
        self.S = None
        self.U = None
        self.B_hat = None
        self.sigma, self.rho = None, None
        self.Omega, self.Sigma = None, None
    
    def data_transform(self):
        """
        Transform the data into the required form (using eigendecomposition):
        
        X_trans = L.T * X
        Y_trans = L.T * Y * U
        """
        self.L, self.S = MrRCE.get_LS(self.X)
        self.U = MrRCE.get_U(self.q)
        # set transformed
        self.X_trans = self.L.T * self.X
        self.Y_trans = self.L.T * self.Y * self.U
    
    @staticmethod
    def get_LS(X):
        """
        Compute L and S s.t X*Xt = L*S*L^T with diag. S and orthogonal L
        
        Parameters
        ----------
        X: X matrix of shape (n, p)
        """
        p = X.shape[1]
        X = np.matrix(X)
        XXt = X * X.T
        L, s, L_t = LA.svd(XXt)
        s[p:] = 0
        return np.matrix(L), np.matrix(np.diag(s))

    @staticmethod
    def get_C(q, rho = .5):
        """
        Generate equicorrelation matrix
        
        Parameters
        ----------
        q: Rows and columns dimension
        rho: correlation parameter in [0,1)
        """
        assert((rho >= 0) & (rho < 1))
        return np.matrix(np.eye(q) + rho * (np.ones((q, q)) - np.eye(q)))
    
    @staticmethod
    def get_U(q):
        """
        Generate orthogonal matrix U s.t C=UDU^T and D is diag.
        
        Parameters
        ----------
        q : dimension
        """
        C = MrRCE.get_C(q)
        U, D, V = LA.svd(C)
        return np.matrix(U)
    
    @staticmethod
    def get_D(rho, q):
        """
        Generate the matrix D
        
        Parameters
        ----------
        rho: correlation parameter
        q: dimension of D
        """
        d = [(q - 1) * rho + 1] + [(1 - rho) for i in range(q - 1)] # eigenvalues
        return np.matrix(np.diag(d))
        
    def fit(self, X, Y, cv = None, init_B = None, max_iter = None, 
            glasso_max_iter = None, init_guess = None, assume_centered = None):
        """
        Assuming data is already transformed and of the required form
        
        Parametes
        ---------
        X : matrix of predictors
        Y : response matrix
        """
        assert(X.shape[0] == Y.shape[0])
        self.n, self.q = Y.shape
        self.p = X.shape[1]
        self.X = X
        self.Y = Y
        # transform the data
        self.data_transform()
        # set initial B value
        if init_B is not None:
            self.init_B = init_B
        if self.init_B == 'zero':
            B_star_hat = np.zeros((self.p, self.q))
        elif self.init_B == 'ols':
            B_star_hat = LinearRegression(fit_intercept = False).fit(self.X, self.Y).coef_.T
        else:
            raise ValueError("\ninit_B should be one of 'zero', 'ols'.")
        # replace self.attr if attr is given
        if cv is not None:
            self.cv = cv
        if glasso_max_iter is not None:
            self.glasso_max_iter = glasso_max_iter
        if init_guess is not None:
            self.init_guess = init_guess
        if assume_centered is not None:
            self.assume_centered = assume_centered
        if max_iter is not None:
            self.max_iter = max_iter
            
        # vec(Y)
        self.y_trans = np.reshape(self.Y_trans.T, (self.n * self.q, 1))
        
        iter_num = 1
        Omega_old = np.matrix(np.eye(self.q))
        Sigma_old = np.matrix(np.eye(self.q))
        glasso_gap = self.tol_glasso * 2
        gap = self.tol * 2
        init_guess = self.init_guess
        while (iter_num <= self.max_iter) and ((glasso_gap > self.tol_glasso) or (gap > self.tol)):
            if self.verbose:
                print("\rIter number {iter_num} ".format(iter_num = iter_num), end = "")
            # steps
            ## 1
            Sigma, Omega = self.step_1(B_star = B_star_hat, init_guess = Sigma_old, 
                                       Omega0 = Omega_old, Sigma0 = Sigma_old)
            ## 2
            if self.exhaustive_search:
                sigma, rho = self.find_minima(self.Neg_Log_Likelihood, 
                                              rhos = self.rhos, 
                                              sigmas = self.sigmas,
                                              # for NLL
                                              Sigma = Sigma)
                
            else:
                sigma, rho = self.step_2(init_guess, Sigma)
            # step 3 - BLUP
            B_star_hat = self.step_3(sigma, rho, Sigma)
            
            # compute gaps
            glasso_gap = LA.norm(Omega - Omega_old, ord = 'fro')
            gap = LA.norm(np.array([sigma, rho]) - np.array(init_guess))
            
            # update
            Omega_old = Omega
            Sigma_old = Sigma
            init_guess = sigma, rho
            iter_num += 1
            
        if iter_num >= self.max_iter and ((glasso_gap > self.tol_glasso) or (gap > self.tol)):
            if self.verbose:
                print("\nFailed to converge after {} iterations: \nGlasso gap: {:.6f} \n(sigma, rho) gap: {:.6f}".\
                      format(self.max_iter, glasso_gap, gap))
        # B est. - inverse transform
        self.B_hat = B_star_hat * self.U.T
        self.Sigma = self.U * Sigma * self.U.T
        self.Omega = self.U * Omega * self.U.T
        # parameters
        self.sigma, self.rho = sigma, rho
        self.iters = iter_num - 1
    
    def step_1(self, B_star, init_guess = None, 
               Omega0 = None, Sigma0 = None):
        """
        Given B_star, we estimate Omega, the graphical lasso solution for the precision matrix
        """
        S_G = self.Y_trans - self.X_trans * B_star
        emp_cov = empirical_covariance(S_G, assume_centered = self.assume_centered)
        if init_guess is not None:
            init_guess = np.array(init_guess)
        # CV
        if self.cv:
            if self.gl_method == 'quic':
                # GL with QUIC
                qgl = QuicGraphLassoCV(lams = self.n_lams, 
                                       max_iter = self.glasso_max_iter, 
                                       Theta0 = Omega0, Sigma0 = Sigma0,
                                       cv = self.n_folds)
                qgl.fit(S_G)
                Sigma, Omega = qgl.covariance_, qgl.precision_
            else:
                # GL no QUIC
                try:
                    gl = GraphLassoCV(alphas=self.n_lams, 
                                      assume_centered=self.assume_centered, 
                                      max_iter = self.glasso_max_iter, 
                                      n_refinements = self.n_refinements,
                                      cv = self.n_folds)
                    gl.fit(S_G)
                    Sigma, Omega = gl.covariance_, gl.precision_
                except:
                    if self.verbose:
                        print("\nUsing LARS solver...")
                    gl = GraphLassoCV(alphas=self.n_lams, 
                                      assume_centered=self.assume_centered, 
                                      max_iter = self.glasso_max_iter, 
                                      n_refinements = self.n_refinements,
                                      cv = self.n_folds,
                                      mode = 'lars')
                    gl.fit(S_G)
                    Sigma, Omega = gl.covariance_, gl.precision_
        else:
            if self.gl_method == 'quic':
                # QUIC
                qgl = QuicGraphLasso(lam = self.lam, 
                                     max_iter = self.glasso_max_iter, 
                                     Theta0 = Omega0, Sigma0 = Sigma0)
                qgl.fit(S_G)
                Sigma, Omega = qgl.covariance_, qgl.precision_
            else:
                # No QUIC
                try:
                    Sigma, Omega = graph_lasso(emp_cov, alpha = self.lam, 
                                               cov_init = init_guess, 
                                               max_iter = self.glasso_max_iter)
                except:
                    if self.verbose:
                        print("\nUsing LARS solver...")
                    # We prefer LARS for very sparse underlying graphs, where p > n.
                    Sigma, Omega = graph_lasso(emp_cov, alpha = self.lam, 
                                               cov_init = init_guess, 
                                               max_iter = self.glasso_max_iter, 
                                               mode = 'lars')
        return Sigma, Omega
    
    def step_2(self, initial_guess, Sigma, maxiter = 100):
        """
        Given Sigma, we estimate sigma & rho
        """
        res = minimize(self.Neg_Log_Likelihood,
                       initial_guess, 
                       args = (Sigma), 
                       bounds = self.bounds,
                       options = {'maxiter':maxiter}
                      )
        return res.x
    
    def step_3(self, sigma, rho, Sigma):
        """
        BLUP for B^* = BU
        
        This is step 3 of the alg.
        """
        C = MrRCE.get_C(self.q)
        Gamma = np.matrix(np.kron(sigma**2 * C, np.eye(self.p)))
        Z = np.matrix(np.kron(self.U.T, self.X_trans))
        R = np.matrix(np.kron(Sigma, np.eye(self.n)))
        V = Z * Gamma * Z.T + R
        V_sp = csr_matrix(V)
        Vi = self.sparse_inverse(V_sp)
        # Gamma * Z^T * V^-1 * y
        beta = Gamma * Z.T * Vi * self.y_trans
        # unvec
        B = np.matrix(np.reshape(beta.T, (self.q, self.p))).T
        return B * self.U
    
    def Neg_Log_Likelihood(self, params, Sigma):
        """
        Evaluate the NLL function of y = vec(Y_tilde)
        """
        sigma, rho = params
        D = MrRCE.get_D(rho, self.q)
        V = np.kron(((sigma ** 2) * D), self.S) + np.kron(Sigma, np.eye(self.n))
        V_sp = csr_matrix(V)
        Vi = self.sparse_inverse(V_sp)
        mu_y = np.zeros_like(self.y_trans) # zero mean  
        return np.float(self.fast_logdet(V) + np.square(mahalanobis(self.y_trans, mu_y, Vi)))
    
    @staticmethod
    def find_minima(func, 
                    rhos, 
                    sigmas, 
                    **kwargs):
        """
        Find the minima of a function with two parameters, rho and sigma,
        using grid search
        
        Parameters
        ----------
        func: Objective function
        rhos: grid of values for rho
        sigmas: grid of values for sigma
        """
        zs = [func((sigma, rho), **kwargs) for 
              sigma, rho in product(sigmas, rhos)]
        argmin = list(product(sigmas, rhos))[np.argmin(zs)]
        return argmin
    
    def predict(self, X):
        """
        Predict using MrRCE
        
        Parameters
        ----------
        X: matrix of covariates
        """
        if self.B_hat is None:
            raise ValueError("You must fit the model first.")
        return np.matrix(X) * np.matrix(self.B_hat)
    
    @staticmethod
    def fast_logdet(A):
        """
        Compute log(det(A)) for A symmetric

        Equivalent to : np.log(nl.det(A)) but more robust.
        It returns -Inf if det(A) is non positive or is not defined.
        """
        sign, ld = np.linalg.slogdet(A)
        if not sign > 0:
            return -np.inf
        return ld
    
    def sparse_inverse(self, V):
        """
        Inverse of the sparse covariance matrix
        """
        I = np.eye(self.n * self.q)
        return np.matrix(spsolve(V, I))