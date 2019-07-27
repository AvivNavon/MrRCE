import logging

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from sklearn.covariance import (GraphicalLasso, GraphicalLassoCV,
                                empirical_covariance, graphical_lasso)
from sklearn.linear_model import LinearRegression

from .data import Data

logging.basicConfig(level=logging.INFO)


class MrRCE:

    def __init__(
            self,
            max_iter=50,
            glasso_cv=True,
            lam=1.5e-1,
            tol_glasso=1e-2,
            tol=1e-3,
            init_coef_matrix='ols',
            rho_init_guess=.5,
            sigma_init_guess=1,
            bounds=None,
            assume_centered=False,
            glasso_max_iter=100,
            n_lams=20,
            n_refinements=4,
            n_folds=3,
            glasso_n_jobs=None,
            verbose=False
    ):
        assert init_coef_matrix in ('zeros', 'ols'), "init_z_matrix must be in ('zeros', 'ols')"
        assert 0 < rho_init_guess < 1
        assert sigma_init_guess > 0

        self.max_iter = max_iter
        self.glasso_cv = glasso_cv
        self.lam = lam
        self.tol_glasso = tol_glasso
        self.tol = tol

        self.init_coef_matrix = init_coef_matrix
        self.sigma_init_guess = sigma_init_guess
        self.rho_init_guess = rho_init_guess
        self.bounds = ((1e-3, 5.), (0., 1 - 1e-3)) if bounds is None else bounds
        self.assume_centered = assume_centered
        self.glasso_max_iter = glasso_max_iter
        self.n_lams = n_lams
        self.n_refinements = n_refinements
        self.n_folds = n_folds
        self.glasso_n_jobs = glasso_n_jobs
        self.verbose = verbose

        # during fit
        self.data = None
        self.Gamma = None
        self.rho = None
        self.sigma = None
        self.Omega = None
        self.Sigma = None
        self.convergence_path = None

    @staticmethod
    def fast_logdet(matrix):
        """
        Compute log(det(matrix)) for symmetric matrix.
        Equivalent to : np.log(nl.det(A)) but more robust. It returns -Inf if det(A) is non positive or is not defined.

        :param matrix:
        :return:
        """
        sign, ld = np.linalg.slogdet(matrix)
        if not sign > 0:
            return -np.inf
        return ld

    @staticmethod
    def sparse_inverse(matrix):
        """

        :param matrix: Inverse of the sparse covariance matrix
        :return:
        """
        sparse_matrix = csr_matrix(matrix)
        identity = np.eye(sparse_matrix.shape[0])
        return np.matrix(spsolve(sparse_matrix, identity))

    def neg_log_conditional_likelihood(self, Omega, Gamma):
        """
        Negative log-likelihood of Y|Gamma ~ MVN(Z*Gamma, I, Sigma)

        :param Omega:
        :param Gamma:
        :return:
        """
        z_gamma = np.matmul(self.data.z_matrix_transform, Gamma)
        return (1/self.data.n) * np.trace(
            np.linalg.multi_dot(
                [Omega, (self.data.y_matrix_transform-z_gamma).transpose(), self.data.y_matrix_transform-z_gamma]
            )
        ) - self.fast_logdet(Omega)  # TODO: go over this again

    @staticmethod
    def penalty(Omega):
        """
        l_1 penalty over the off-diagonal elements of a matrix

        :param Omega:
        :return:
        """
        abs_omega = np.abs(Omega)
        return np.sum(abs_omega) - np.trace(abs_omega)

    def neg_log_likelihood_gamma(self, params, Gamma):
        """
        Negative log-likelihood of Gamma ~ MVN(0, I, sigma^2 * D_rho)

        :param params: (sigma, rho)
        :param Gamma:
        :return:
        """
        sigma, rho = params
        d_matrix = self.data.get_d_matrix(rho)
        Lambda = np.power(sigma, -2) * self.sparse_inverse(d_matrix)  # TODO: we can find inverse by 1 / diag...
        trace = (1 / self.data.p) * np.trace(
            np.linalg.multi_dot([Lambda, Gamma.transpose(), Gamma])
        )
        logdet = self.fast_logdet(Lambda)   # TODO: again, we know the exact form of this since Lambda is diag...
        return trace - logdet

    def neg_log_penalized_likelihood(self, sigma, rho, Omega, Gamma):
        """
        Negative log-likelihood for the complete data (Y, Gamma)

        :param sigma:
        :param rho:
        :param Omega:
        :param Gamma:
        :return:
        """
        return (
                self.neg_log_conditional_likelihood(Omega, Gamma) +
                self.penalty(Omega) +
                self.neg_log_likelihood_gamma((sigma, rho), Gamma)
        )

    def _set_init_gamma(self):
        # TODO: change to ridge solution so it will always be well-define
        # note that the sklearn implementation for LinearRegression uses Ridge for cases where n < p
        if self.init_coef_matrix == 'zeros':
            return np.zeros((self.data.p, self.data.q))

        return LinearRegression(fit_intercept=False).fit(
            self.data.z_matrix_transform,
            self.data.y_matrix_transform
        ).coef_.transpose()

    def _check_stop_criteria(self, iter_number, sigma_rho_gap, glasso_gap):
        if iter_number == 0:
            return True
        else:
            return (iter_number < self.max_iter) and (sigma_rho_gap > self.tol or glasso_gap > self.tol_glasso)

    def _fit(self, Gamma, sigma_init_guess, rho_init_guess):
        # TODO: at the moment we do not use Sigma from previous iteration as hot start
        Sigma, Omega = self.graphical_lasso(Gamma)
        sigma, rho = self.estimate_sigma_rho(initial_guess=(sigma_init_guess, rho_init_guess), Gamma=Gamma)
        Gamma = self.predict_blup(sigma, rho, Sigma)
        return sigma, rho, Sigma, Omega, Gamma

    def fit(self, z_matrix, y_matrix):
        self.data = Data(y_matrix, z_matrix)
        # initial guesses
        Gamma = self._set_init_gamma()
        sigma = self.sigma_init_guess
        rho = self.rho_init_guess
        sigma_rho_gap = self.tol * 2
        glasso_gap = self.tol_glasso * 2  # TODO: use a single tol?
        Omega = np.matrix(np.eye(self.data.q))
        Sigma = np.matrix(np.eye(self.data.q))
        # save convergence path
        self.convergence_path = []

        iter_number = 0
        while self._check_stop_criteria(iter_number, sigma_rho_gap, glasso_gap):
            Omega_old, sigma_old, rho_old = Omega, sigma, rho
            sigma, rho, Sigma, Omega, Gamma = self._fit(Gamma, sigma_old, rho_old)
            glasso_gap = np.linalg.norm(Omega - Omega_old, ord='fro')
            sigma_rho_gap = np.linalg.norm(np.array([sigma, rho]) - np.array([sigma_old, rho_old]))

            iter_number += 1
            nll = self.neg_log_penalized_likelihood(sigma, rho, Omega=Omega, Gamma=Gamma)
            self.convergence_path.append(nll)
            if self.verbose:
                logging.info(f"iter {iter_number}, loss {nll:.6f}")

        # update estimations
        self.Gamma = Gamma * self.data.u_matrix.transpose()
        self.rho = rho
        self.sigma = sigma
        self.Sigma = self.data.u_matrix * Sigma * self.data.u_matrix.transpose()
        self.Omega = self.data.u_matrix * Omega * self.data.u_matrix.transpose()

        if (iter_number >= self.max_iter) and (sigma_rho_gap > self.tol or glasso_gap > self.tol_glasso):
            if self.verbose:
                logging.info(f"Failed to converge after {iter_number} iterations: "
                             f"Glasso gap: {glasso_gap:.6f}, (sigma, rho) gap: {sigma_rho_gap:.6f}")

    def graphical_lasso(self, Gamma, Sigma_init=None):
        """
        Given Gamma, we estimate Omega, the graphical lasso solution for the precision matrix

        :param Gamma:
        :param Sigma_init:
        :return:
        """
        y_matrix_minus_mean = self.data.y_matrix_transform - np.matmul(self.data.z_matrix_transform, Gamma)
        emp_cov = empirical_covariance(y_matrix_minus_mean, assume_centered=self.assume_centered)
        if Sigma_init is not None:
            Sigma_init = np.matrix(Sigma_init)

        mode = 'cd'
        if self.data.n < self.data.p:
            # We preffer the LARS solver for very sparse underlying graphs
            # TODO: move this so we want need to check this every iteration
            mode = 'lars'

        if self.glasso_cv:
            gl = GraphicalLassoCV(
                alphas=self.n_lams,
                assume_centered=self.assume_centered,
                max_iter=self.glasso_max_iter,
                n_refinements=self.n_refinements,
                cv=self.n_folds,
                mode=mode,
                n_jobs=self.glasso_n_jobs
            )
            gl.fit(y_matrix_minus_mean)
            Sigma, Omega = gl.covariance_, gl.precision_

        elif Sigma_init is None:
            gl = GraphicalLasso(
                alpha=self.lam,
                assume_centered=self.assume_centered,
                max_iter=self.glasso_max_iter,
                mode=mode
            )
            gl.fit(y_matrix_minus_mean)
            Sigma, Omega = gl.covariance_, gl.precision_

        else:
            Sigma_init = np.matrix(Sigma_init)

            Sigma, Omega = graphical_lasso(
                emp_cov,
                alpha=self.lam,
                cov_init=Sigma_init,
                max_iter=self.glasso_max_iter,
                mode=mode
            )

        return Sigma, Omega

    def estimate_sigma_rho(self, initial_guess, Gamma, maxiter=100):
        result = minimize(
            self.neg_log_likelihood_gamma,
            x0=initial_guess,
            args=(Gamma,),
            bounds=self.bounds,
            options={'maxiter': maxiter}
        )
        return result.x

    def predict_blup(self, sigma, rho, Sigma):
        """
        BLUP for Gamma

        """
        d_matrix = self.data.get_d_matrix(rho)
        l_matrix = np.matrix(np.kron(sigma ** 2 * d_matrix, np.eye(self.data.p)))
        z_tilde_matrix = np.matrix(np.kron(np.eye(self.data.q), self.data.z_matrix_transform))
        r_matrix = np.matrix(np.kron(Sigma, np.eye(self.data.n)))
        lambda_matrix = np.linalg.multi_dot([z_tilde_matrix, l_matrix, z_tilde_matrix.transpose()]) + r_matrix

        inverse_lambda_matrix = self.sparse_inverse(lambda_matrix)
        gamma_vector = np.linalg.multi_dot(
            [
                l_matrix.transpose(),
                z_tilde_matrix.transpose(),
                inverse_lambda_matrix,
                self.data.y_vector_transform
            ]
        )
        # unvec gamma
        Gamma = np.matrix(np.reshape(gamma_vector.transpose(), (self.data.q, self.data.p))).transpose()
        return Gamma

    def predict(self, matrix):
        """
        Predict using MrRCE

        :param matrix: observations
        :return:
        """
        if self.Gamma is None:
            raise ValueError("You must fit the model first.")
        return np.matmul(np.matrix(matrix), np.matrix(self.Gamma))
