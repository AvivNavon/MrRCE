import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Sequence

import numpy as np
from pandas.core.frame import DataFrame
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from sklearn.covariance import graphical_lasso
from sklearn.covariance.empirical_covariance_ import (empirical_covariance,
                                                      log_likelihood)
from sklearn.covariance.graph_lasso_ import alpha_max
from sklearn.model_selection import KFold

from .data import Data

logging.basicConfig(level=logging.INFO)


class MrRCE:

    def __init__(
        self,
        max_iter=50,
        glasso_max_iter=100,
        alpha=1.5e-3,
        tol_glasso=1e-2,
        tol=1e-3,
        rho_init_guess=.5,
        sigma_init_guess=1,
        bounds=None,
        alphas=10,
        n_folds=3,
        use_cv=True,
        verbose=False
    ):
        assert 0 < rho_init_guess < 1
        assert sigma_init_guess > 0

        self.max_iter = max_iter
        self.alpha = alpha
        self.tol_glasso = tol_glasso
        self.tol = tol

        self.sigma_init_guess = sigma_init_guess
        self.rho_init_guess = rho_init_guess
        self.bounds = ((1e-3, 10.), (0., .9995)) if bounds is None else bounds
        self.glasso_max_iter = glasso_max_iter
        self.alphas = alphas
        if isinstance(alphas, Sequence):
            self.n_alphas = len(alphas)
        else:
            self.n_alphas = alphas

        self.n_folds = n_folds
        self.use_cv = use_cv
        self.verbose = verbose

        # during fit
        self.data = None
        self.Gamma = None
        self.rho = None
        self.sigma = None
        self.Omega = None
        self.Sigma = None
        self.convergence_path = None
        self.n_iters = None

    @staticmethod
    def fast_logdet(matrix):
        """
        Compute log(det(matrix)) for symmetric matrix. Equivalent to : np.log(nl.det(A)) but more robust.
        It returns -Inf if det(A) is non positive or is not defined.

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

    def neg_log_conditional_likelihood(self, Omega, residuals_transpose_residuals):
        """
        Negative log-likelihood of Y|Gamma ~ MVN(Z*Gamma, I, Sigma)

        :param Omega:
        :param residuals_transpose_residuals:
        :return:
        """
        return (1 / self.data.n) * np.trace(
            np.matmul(Omega, residuals_transpose_residuals)
        ) - self.fast_logdet(Omega)  # TODO: go over this again

    @staticmethod
    def penalty(Omega, alpha):
        """
        l_1 penalty over the off-diagonal elements of a matrix

        :param Omega:
        :return:
        """
        abs_omega = np.abs(Omega)
        return alpha * (np.sum(abs_omega) - np.trace(abs_omega))

    @staticmethod
    def _glasso_generalisation_likelihood(mle, precision_):
        return log_likelihood(mle, precision_)

    def neg_log_likelihood_gamma(self, params, Gamma_transpose_Gamma):
        """
        Negative log-likelihood of Gamma ~ MVN(0, I, sigma^2 * D_rho)

        :param params: (sigma, rho)
        :param Gamma_transpose_Gamma:
        :return:
        """
        sigma, rho = params
        d_matrix = self.data.get_d_matrix(rho)
        Lambda = np.power(sigma, -2) * self.sparse_inverse(d_matrix)  # TODO: we can find inverse by 1 / diag...

        trace = (1 / self.data.p) * np.trace(
            np.matmul(Lambda, Gamma_transpose_Gamma)
        )
        logdet = self.fast_logdet(Lambda)   # TODO: again, we know the exact form of this since Lambda is diag...
        return trace - logdet

    def neg_log_penalized_likelihood(
            self,
            sigma,
            rho,
            Omega,
            residuals_transpose_residuals,
            Gamma_transpose_Gamma,
            alpha
    ):
        """
        Negative log-likelihood for the complete data (Y, Gamma)

        :param sigma:
        :param rho:
        :param Omega:
        :param residuals_transpose_residuals:
        :param Gamma_transpose_Gamma:
        :param alpha:
        :return:
        """
        return (
            self.neg_log_conditional_likelihood(Omega, residuals_transpose_residuals=residuals_transpose_residuals) +
            self.penalty(Omega, alpha=alpha) +
            self.neg_log_likelihood_gamma((sigma, rho), Gamma_transpose_Gamma=Gamma_transpose_Gamma)
        )

    def _check_stop_criteria(self, iter_number, sigma_rho_gap, glasso_gap):
        if iter_number == 0:
            return True
        else:
            return (iter_number < self.max_iter) and (sigma_rho_gap > self.tol or glasso_gap > self.tol_glasso)

    def _fit(self, Sigma, rho, sigma, alpha=None):
        Gamma_transpose_Gamma = self._conditional_expectation_of_gamma_transpose_gamma(
            Sigma=Sigma,
            rho=rho,
            sigma=sigma
        )

        expected_value_resid_square = self._conditional_expectation_of_residuals_sqaure(
            Sigma=Sigma,
            sigma=sigma,
            rho=rho
        )
        # TODO: at the moment we do not use Sigma from previous iteration as hot start
        Sigma, Omega = self._graphical_lasso(expected_value_resid_square=expected_value_resid_square, alpha=alpha)
        sigma, rho = self.estimate_sigma_rho_exact(Gamma_transpose_Gamma=Gamma_transpose_Gamma)

        return sigma, rho, Sigma, Omega, expected_value_resid_square, Gamma_transpose_Gamma

    def _calc_glasso_score(self, z_matrix, y_matrix, Omega, sigma, rho, Sigma):
        Gamma = self.predict_blup(
            sigma,
            rho,
            Sigma,
            z_matrix=z_matrix,
            y_matrix=y_matrix
        )
        # TODO: we need the expectation here, this is an approximation
        mle = empirical_covariance(y_matrix - np.matmul(z_matrix, Gamma))
        return self._glasso_generalisation_likelihood(mle, Omega)

    def _kfold_multiprocessing(self, z_matrix, y_matrix, idxs):
        curr_z = z_matrix[idxs[0], :]
        curr_y = y_matrix[idxs[0], :]

        curr_z_test = z_matrix[idxs[1], :]
        curr_y_test = y_matrix[idxs[1], :]

        curr_scores_, alphas = self._calc_glasso_cv_scores(
            z_train_matrix=curr_z,
            y_train_matrix=curr_y,
            z_test_matrix=curr_z_test,
            y_test_matrix=curr_y_test,
        )
        return curr_scores_, alphas

    def _fit_cv(self, z_matrix, y_matrix):

        k_fold_splitter = KFold(n_splits=self.n_folds)

        scores_ = []
        for idxs in k_fold_splitter.split(z_matrix, y_matrix):
            curr_z = z_matrix[idxs[0], :]
            curr_y = y_matrix[idxs[0], :]

            curr_z_test = z_matrix[idxs[1], :]
            curr_y_test = y_matrix[idxs[1], :]

            curr_scores_, alphas = self._calc_glasso_cv_scores(
                    z_train_matrix=curr_z,
                    y_train_matrix=curr_y,
                    z_test_matrix=curr_z_test,
                    y_test_matrix=curr_y_test,
                )

            scores_.append(curr_scores_)

        best_alpha_idx = np.nan_to_num(
            np.nanmean(np.array(scores_), axis=0),
            nan=-np.inf,
        ).argmax()  # scores are log-likelihood, so we want max
        best_alpha = alphas[best_alpha_idx]

        # refit
        self._fit_wo_cv(z_matrix, y_matrix, alpha=best_alpha)

    def _set_alpha_range(self, emp_cov):
        if isinstance(self.alphas, Sequence):
            return self.alphas

        else:
            alpha_1 = alpha_max(emp_cov)
            alpha_0 = 1e-4 * alpha_1
            return np.logspace(
                np.log10(alpha_0),
                np.log10(alpha_1),
                self.n_alphas
            )[::-1]

    def _calc_glasso_cv_scores(
            self,
            z_train_matrix,
            y_train_matrix,
            z_test_matrix,
            y_test_matrix,
    ):
        if self.sigma is None or self.rho is None or self.Sigma is None:
            sigma, rho, Omega, Sigma, sigma_rho_gap, glasso_gap = self._fit_init(z_train_matrix, y_train_matrix)
        else:
            self.data = Data(y_train_matrix, z_train_matrix)
            sigma = self.sigma
            rho = self.rho
            Sigma = self.Sigma

        residuals_transpose_residuals = self._conditional_expectation_of_residuals_sqaure(
            Sigma=Sigma,
            sigma=sigma,
            rho=rho
        )

        alphas = self._set_alpha_range(residuals_transpose_residuals)

        process_count = min([cpu_count() - 1, len(alphas)])
        normalize_param = y_train_matrix.shape[0]

        partial_glasso = partial(
            self._calc_glasso_cv_score,
            z_test_matrix,
            y_test_matrix,
            sigma,
            rho,
            residuals_transpose_residuals,
            normalize_param
        )
        with Pool(process_count) as p:
            curr_scores_ = list(
                    p.imap(partial_glasso, alphas, chunksize=1)
            )

        return curr_scores_, alphas

    def _calc_glasso_cv_score(
            self,
            z_test_matrix,
            y_test_matrix,
            sigma,
            rho,
            residuals_transpose_residuals,
            normalize_param,
            alpha
    ):
        try:
            Sigma, Omega = self._graphical_lasso(
                residuals_transpose_residuals,
                alpha=alpha,
                normalize_param=normalize_param
            )

            this_score = self._calc_glasso_score(
                z_matrix=z_test_matrix,
                y_matrix=y_test_matrix,
                Omega=Omega,
                sigma=sigma,
                rho=rho,
                Sigma=Sigma
            )

        except FloatingPointError:
            this_score = np.nan  # -np.inf

        if not np.isfinite(this_score):
            this_score = np.nan  # -np.inf

        return this_score

    def _fit_init(self, z_matrix, y_matrix):
        self.data = Data(y_matrix, z_matrix)
        # initial guesses
        sigma = self.sigma_init_guess
        rho = self.rho_init_guess
        sigma_rho_gap = np.inf
        glasso_gap = np.inf  # TODO: use a single tol?
        Omega = np.matrix(np.eye(self.data.q))
        Sigma = np.matrix(np.eye(self.data.q))
        # save convergence path
        self.convergence_path = []
        return sigma, rho, Omega, Sigma, sigma_rho_gap, glasso_gap

    def _fit_wo_cv(self, z_matrix, y_matrix, alpha=None):
        sigma, rho, Omega, Sigma, sigma_rho_gap, glasso_gap = self._fit_init(z_matrix, y_matrix)

        if alpha is None:
            alpha = self.alpha

        iter_number = 0
        while self._check_stop_criteria(iter_number, sigma_rho_gap, glasso_gap):
            Omega_old, sigma_old, rho_old = Omega, sigma, rho
            sigma, rho, Sigma, Omega, residuals_transpose_residuals, Gamma_transpose_Gamma = self._fit(
                Sigma=Sigma,
                rho=rho,
                sigma=sigma,
                alpha=alpha
            )

            glasso_gap = np.linalg.norm(Omega - Omega_old, ord='fro')
            sigma_rho_gap = np.linalg.norm(np.array([sigma, rho]) - np.array([sigma_old, rho_old]))

            iter_number += 1

            nll = self.neg_log_penalized_likelihood(
                sigma,
                rho,
                Omega=Omega,
                residuals_transpose_residuals=residuals_transpose_residuals,
                Gamma_transpose_Gamma=Gamma_transpose_Gamma,
                alpha=alpha
            )
            self.convergence_path.append(nll)
            if self.verbose:
                logging.info(f"iter {iter_number}, loss {nll:.6f}")

        # update estimations
        Gamma = self.predict_blup(sigma=sigma, rho=rho, Sigma=Sigma)
        self.Gamma = Gamma * self.data.u_matrix.transpose()
        self.rho = rho
        self.sigma = sigma
        self.Sigma = self.data.u_matrix * Sigma * self.data.u_matrix.transpose()
        self.Omega = self.data.u_matrix * Omega * self.data.u_matrix.transpose()

        if (iter_number >= self.max_iter) and (sigma_rho_gap > self.tol or glasso_gap > self.tol_glasso):
            if self.verbose:
                logging.info(f"Failed to converge after {iter_number} iterations: "
                             f"Glasso gap: {glasso_gap:.6f}, (sigma, rho) gap: {sigma_rho_gap:.6f}")

        self.n_iters = iter_number

    def fit(self, z_matrix, y_matrix):

        if isinstance(z_matrix, DataFrame):
            z_matrix = z_matrix.values

        if isinstance(y_matrix, DataFrame):
            y_matrix = y_matrix.values

        if self.use_cv:
            self._fit_cv(z_matrix, y_matrix)

        else:  # no cv
            self._fit_wo_cv(z_matrix, y_matrix)

    def _graphical_lasso(self, expected_value_resid_square, alpha=None, normalize_param=None, Sigma_init=None):
        """
        Given Gamma, we estimate Omega, the graphical lasso solution for the precision matrix

        :param expected_value_resid_square:
        :param Sigma_init:
        :param normalize_param: number of rows in Y matrix. This is not self.data.n when we use CV.
        :return:
        """
        if not normalize_param:
            normalize_param = self.data.n

        if not alpha:
            alpha = self.alpha

        expected_value_resid_square *= (1 / normalize_param)
        expected_value_resid_square = np.array(expected_value_resid_square)

        # TODO: for now we do not do CV since the func that does this requres (Y-Z\Gamma),
        #  and not (Y-Z\Gamma)^T(Y-Z\Gamma). In theory, we can do SVD of the latter to return to (Y-Z\Gamma),
        #  but this is weird, because this is an expectation... If we think about it, in the CV func we will get
        #  (Y-Z\Gamma)^T(Y-Z\Gamma) again as the covariance, so it's probably OK.
        if Sigma_init is not None:
            Sigma_init = np.matrix(Sigma_init)

        mode = 'cd'
        if self.data.n < self.data.p:
            # We preffer the LARS solver for very sparse underlying graphs
            # TODO: move this so we want need to check this every iteration
            mode = 'lars'

        Sigma, Omega = graphical_lasso(
            expected_value_resid_square,
            alpha=alpha,
            cov_init=Sigma_init,
            max_iter=self.glasso_max_iter,
            mode=mode
        )

        return Sigma, Omega

    def estimate_sigma_rho_exact(self, Gamma_transpose_Gamma):
        Gamma_transpose_Gamma = np.array(Gamma_transpose_Gamma)
        gamma_transpose_gamma_diag = Gamma_transpose_Gamma.diagonal()
        m1 = self.data.p / gamma_transpose_gamma_diag[0]
        m2 = (self.data.p * (self.data.q - 1)) / np.sum(gamma_transpose_gamma_diag[1:])

        rho = (m2-m1) / (m2 + (self.data.q - 1) * m1)

        if rho > self.bounds[1][1]:
            rho = self.bounds[1][1]
        if rho < self.bounds[1][0]:
            rho = self.bounds[1][0]

        sigma = np.sqrt(np.power(m2, -1) / (1 - rho))

        return sigma, rho

    def estimate_sigma_rho(self, initial_guess, Gamma_transpose_Gamma, maxiter=100):
        result = minimize(
            self.neg_log_likelihood_gamma,
            x0=initial_guess,
            args=(Gamma_transpose_Gamma,),
            bounds=self.bounds,
            options={'maxiter': maxiter}
        )
        return result.x

    def predict_blup(self, sigma, rho, Sigma, z_matrix=None, y_matrix=None):
        """
        BLUP for Gamma

        """
        if z_matrix is None:
            z_matrix = self.data.z_matrix_transform

        if y_matrix is None:
            y_matrix = self.data.y_matrix_transform

        # set dims
        n = y_matrix.shape[0]

        # y vec
        shapes = y_matrix.shape
        y_vector = np.reshape(y_matrix.transpose(), (shapes[0] * shapes[1], 1))

        d_matrix = self.data.get_d_matrix(rho)
        l_matrix = np.matrix(np.kron(sigma ** 2 * d_matrix, np.eye(self.data.p)))
        z_tilde_matrix = np.matrix(np.kron(np.eye(self.data.q), z_matrix))
        r_matrix = np.matrix(np.kron(Sigma, np.eye(n)))

        lambda_matrix = np.linalg.multi_dot([z_tilde_matrix, l_matrix, z_tilde_matrix.transpose()]) + r_matrix

        inverse_lambda_matrix = self.sparse_inverse(lambda_matrix)
        gamma_vector = np.linalg.multi_dot(
            [
                l_matrix.transpose(),
                z_tilde_matrix.transpose(),
                inverse_lambda_matrix,
                y_vector
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

    def _calculate_joint_covariance_components(self, Sigma, rho, sigma, A):
        """
        Covariance matrix for (vec(A\Gamma), vec(Y))^T = [[Sigma_1_1, Sigma_1_2],[Sigma_2_1, Sigma_2_2]]

        :return:
        """
        inverse_delta = sigma ** 2 * self.data.get_d_matrix(rho)
        a_gamma_cov = np.matrix(
            np.kron(
                inverse_delta,
                np.matmul(A, A.transpose())
            )
        )
        y_cov = np.matrix(
            np.kron(Sigma, np.eye(self.data.n)) +
            np.kron(
                inverse_delta,
                np.matmul(self.data.z_matrix_transform, self.data.z_matrix_transform.transpose())
            )
        )
        gamma_y_cov = np.matrix(
            np.kron(
                inverse_delta,
                np.matmul(A, self.data.z_matrix_transform.transpose())
            )
        )
        return [[a_gamma_cov, gamma_y_cov], [gamma_y_cov.transpose(), y_cov]]

    def _calc_conditional_expectation_component(self, mu, cov, i, j, k):
        """
        E[\Gamma^TA^TA\Gamma]_i,j = \left[\mu_{\cdot i}^{\gamma\mid y}\right]^{T}\mu_{\cdot j}^{\gamma\mid y}+
        \sum_{k}\Sigma_{\left(i-1\right)p+k,\left(j-1\right)p+k}^{\gamma\mid y}

        :param mu: conditional mean
        :param cov: conditional covariance
        :param i: row index
        :param j: col index
        :param k: first dimension of A
        :return:
        """
        # TODO: verify that the indexing for mu and Sigma is correct !!!!
        mu_i = mu[i * k: (i+1) * k].reshape(-1)
        mu_j = mu[j * k: (j+1) * k].reshape(-1)

        exp_part = np.inner(mu_i, mu_j)

        cov_part = np.sum([cov[i * k + ell, j * k + ell] for ell in range(k)])

        return exp_part[0, 0] + cov_part

    def _conditional_expectation_of_gram_matrix(self, Sigma, rho, sigma, A):
        """
        calculating the expectation of \Gamma^TA^TA\Gamma conditioned on Y and the parameters \Theta

        :return:
        """
        cov_components = self._calculate_joint_covariance_components(Sigma, rho, sigma, A)
        # calculate cond. exp/cov
        partial_ = np.matmul(
            cov_components[0][1],
            self.sparse_inverse(cov_components[1][1])
        )
        # mu = \Sigma_{12}^{\gamma,y}\left[\Sigma_{22}^{\gamma,y}\right]^{-1}y
        mu_cond_dist = np.matmul(
            partial_,
            self.data.y_vector_transform
        )
        # cov = \Sigma_{11}^{\gamma,y}-
        # \Sigma_{12}^{\gamma,y}\left[\Sigma_{22}^{\gamma,y}\right]^{-1}\Sigma_{21}^{\gamma,y}
        cov_cond_dist = cov_components[0][0] - np.matmul(partial_, cov_components[1][0])

        expected_matrix = np.zeros((self.data.q, self.data.q))

        k = A.shape[0]

        for i in range(self.data.q):
            for j in range(i+1):
                exp_val = self._calc_conditional_expectation_component(mu_cond_dist, cov_cond_dist, i, j, k=k)
                expected_matrix[i, j] = expected_matrix[j, i] = exp_val

        return np.matrix(expected_matrix)

    def _conditional_expectation_of_residuals_sqaure(self, Sigma, rho, sigma):
        """
        expected value of (Y-Z\Gamma)^T(Y-Z\Gamma)

        We have (Y-Z\Gamma)^T(Y-Z\Gamma)=Y^TY-Y^TZ\Gamma-\Gamma^TZ^TY+\Gamma^TZ^TZ\Gamma

        :return:
        """
        expected_gamma = self.predict_blup(sigma=sigma, rho=rho, Sigma=Sigma)
        y_transpose_z_gamma = np.linalg.multi_dot([
            self.data.y_matrix_transform.transpose(),
            self.data.z_matrix_transform,
            expected_gamma
        ])

        res = (
                np.matmul(self.data.y_matrix_transform.transpose(), self.data.y_matrix_transform) -
                y_transpose_z_gamma -
                y_transpose_z_gamma.transpose() +
                self._conditional_expectation_of_gram_matrix(Sigma, rho, sigma, self.data.z_matrix_transform)
        )

        return res

    def _conditional_expectation_of_gamma_transpose_gamma(self, Sigma, rho, sigma):
        return self._conditional_expectation_of_gram_matrix(Sigma, rho, sigma, np.eye(self.data.p))
