# Author: Aviv Navon
# Email: avivnav@gmail.com
#
# Code for running the simulations presented in the paper 
# "Capturing Between-Tasks Covariance and Similarities Using Multivariate Linear Mixed Models"
from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pandas as pd
import numpy as np
from simulation_utils import generate_data, me
from mrrce import MrRCE
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskLassoCV
import matplotlib.pyplot as plt

# try to create folders for results and plots
try:
    os.mkdir("results")
    os.mkdir("plots")
except:
    pass

possible_configurations = ['ar_sparse', 'ar_dense', 'equi', 'fgn', 'identity']
default_N = 200
# catch configurations and number of replications
try:
    conf = sys.argv[1]
except:
    raise ValueError("Please pass configurations file name.")
try:
    N = sys.argv[2]
    N = int(N)
except:
    print("Setting the number of replications to the default {}".format(default_N))
    N = default_N

assert(conf in possible_configurations), "Configurations name must be one of {}.".format(possible_configurations)

# settings for the different simulations
rhos = np.linspace(0, .8, 5)     # grid of values for rho (correlation coefficient)
# parameters dict for the different simulations
params_ar_dense = dict(
n = 50,                          # num obs
p = 20,                          # num predictors
q = 5,                           # num tasks
sigma = 1,                       # coeff variance
corr_x = .7,                     # grid of values for rho (correlation coefficient)
sigma_err = 1,                   # correlation coefficient for predictors
err_corr = .75,                  # correlation coefficient for errors
g_sparse_level = .0,             # group sparsity level
sparse_level = .0,               # sparsity level
err_cov_type = 'ar'              # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

params_ar_sparse = dict(
n = 50,                          # num obs
p = 20,                          # num predictors
q = 5,                           # num tasks
sigma = 1,                       # coeff variance
corr_x = .7,                     # grid of values for rho (correlation coefficient)
sigma_err = 1,                   # correlation coefficient for predictors
err_corr = .75,                  # correlation coefficient for errors
g_sparse_level = .1,             # group sparsity level
sparse_level = .1,               # sparsity level
err_cov_type = 'ar'              # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

params_fgn = dict(
n = 50,                          # num obs
p = 20,                          # num predictors
q = 5,                           # num tasks
sigma = 1,                       # coeff variance
corr_x = .7,                     # grid of values for rho (correlation coefficient)
sigma_err = 1,                   # correlation coefficient for predictors
g_sparse_level = .0,             # group sparsity level
sparse_level = .0,               # sparsity level
err_cov_type = 'fgn'             # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

params_equi = dict(
n = 50,                          # num obs
p = 20,                          # num predictors
q = 5,                           # num tasks
sigma = 1,                       # coeff variance
corr_x = .7,                     # grid of values for rho (correlation coefficient)
sigma_err = 1,                   # correlation coefficient for predictors
err_corr = .9,                   # correlation coefficient for errors
g_sparse_level = .1,             # group sparsity level
sparse_level = .1,               # sparsity level
err_cov_type = 'equi'            # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

params_idntity = dict(
n = 50,                          # num obs
p = 20,                          # num predictors
q = 5,                           # num tasks
sigma = .75,                       # coeff variance
corr_x = .7,                     # grid of values for rho (correlation coefficient)
sigma_err = 1,                   # correlation coefficient for predictors
g_sparse_level = .0,             # group sparsity level
sparse_level = .2,               # sparsity level
err_cov_type = 'identity'        # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

settings_dict = {'ar_sparse':params_ar_sparse, 'ar_dense':params_ar_dense,
                 'equi':params_equi, 'fgn':params_fgn, 'identity':params_idntity}

simulation_params = settings_dict[conf]

print("Running simulation name '{conf}' with {N} replications.".format(conf=conf, N=N))
# running simulation
np.random.seed(1) # for reproducibility
results = []
for rep in range(1, N + 1):
    for rho in rhos:
        print('\rRound {rep}, rho = {rho:.1f}'.format(rep=rep, rho=rho), end = '')
        X, Y, B, Sigma, Sigma_X = generate_data(rho = rho, **simulation_params)
        mrrce = MrRCE()
        mrrce.fit(X, Y)
        # OLS
        lm = LinearRegression(fit_intercept=False).fit(X, Y)
        B_ols = np.matrix(lm.coef_.T)
        # Ridge
        ridge = RidgeCV(fit_intercept=False).fit(X, Y)
        B_ridge = np.matrix(ridge.coef_.T)
        # Group Lasso
        gl = MultiTaskLassoCV(fit_intercept=False).fit(X, Y)
        B_gl = np.matrix(gl.coef_.T)
        # Results
        results.append({'rho':rho, 
                        'MrRCE':me(B, mrrce.B_hat, Sigma_X),
                        'OLS':me(B, B_ols, Sigma_X),
                        'Ridge':me(B, B_ridge, Sigma_X),
                        'GroupLasso':me(B, B_gl, Sigma_X)})

# create a data frame with the data
results_df = pd.DataFrame(results)
try:
    path = 'results' 
    results_df.to_csv(os.path.join(path, "simulation_results_{}.csv".format(conf)), index=False)
except:
    results_df.to_csv("simulation_results_{}.csv".format(conf), index=False)
# for plotting
to_plot = results_df.groupby('rho', as_index = False).mean().melt(id_vars = 'rho', 
                                                                  var_name = 'method',
                                                                  value_name = 'ME')
# plot
fig, ax = plt.subplots(figsize = (15, 5))
for m in to_plot.method.unique():
    curr = to_plot.loc[to_plot.method == m].copy()
    plt.plot(curr.rho, curr.ME, 'o--')
    
ax.legend(to_plot.method.unique(), loc='upper center', fancybox=True, 
          shadow=True, ncol=4, fontsize='x-large', bbox_to_anchor=(0.5, 1.15))
ax.set_xlabel('rho', fontsize='x-large')
ax.set_ylabel('Model Error', fontsize='x-large')
try:
    path = 'plots'
    plt.savefig(os.path.join(path, 'simulation_plot_{}.png'.format(conf)))
except:
    plt.savefig('simulation_plot_{}.png'.format(conf))
print("\nDone.")
