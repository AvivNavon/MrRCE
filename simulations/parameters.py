import numpy as np

"""
settings for the different simulations
"""

RHOS = np.linspace(0, .8, 5)      # grid of values for rho (correlation coefficient)

params_ar_dense = dict(
    n=50,                          # num obs
    p=20,                          # num predictors
    q=5,                           # num tasks
    sigma=1,                       # coeff variance
    corr_x=.7,                     # grid of values for rho (correlation coefficient)
    sigma_err=1,                   # correlation coefficient for predictors
    err_corr=.75,                  # correlation coefficient for errors
    g_sparse_level=.0,             # group sparsity level
    sparse_level=.0,               # sparsity level
    err_cov_type='ar'              # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

params_ar_sparse = dict(
    n=50,                          # num obs
    p=20,                          # num predictors
    q=5,                           # num tasks
    sigma=1,                       # coeff variance
    corr_x=.7,                     # grid of values for rho (correlation coefficient)
    sigma_err=1,                   # correlation coefficient for predictors
    err_corr=.75,                  # correlation coefficient for errors
    g_sparse_level=.1,             # group sparsity level
    sparse_level=.1,               # sparsity level
    err_cov_type='ar'              # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

params_fgn = dict(
    n=50,                          # num obs
    p=20,                          # num predictors
    q=5,                           # num tasks
    sigma=1,                       # coeff variance
    corr_x=.7,                     # grid of values for rho (correlation coefficient)
    sigma_err=1,                   # correlation coefficient for predictors
    g_sparse_level=.0,             # group sparsity level
    sparse_level=.0,               # sparsity level
    err_cov_type='fgn'             # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

params_equi = dict(
    n=50,                          # num obs
    p=20,                          # num predictors
    q=5,                           # num tasks
    sigma=1,                       # coeff variance
    corr_x=.7,                     # grid of values for rho (correlation coefficient)
    sigma_err=1,                   # correlation coefficient for predictors
    err_corr=.9,                   # correlation coefficient for errors
    g_sparse_level=.1,             # group sparsity level
    sparse_level=.1,               # sparsity level
    err_cov_type='equi'            # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

params_idntity = dict(
    n=50,                          # num obs
    p=20,                          # num predictors
    q=5,                           # num tasks
    sigma=.75,                     # coeff variance
    corr_x=.7,                     # grid of values for rho (correlation coefficient)
    sigma_err=1,                   # correlation coefficient for predictors
    g_sparse_level=.0,             # group sparsity level
    sparse_level=.2,               # sparsity level
    err_cov_type='identity'        # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

settings_dict = dict(
    ar_sparse=params_ar_sparse,
    ar_dense=params_ar_dense,
    equi=params_equi,
    fgn=params_fgn,
    identity=params_idntity
)


def get_simulation_settings(simulation_name):
    return settings_dict[simulation_name]
