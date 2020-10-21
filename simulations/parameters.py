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

# extra - for revision
params_ar_dense_n100 = dict(
    n=100,                          # num obs
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

params_ar_dense_n150 = dict(
    n=150,                          # num obs
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

params_ar_dense_n200 = dict(
    n=200,                          # num obs
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

# number of regressors
params_ar_dense_p30 = dict(
    n=50,                          # num obs
    p=30,                          # num predictors
    q=5,                           # num tasks
    sigma=1,                       # coeff variance
    corr_x=.7,                     # grid of values for rho (correlation coefficient)
    sigma_err=1,                   # correlation coefficient for predictors
    err_corr=.75,                  # correlation coefficient for errors
    g_sparse_level=.0,             # group sparsity level
    sparse_level=.0,               # sparsity level
    err_cov_type='ar'              # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

params_ar_dense_p40 = dict(
    n=50,                          # num obs
    p=40,                          # num predictors
    q=5,                           # num tasks
    sigma=1,                       # coeff variance
    corr_x=.7,                     # grid of values for rho (correlation coefficient)
    sigma_err=1,                   # correlation coefficient for predictors
    err_corr=.75,                  # correlation coefficient for errors
    g_sparse_level=.0,             # group sparsity level
    sparse_level=.0,               # sparsity level
    err_cov_type='ar'              # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

params_ar_dense_p50 = dict(
    n=50,                          # num obs
    p=50,                          # num predictors
    q=5,                           # num tasks
    sigma=1,                       # coeff variance
    corr_x=.7,                     # grid of values for rho (correlation coefficient)
    sigma_err=1,                   # correlation coefficient for predictors
    err_corr=.75,                  # correlation coefficient for errors
    g_sparse_level=.0,             # group sparsity level
    sparse_level=.0,               # sparsity level
    err_cov_type='ar'              # error covariance form. One of ['ar', 'equi', 'fgn', 'identity']
)

settings_dict = dict(
    # main text
    ar_sparse=params_ar_sparse,
    ar_dense=params_ar_dense,
    equi=params_equi,
    fgn=params_fgn,
    identity=params_idntity,

    # extra
    ar_dense_n100=params_ar_dense_n100,
    ar_dense_n150=params_ar_dense_n150,
    ar_dense_n200=params_ar_dense_n200,

    ar_dense_p30=params_ar_dense_p30,
    ar_dense_p40=params_ar_dense_p40,
    ar_dense_p50=params_ar_dense_p50,
)


def get_simulation_settings(simulation_name):
    return settings_dict[simulation_name]
