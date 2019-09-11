import argparse
import logging
import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('PS')  # noqa: fix mac OS issue
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, MultiTaskLassoCV, RidgeCV
from tqdm import tqdm

from mrrce import MrRCE
from simulations.parameters import RHOS, get_simulation_settings
from simulations.simulation_utils import generate_data, model_error


parser = argparse.ArgumentParser(description='MrRCE simulations.')
parser.add_argument(
    '--simulation-name',
    required=True,
    help="simulation mane, one of ['ar_dense', 'ar_sparse', 'fgn', 'equi', 'identity']"
)
parser.add_argument('--n', type=int, default=200, help='number of repetitions')
parser.add_argument('--output-path', default='output', help='output folder')
parser.add_argument('--save-data', action='store_true', help='whether to save the simulation data')
args = parser.parse_args()

simulation_params = get_simulation_settings(args.simulation_name)

# create folder structure
out_path = Path(args.output_path)
plots_path = out_path / "plots"
plots_path.mkdir(parents=True, exist_ok=True)
results_path = out_path / "results"
results_path.mkdir(parents=True, exist_ok=True)
if args.save_data:
    data_path = out_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# create a file handler
handler = logging.FileHandler(results_path / f"simulation_{args.simulation_name}.log")
handler.setLevel(logging.INFO)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(handler)

logging.info(f"running simulation {args.simulation_name} with {args.n} replications.")


np.random.seed(1)  # for reproducibility
results = []
convergence_results = []
data = []

with warnings.catch_warnings():
    # No need to see the convergence warnings on this grid:
    # there will always be points that will not converge
    # during the cross-validation
    warnings.simplefilter('ignore', ConvergenceWarning)
    pass

for rep in tqdm(range(1, args.n + 1), desc="repetition loop"):
    for rho in tqdm(RHOS, desc="rho values loop"):
        with warnings.catch_warnings():  # for clean output
            warnings.simplefilter('ignore', ConvergenceWarning)
            warnings.simplefilter('ignore', RuntimeWarning)
            os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses (n_jobs > 1)

            X, Y, B, Sigma, Sigma_X = generate_data(rho=rho, **simulation_params)
            # MrRCE
            mrrce = MrRCE(glasso_max_iter=200, max_iter=150, tol_glasso=1e-3)
            mrrce.fit(X, Y)
            # OLS
            lm = LinearRegression(fit_intercept=False).fit(X, Y)
            B_ols = np.matrix(lm.coef_.transpose())
            # Ridge
            ridge = RidgeCV(fit_intercept=False).fit(X, Y)
            B_ridge = np.matrix(ridge.coef_.transpose())
            # Group Lasso
            gl = MultiTaskLassoCV(fit_intercept=False, cv=3).fit(X, Y)
            B_gl = np.matrix(gl.coef_.T)
            # Results
            results.append(
                dict(
                    rho=rho,
                    rho_hat=mrrce.rho,
                    sigma_hat=mrrce.sigma,
                    MrRCE=model_error(B, mrrce.Gamma, Sigma_X),
                    OLS=model_error(B, B_ols, Sigma_X),
                    Ridge=model_error(B, B_ridge, Sigma_X),
                    GroupLasso=model_error(B, B_gl, Sigma_X)
                )
            )
            convergence_results.append(
                dict(
                    iter_number=rep,
                    rho=rho,
                    convergence_path=mrrce.convergence_path,
                    n_iters=mrrce.n_iters,
                )
            )

            data.append((X, Y, B))

# create a data frame with the data
results_df = pd.DataFrame(results)
results_df.to_csv(
    (results_path / f"simulation_results_{args.simulation_name}.csv").as_posix(),
    index=False
)

# convergence
convergence_df = pd.DataFrame(convergence_results)
convergence_df.to_json(
    (results_path / f"convergence_results_{args.simulation_name}.jsonl").as_posix(),
    orient="records",
    lines=True
)

# save data if needed
if args.save_data:
    pickle.dump(data, open(data_path / f"simulation_data_{args.simulation_name}.pkl", "wb"))

# plot
to_plot = (
    results_df[['rho', 'GroupLasso', 'MrRCE', 'OLS', 'Ridge']].
    groupby('rho', as_index=False).
    mean().
    melt(
        id_vars='rho',
        var_name='method',
        value_name='ME'
    )
)

fig, ax = plt.subplots(figsize=(15, 5))

for method in to_plot.method.unique():
    curr = to_plot.loc[to_plot.method == method].copy()
    plt.plot(curr.rho, curr.ME, 'o--')

ax.legend(
    to_plot.method.unique(),
    loc='upper center',
    fancybox=True,
    shadow=True,
    ncol=5,
    fontsize='x-large',
    bbox_to_anchor=(0.5, 1.15)
)
ax.set_xlabel('rho', fontsize='x-large')
ax.set_ylabel('Model Error', fontsize='x-large')
# save
plt.savefig((plots_path / f"simulation_plot_{args.simulation_name}.png").as_posix())

logging.info("done")
