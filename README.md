# Capturing Between-Tasks Covariance and Similarities Using Multivariate Linear Mixed Models

This is the implementation of the Multivariate random Regression with Covariance Estimation (MrRCE) algorithm.

## Useful Links

- [Code and examples](https://github.com/AvivNavon/MrRCE)
- [Example](https://github.com/AvivNavon/MrRCE/blob/master/example.ipynb)

## Installation and Requirements

Clone this repo and run

```
pip install -r requirements.txt
```

## Simulations

There are five simulations that can be easily executed:

- Autoregressive (AR) error covariance with dense coefficient matrix (`ar_dense`)
- AR error covariance with sparse coefficient matrix (`ar_sparse`)
- Fractional Gaussian Noise (FGN) error covariance (`fgn`)
- Equicorrelation error covariance (`equi`)
- Identity error covariance (`identity`)

The names in parentheses are the simulations names.

Running the simulations will create a file call `simulation_results_<simulation name>.csv` with Model Error (ME) for each method and replication.
In addition, it will create a plot of ME against the correlation parameter, and save it as `simulation_plot_<simulation name>.png`. It will also try and 
create folders for results and plots, and save the relevant files into these locations.

### Option 1

```
python <simulation name>
```

This will run simulation <simulation name> with the default 200 replication. You can also run:

```
python <simulation name> <N>
```
where `<N>` is an integer for the number of replications. For example, the following line,

```
python equi 100
```
will run the Equicorrelation simulation with 100 replications (for each value of the correlation coefficient $\rho$).

### Option 2

```
./run_simulation.sh <simulation name> [<N>]
```
## Example

Example of running MrRCE:

```
>>> from mrrce import MrRCE
>>> m = MrRCE()
>>> m.fit(X, Y)  # X and Y are matrices of shapes (n,p) and (n,q) correspondingly
>>> m.B_hat # estimated coefficient matrix
>>> m.rho # estimated correlation coefficient
>>> m.sigma # estimated variance for coefficients
>>> m.Sigma # estimated covariance matrix for the error terms
>>> m.Omega # estimated precision matrix for the error terms
```

See full example at [this](https://github.com/AvivNavon/MrRCE/blob/master/example.ipynb) notebook.