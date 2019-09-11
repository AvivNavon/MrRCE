# Capturing Between-Tasks Covariance and Similarities Using Multivariate Linear Mixed Models

This is an implementation of the Multivariate random Regression with Covariance Estimation (MrRCE) algorithm, 
designed to take advantage of correlations and similarities among responses and coefficients, in a multi-task regression framework
(see the [paper](https://arxiv.org/abs/1812.03662) for details).

## Useful Links

- [GitHub repo](https://github.com/AvivNavon/MrRCE)
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

Running the simulations will create a file with the name `simulation_results_<simulation name>.csv` with Model Error (ME) for each method and replication.
In addition, it will create a plot of ME against the correlation parameter, and save it as `simulation_plot_<simulation name>.png`. The files will be saved into a `results` and `plots` folders.

### Running Simulations

For help run:

```
$ python run_simulation.py --help

usage: run_simulation.py [-h] [--simulation-name SIMULATION_NAME] [--n N]
                         [--output-path OUTPUT_PATH]

MrRCE simulations.

optional arguments:
  -h, --help            show this help message and exit
  --simulation-name SIMULATION_NAME
                        simulation mane, one of ['ar_dense', 'ar_sparse',
                        'fgn', 'equi', 'identity']
  --n N                 number of repetitions
  --output-path OUTPUT_PATH
                        output folder

```

```
python run_simulation.py --simulation-name <simulation name>
```

This will run the simulation <simulation name> with the default 200 replication. You can also run:

```
python run_simulation.py --simulation-name <simulation name> --n <N>
```
where `<N>` is an integer for the number of replications. For example, the following line,

```
python run_simulation.py --simulation-name equi --n 200
```
will run the equicorrelation (matrix) simulation with 200 replications (for each value of the correlation coefficient, rho), and the outcome should look like the following:

 <p align="center"> 
    <img src="https://github.com/AvivNavon/MrRCE/blob/master/output/plots/simulation_plot_equi.png" width="700">
 </p>

## Example

Example of running MrRCE:

```python
from mrrce import MrRCE
m = MrRCE()
m.fit(X, Y) # X and Y are matrices of shapes (n,p) and (n,q) correspondingly
m.Gamma     # estimated coefficient matrix
m.rho       # estimated correlation coefficient
m.sigma     # estimated sd for coefficients
m.Sigma     # estimated covariance matrix for the error terms
m.Omega     # estimated precision matrix for the error terms
```

See full example at [this](https://github.com/AvivNavon/MrRCE/blob/master/example.ipynb) notebook.