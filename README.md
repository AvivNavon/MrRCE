# MrRCE

Implementation of _Capturing Between-Tasks Covariance and Similarities Using Multivariate Linear Mixed Models_, Volume 14, Number 2 (2020), Electronic Journal of Statistics. 

Specifically, this is an implementation of the _Multivariate random Regression with Covariance Estimation_ (MrRCE) algorithm, 
designed to take advantage of correlations and similarities among responses and coefficients, in a multi-task regression framework
(see the [paper](https://projecteuclid.org/euclid.ejs/1603245663) for details).

## Useful Links

- [Paper](https://projecteuclid.org/euclid.ejs/1603245663)
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

usage: run_simulation.py [-h] --simulation-name SIMULATION_NAME [--n N]
                         [--output-path OUTPUT_PATH] [--save-data]

MrRCE simulations.

optional arguments:
  -h, --help            show this help message and exit
  --simulation-name SIMULATION_NAME
                        simulation mane, one of ['ar_dense', 'ar_sparse',
                        'fgn', 'equi', 'identity']
  --n N                 number of repetitions
  --output-path OUTPUT_PATH
                        output folder
  --save-data           whether to save the simulation data


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
will run the equicorrelation (covariance matrix) simulation with 200 replications (for each value of the correlation coefficient, rho), and the outcome should look like the following:

 <p align="center"> 
    <img src="https://github.com/AvivNavon/MrRCE/blob/master/output/plots/simulation_plot_equi.png" width="700">
 </p>

## Example

Example of running MrRCE:

```python
from mrrce import MrRCE
mrrce = MrRCE()
mrrce.fit(X, Y) # X and Y are matrices of shapes (n,p) and (n,q) correspondingly
mrrce.Gamma     # estimated coefficient matrix
mrrce.rho       # estimated correlation coefficient
mrrce.sigma     # estimated sd for coefficients
mrrce.Sigma     # estimated covariance matrix for the error terms
mrrce.Omega     # estimated precision matrix for the error terms
```

See full example at [this](https://github.com/AvivNavon/MrRCE/blob/master/example.ipynb) notebook.

## Citation

If you find `MrRCE` to be useful in your own research, please consider citing the following paper:

```bib
@ARTICLE{NavRos2020,
    AUTHOR = {Aviv Navon and Saharon Rosset},
     TITLE = {Capturing between-tasks covariance and similarities using multivariate linear mixed models},
   JOURNAL = {Electron. J. Statist.},
  FJOURNAL = {Electronic Journal of Statistics},
      YEAR = {2020},
    VOLUME = {14},
    NUMBER = {2},
     PAGES = {3821-3844},
      ISSN = {1935-7524},
       DOI = {10.1214/20-EJS1764},
      SICI = {1935-7524(2020)14:2<3821:CBTCAS>2.0.CO;2-2},
}
```