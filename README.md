# `tradetestlib`: A MetaTrader5 backtesting tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/tradetestlib.svg)](https://badge.fury.io/py/tradetestlib)

`tradetestlib` is a backtesting library built to integrate with MetaTrader5, with the purpose being able to provide a broad overview of a trading strategy/idea, more specifically, an evaluation of a strategy. Some of the main evaluation metrics used in this project are the sharpe ratio, and profit factor. 

`tradetestlib` also provides the option to optimize a strategy by using a Grid Search algorithm for hyperparameter tuning. 

Currently, hyperparameters are limited to position sizing, and exposure. 

A demonstration can be found [here](https://github.com/alfarasjb/Alpha-Research/blob/main/backtest_demo/backtest_demo.ipynb)

## Installation 
`tradetestlib` can be installed with `pip` 

```python
pip install tradetestlib
```

## Usage
A simulation instances can be created by calling the Simulation class. 

`symbol` and `timeframe` are only used for naming conventions, for comparing a basket of assets.

`train_raw` and `test_raw` are dataframes that contain `Open, High, Low, Close, signal, and true_signal` columns.

`lot` is the lot size used to trade

`starting_balance` is the initial deposit of the simulation in USD.
```python
from tradetestlib import Simulation

sim = Simulation(
    symbol = symbol,
    timeframe = tf,
    train_raw = train,
    test_raw = test,
    lot = 1, 
    starting_balance = 100000    
)
```
Optimization can be used by creating a `params` dictionary with the required hyperparameters. 

`run_grid_search` returns the optimal configuration, and overall testing set. 

Optimized hyperparameters may also be accessed as attributes, which can then be used to create a final simulation instance, to verify result with the test set. 

```python
from tradetestlib import Optimize

params = {
    'lot' : [1,2],
    'hold_time': [5, 10],
    'max_loss': [0.005, 0.01]
}

o = Optimize(symbol = symbol, 
    timeframe = tf, 
    train = train, 
    test = test,
    metric = 'sharpe_ratio',
    how = 'maximize')

best, df = o.run_grid_search(params)

o.optimized_lot # best lot 
o.optimized_holdtime # best holdtime
o.optimized_max_loss # best exposure 
```
