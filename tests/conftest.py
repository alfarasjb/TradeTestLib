import pytest 
import pandas as pd 
from src.tradetestlib import * 

@pytest.fixture
def sim():
    symbol = 'CHFSGD'
    tf = 'm5'
    train_data = pd.read_csv('./tests/train.csv', index_col = 'time')
    train_data.index = pd.to_datetime(train_data.index)
    test_data = pd.read_csv('./tests/test.csv', index_col = 'time')
    test_data.index = pd.to_datetime(test_data.index)

    return Simulation(symbol = symbol,
                 timeframe = tf, 
                 train_raw = train_data, 
                 test_raw = test_data,
                 lot = 0.1, 
                 starting_balance = 100000,
                 hold_time = 5, 
                 show_properties = True, 
                 commission = 3.5,
                 max_loss_pct = 0.01, 
                 orders = 'all')
