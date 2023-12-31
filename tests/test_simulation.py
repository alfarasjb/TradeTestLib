import sys
#sys.path.append('C:\\Users\\JB\\Desktop\\Alpha\\Repositories\\TradeTestLib')
from tradetestlib import *
import pytest

@pytest.mark.usefixtures('sim')
def test_dataframe_processing(sim):
    """""" 
    raw = sim.train_data
    max_loss = sim.max_loss
    assert len(raw.query('signal != 0 & trade_diff == 0 & net_profit > 0')) == 0
    assert len(raw.query('signal != true_signal & raw_profit > 0')) == 0
    assert len(raw.query(f'signal == true_signal & raw_profit < 0 & raw_profit != -{max_loss}')) == 0
    assert len(raw.query('signal != true_signal & trade_diff > 0')) == 0
    assert len(raw.query('raw_profit == 0 & spread_adjusted_profit > 0 & signal != 0')) == 0
    assert len(raw.query('raw_profit < 0 & raw_profit < spread_adjusted_profit')) == 0
    assert len(raw.query('spread_adj_trade_diff < 0 & net_profit > 0')) == 0
    
    assert len(raw.query('signal == 0 & net_profit < 0')) == 0
    assert len(raw.query('signal == 0 & net_profit > 0')) == 0
    assert len(raw.query('match == 0 & net_profit > 0')) == 0
    assert len(raw.query('match == 0 & net_profit < 0')) == 0
    assert len(raw.query('raw_profit < 0 & raw_profit < spread_adjusted_profit')) == 0

@pytest.mark.usefixtures('sim')
def test_plot_returns_comparison(sim):
    with pytest.raises(ValueError):
        sim.plot_returns_comparison(dataset='full',compare='balance')

    with pytest.raises(ValueError):
        sim.plot_returns_comparison(dataset='train',compare='equity')

@pytest.mark.usefixtures('sim')
def test_plot_periodic_statistics(sim):
    with pytest.raises(ValueError):
        sim.plot_periodic_statistics(metric = 'volatility')
        
@pytest.mark.usefixures('sim')
def test_select_dataset(sim):
    with pytest.raises(ValueError):
        sim.select_dataset('full')