import pandas as pd 
import numpy as np 
from tradetestlib.simulation import Simulation
import tqdm 
from itertools import product 

class Optimize:
    """
    Performs a grid-search algorithm to tune hyperparameters that yield in the best performing metric.
    
    Parameters
    ----------
    symbol: str
        symbol to test
        
    timeframe: str
        timeframe to test
    
    train: pd.DataFrame
        training dataset
        
    test: pd.DataFrame
        testing dataset
        
    metric: str
        performance metric for evaluating strategy performance using specified hyperparameter configuration
    
    how: str
        target value for specified metric
        
        maximize, minimize
        
    dataset: str
        dataset name
    
    """
    def __init__(self, 
                 symbol: str, 
                 timeframe: str, 
                 train: pd.DataFrame, 
                 test: pd.DataFrame,
                 metric: str = 'sharpe_ratio', 
                 how: str = 'maximize', 
                 dataset: str = 'train'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.train = train
        self.test = test
        self.metric = metric
        self.how = how
        self.dataset = dataset
        self.print_optimization_parameters()
        
    def print_optimization_parameters(self):
        """
        Prints optimization parameters
        """
        
        print('========== OPTIMIZATION PARAMETERS ==========')
        print('Symbol: ', self.symbol)
        print('Timeframe: ', self.timeframe)
        print('Metric: ', self.metric)
        print('Target: ', self.how)
        print('Optimization Method: Grid Search')
        print('Dataset: ', self.dataset)
        print('========== OPTIMIZATION PARAMETERS ==========')
    
    
    def run_grid_search(self, params: dict):
        """
        Runs a grid search algorithm using the params dictionary, and creates individual simulation instances, and evaluates its performance
        based on the specified evaluation metric and target.
        
        
        Parameters
        ----------
        params: dict
            dictionary containing different parameter combinations to test. 
            
        Returns
        -------
        optimized_parameters
        """
        print('Lots: ', params['lot'])
        print('Hold Time: ', params['hold_time'])
        print('Max Loss: ', params['max_loss'])
        print('========== RUNNING OPTIMIZATION ==========')
        sim_elements = []
        summaries = []
        sims = []
        for l,h,m in tqdm(product(params['lot'], params['hold_time'],params['max_loss'])):
            sim = Simulation(symbol = self.symbol, timeframe = self.timeframe, train_raw = self.train, test_raw = self.test, 
                             lot = l, 
                             starting_balance = 100000, 
                             hold_time = h, 
                             #trading_hours = th, 
                             show_properties = False,
                             commission = 3.5,
                             max_loss_pct = m)
            
            if self.dataset == 'combined':
                data = sim.combined_summary
            elif self.dataset == 'train':
                data = sim.train_filtered_summary
                spread = sim.train_filtered_evaluation.spread
            elif self.dataset == 'test':
                data = sim.test_summary
            elif self.dataset == 'filtered':
                data = sim.test_filtered_summary
                spread = sim.test_filtered_evaluation.spread
            else:
                raise ValueError('Invalid Dataset Selection')
                
            if self.metric not in data.index:
                raise ValueError('Metric not found in data')
                
            #sims.append(sim)    
            summaries.append(data)
            y_value = data.loc[self.metric].item()
            sim_elements.append([l,h,m,spread,y_value])
            
        
        optimization_df = pd.DataFrame(sim_elements, columns = ['lot', 'holdtime', 'max_loss', 'spread', self.metric])
        
        
        if self.how == 'maximize':
            self.optimized_parameters = optimization_df.loc[optimization_df[self.metric] == optimization_df[self.metric].max()]
        elif self.how == 'minimize':
            self.optimized_parameters = optimization_df.loc[optimization_df[self.metric] == optimization_df[self.metric].min()]
        
        else: 
            raise ValueError('Invalid Optimization')
        
        self.optimized_lot = self.optimized_parameters[:1]['lot'].item()
        self.optimized_holdtime = self.optimized_parameters[:1]['holdtime'].item()
        self.optimized_max_loss = self.optimized_parameters[:1]['max_loss'].item()
        
        return self.optimized_parameters, optimization_df
        
            
