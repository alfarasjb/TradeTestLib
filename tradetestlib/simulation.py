import warnings
warnings.filterwarnings('ignore')
import pandas as pd 
import numpy as np
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
import seaborn as sns
from statsmodels.stats.stattools import jarque_bera

from tradetestlib.evaluation import Evaluation


bg_color_light = '#3A3B3C'
bg_color = '#242526'
fg_color = '#B0B3B8'

plt.rcParams['font.size'] = 11 
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['axes.labelcolor'] = fg_color
plt.rcParams['axes.titlecolor'] = fg_color
plt.rcParams['axes.facecolor'] = bg_color
plt.rcParams['xtick.color'] = fg_color
plt.rcParams['xtick.labelcolor'] = fg_color
plt.rcParams['ytick.color'] = fg_color
plt.rcParams['ytick.labelcolor'] = fg_color
plt.rcParams['grid.color'] = bg_color_light
plt.rcParams['figure.facecolor'] = bg_color
plt.rcParams['legend.labelcolor'] = fg_color



class Simulation:
    """
    Simulation is a library for evaluating trading strategies for MetaTrader5. This takes into account the impact of 
    bid-ask spreads, and transaction costs.
    
    Creates a summary of overall strategy performance showing key evaluation metrics such as Sharpe Ratio, Profit Factor, etc. 
    
    Allows for automatic optimization by time intervals: filters times during the trading day where the strategy underperforms. 
    
    
    
    Parameters
    ----------
    
    symbol:str
        Symbol to test
        
    timeframe:str
        Timeframe of test
        
    train_raw: pd.DataFrame
        Raw training dataframe containing OHLC, and signal columns
        
        Columns required: Datetime, Open, High, Low, Close, Signal, True Signal
    
    test_raw: pd.DataFrame
        Raw testing dataframe containing OHLC, and signal columns
        
    lot: float
        lot size for profit calculation
        
    starting_balance: int
        starting balance for backtest
        
    commission: float
        broker commission per lot per side 
        
    max_loss_pct: float
        percent of balance used for maximum allowable loss
        
    hold_time: int
        number of candles to hold a trade
        
    orders: str
        orders to execute
        
        long, short, all 
        
    show_properties: bool
        prints backtest properties 

    filter_by: str
        filtering method used for time interval optimization
        
    num_elements: int 
        number of elements to use for interval filtering, ranked from best to worst performance

    spread: int 
        overrides bid/ask spread from raw input 

    exclude_time: list 
        time in format '%H:%M' to exclude trading activity

    trade_time: list
        time in format '%H:%M' to allow trading 

    trading_window_start: int 
        trading window start (hours, GMT+2)

    trading_window_end: int 
        trading window end (hours, GMT+2)

    default_figsize: tuple 
        default figsize for plots
    
    Methods
    -------
    print_backtest_properties()
        Method for printing backtest properties
        
    build_dataframe()
        method for building main backtest dataframe. contains relevant information for assessing strategy performance
        
    summary()
        creates a summary dataframe containing strategy performance metrics
        
    plot_returns_comparison()
        generates a plot evaluating overall performance cost of broker commissions
        
    plot_periodic_statistics()
        generates a plot of statistics by period/interval
        
    train_test_comparison()
        compares in-sample and out-of-samples datasets and generates a dataframe
        
    plot_performance_comparison()
        generates a figure on in-sample and out-of-sample performance comparison, comparing balance, peak balance, and interval-filtered datasets.
        
    plot_transaction_cost_composition()
        generates a plot comparing raw propfits and adjusting for transaction costs and spread
        
    plot_interval_filter_comparison()
        plots an equity curve comparing results of a raw dataset, compared with a dataset with interval filtering. 
    
    best_interval_by_profit()
        Builds a dataframe of best time intervals. 
        
        Ranks the time intervals by getting an aggregated mean of net profit. 
        
        Time intervals that are the most profitable on average are ranked first. 
        
        
    best_interval_by_cost_delta()
        Builds a dataframe of best performing time intervals. 
        
        Ranks the time intervals by getting an aggregated sum of net_profit vs net transaction costs. 
        
        Time intervals that outperforms net transaction costs are ranked first. 
    
    plot_returns_distribution()
        Plots a distribution of cost adjusted profit.

    plot_equity_curve()
        Plots equity curve and drawdown 

    plot_cumulative_gain_by_month()
        Plots cumulative gain by month. 

    plot_cumulative_gain_by_year()
        Plots cumulative gain by year.    
    
    select_dataset()
        Helper function for selecting dataset. Primarily used for plotting

    filter_default()
        Helper function for identifying filtering method
    """
    
    def __init__(self, 
                 symbol:str, 
                 timeframe:str, 
                 train_raw: pd.DataFrame, 
                 test_raw: pd.DataFrame, 
                 lot: float, 
                 starting_balance: int, 
                 commission: float = 3.5, 
                 max_loss_pct: float = None,
                 hold_time: int = 1, 
                 orders:str = 'all',
                 show_properties: bool = True,
                 filter_by: str = None,
                 num_elements: int = 10,
                 spread: int = None,
                 exclude_time: list = [],
                 trade_time: list = [],
                 trading_window_start: int = None, 
                 trading_window_end: int = None,
                 default_figsize: tuple = (14, 6),
                ):
        
        self.timeframes = ['m1','m5','m15','m30','h1','h4','d1']
        self.intraday = ['m1','m5','m15','m30','h1','h4']
        self.htf = ['d1','w1','mn1']
        self.valid_orders = ['long','short','all']
        
        assert timeframe in self.timeframes, 'Invalid Timeframe'
        assert lot > 0, 'Invalid Lot'
        assert starting_balance > 0, 'Invalid starting_balance'
        assert commission >= 0, 'Invalid Commission'
        assert hold_time >= 0, 'Invalid Hold Time'
        assert orders in self.valid_orders, 'Invalid Orders'

        if len(exclude_time) != 0 and len(trade_time) != 0:
            raise ValueError('Conflicting trading sessions. Specifying both exclude_time and trade_time is not allowed.')
        
        self.symbol = symbol
        self.timeframe = timeframe
        
        """
        Raw datasets that contain strategy signal, and true signal columns
        """
        self.train_raw = train_raw.copy()
        self.test_raw = test_raw.copy()

        self.train_raw.columns = [tr.lower() for tr in self.train_raw.columns]
        self.test_raw.columns = [te.lower() for te in self.test_raw.columns]
        
        """
        Trade Parameters
        """
        self.lot = lot
        self.starting_balance = starting_balance
        self.commission = (commission * self.lot) * 2
        self.max_loss_pct = max_loss_pct if max_loss_pct is not None else 1
        self.max_loss = self.max_loss_pct * self.starting_balance
        
        self.hold_time = hold_time
        self.orders = orders
        
        self.contract_size, self.trade_point, self.trade_tick, self.spread = self.request_mt5_symbol_info(self.symbol)
        self.spread = spread if spread is not None else self.spread
        
        self.exclude_time = exclude_time # time in format "HH:MM" to exclude trading activity 
        self.trade_time = trade_time # time in format "HH:MM" to trade only this 
        
        self.trading_window_start = trading_window_start
        self.trading_window_end = trading_window_end
        self.default_figsize = default_figsize

        self.show_properties = show_properties

        self.filter_by = self.filter_default(self.timeframe) if filter_by is None else filter_by
        self.num_elements = num_elements

        if self.show_properties:
            self.print_backtest_properties()
        
        """
        Tunable Hyperparameters: 
            lot: lot size
            
            holdtime: time intervals to hold a trade
            
            max_loss_pct: maximum allowable loss as a percent of deposit capital
        """
        self.hyperparameters = {
            'lot' : self.lot,
            'holdtime' : self.hold_time,
            'max_loss_pct' : self.max_loss_pct
        }
    
        """
        Dataframes
        ----------
        Creates three main, uncalibrated dataframes: 
            train_data, test_data, combined
            
            train data: data used for calibrating best interval
            test data: initially used for testing uncalibrated train data
            combined: combined set of train, and test data
            
        Filtered dataframes are created from calibrating raw train_data, identifying best time intervals for 
        strategy. 
        
        train_filtered: used for grid-search optimization
        
        test_filtered: used for final evaluation
        
        combined_filtered: used for comparing raw, unfiltered dataset to calibrated and optimized dataset. 
        
        
        The Datasets also have 2 additional components:
            1. Evaluation
            2. Summary
            
            Evaluation
                an object to hold alpha performance metrics such as sharpe ratio, profit factor, max dd
                
                this is used for optimization and cherry-picking assets in which the alpha in test is most effective
                
            Summary
                a summary dataframe used purely for evaluation purposes
        
        """

        self.train_data = self.build_dataframe(self.train_raw.copy())
        self.train_evaluation = Evaluation(self.train_data, self.hyperparameters)
        self.train_summary = self.summary(self.train_evaluation, 'train')
        
        self.test_data = self.build_dataframe(self.test_raw.copy())
        self.test_evaluation = Evaluation(self.test_data, self.hyperparameters)
        self.test_summary = self.summary(self.test_evaluation, 'test')
        
        self.combined = self.build_dataframe(pd.concat([self.train_raw.copy(), self.test_raw.copy()], axis = 0))
        self.combined_evaluation = Evaluation(self.combined, self.hyperparameters)
        self.combined_summary = self.summary(self.combined_evaluation, 'combined')
        
        self.train_filtered = self.build_dataframe(self.train_raw.copy(), filtered = True)
        self.train_filtered_evaluation = Evaluation(self.train_filtered, self.hyperparameters)
        self.train_filtered_summary = self.summary(self.train_filtered_evaluation, 'train_filtered')
        
        self.test_filtered = self.build_dataframe(self.test_raw.copy(), filtered = True)
        self.test_filtered_evaluation = Evaluation(self.test_filtered, self.hyperparameters)
        self.test_filtered_summary = self.summary(self.test_filtered_evaluation, 'test_filtered')
    
        
        self.combined_filtered = self.build_dataframe(pd.concat([self.train_raw.copy(), self.test_raw.copy()], axis = 0), filtered = True)
        self.combined_filtered_evaluation = Evaluation(self.combined_filtered, self.hyperparameters)
        self.combined_filtered_summary = self.summary(self.combined_filtered_evaluation, 'combined_filtered')
        
    def print_backtest_properties(self):
        """
        Method to print backtest properties
        """
        print('---------- BACKTEST PROPERTIES ----------')
        print('Symbol: ', self.symbol)
        print('Timeframe: ', self.timeframe)
        print('Trade Point: ', self.trade_point)
        print('Trade Tick: ', self.trade_tick)
        print('Lot Size: ', self.lot)
        print('Hold Time: ', self.hold_time)
        print('Max Loss Pct: ', self.max_loss_pct)
        print('Max Loss: ', self.max_loss)
        print('Commission: ', self.commission)
        print('Contract Size: ', self.contract_size)
        print('Spread: ', self.spread)
        print('Starting Balance: ', self.starting_balance)
        print('---------- BACKTEST PROPERTIES ----------')

    @staticmethod
    def request_mt5_symbol_info(symbol):
        try:
            element = mt5.symbol_info(symbol)._asdict()
            contract_size = element['trade_contract_size']
            trade_point = element['point'] # num ticks
            trade_tick = element['trade_tick_value'] # value per tick in USD per 1 standard lot
            spread = element['spread'] if element['spread'] != 0 else 1# approximation calculated from weekend spread

            return contract_size, trade_point, trade_tick, spread
        
        except AttributeError:
            print('MT5 is not running. Run mt5.initialize()')
            return 100000, 0.00001, 1, 1 
            
        
    def build_dataframe(self, data: pd.DataFrame = None, filtered:bool = False):
        """
        Builds main dataframe which serves as reference for optimization and evaluation.
        
        Parameters
        ----------
        
        data: pd.DataFrame
            dataset to backtest
            
        filtered: bool
            specified if the required dataframe is an interval-filtered dataframe. 
            
        
        Column Description
        ------------------
        
        signal
            strategy signal
            1: long, -1: short
        
        valid
            1: if valid, 0: if invalid
            
            used for filtering trades based on time interval filtering, 
            allows/prevents trades based on time intervals/sessions
            
            determines if trade is valid for current time interval. 
            modified by best_interval_by_cost_delta
            
        pct_change
            pct change from previous closing price
        
        strategy_returns
            cummulative sum of strategy returns. first indication if alpha is potentially profitable
        
        true_returns
            true market returns. assumes that prediction accuracy is 100% 
            used for initial performance evaluation of alpha.
        
        commission
            broker commission. varies with instrument
            IC-Markets MT5: 
                3.5 per lot per side FX CFD
        
        candle_body
            candle size
        
        lot
            lot size/volume
        
        close_price
            trade close price depending on holding time (intervals) 
            used for calculating trade diff, and p/l
            
        
        trade_diff
            overall price change of trade. 
            used for calculating p/l
        
        lowest
            lowest price reached during trade duration. 
            used for determining if trade was stopped out
            shows maximum drawdown for long positions
        
        highest
            highest price reached during trade duration
            used for determining if trade was stopped out
            shows maximum drawdown for short positions
        
        lowest_trade_diff
            price difference from trade opening price to lowest price reached
            used for calculating maximum drawdown for long positions
        
        highest_trade_diff
            price difference from trade opening price to highest price reached
            used for calculating maximum drawdown for short positions
        
        spread
            bid-ask spread approximated from current market spread at the time of testing. 
            backtest results may vary based on spread.
        
        spread_factor
            bid-ask spread scaled to asset price. 
        
        raw_profit
            raw profit disregarding spread and transaction costs
        
        raw_stop_diff
            determines price diff, considering long and short positions
            used for calculating drawdown
        
        raw_dd
            maximum drawdown reached during trade duration
        
        running_raw
            cummulative sum of raw profit. used for comparing spread adjusted profit, and total cost adjusted profit
            helps in determining overall impact of transaction costs to alpha performance.
        
        raw_balance
            resulting balance after trade close
        
        spread_adj_trade_diff
            trade difference accounting for current market spread. results may vary based on received spread from MT5
            
        spread_adjusted_profit
            profit resulting in adjusting for bid-ask spread
            
        spread_adj_trade_points 
            trade diff converted to points
            
        spread_adj_stop_diff
            determines price diff considering long and short positions, and accounting for spread
            used for calculating drawdown
            
        spread_adj_dd
            maximum drawdown reached during trade duration, accounting for spread
            
        running_spread_adj
            cummulative sum of spread adjusted profit. used for comparison with raw profit and total cost adjusted profit
            helps in determining overall impact of transaction costs to alpha performance
            
        spread_adj_balance
            resulting balance after trade close, accounting for spread
            
        net_profit
            total cost adjusted profit
            profit after accounting for spread and broker commissions
            
        running_profit
            cummulative sum of net profit
            
        running_profit_pct
            percent change of running profit
        
        balance
            true balance after adjusting for transaction costs
            
        composition 
            shows the composition of cost adjusted profit, to raw profit
            determines 'how much is left on the table'
            
        running_pct_gain
            balance pct change
            
        peak_balance
            peak balance reached from starting simulation. 
            used for calculating drawdown
            
        drawdown
            drawdown after trade close (if any)
        """
        
        if data is None:
            return 
        
        if filtered:
            trading_hours = self.best_interval_by_cost_delta(self.num_elements).index.tolist()
            trading_hours = [t.strftime('%H:%M') for t in trading_hours]
        else:
            trading_hours = []
        

        
        if len(trading_hours) > 0:
            data['valid'] = 0
            
            if self.filter_by == 'time':
                #data['valid'] = data.index.hour.isin(trading_hours).astype('int')
                data.loc[(data.index.strftime('%H:%M').isin(trading_hours)), 'valid'] = 1
            elif self.filter_by == 'day_of_week':
                data.loc[(data.index.dayofweek.isin(trading_hours)), 'valid'] = 1
            else:
                raise ValueError('Invalid Filter')
            
        else: 
            data['valid'] = 1

        # This is working
        #data['valid'] = data.index.hour.isin(trading_hours).astype('int') if len(trading_hours) > 0 else 1
        
        #if len(self.exclude_time > 0):
        #    data.loc[(data.index.time.isin(self.exclude_time)), 'valid'] = 0
        
        #if len(self.trade_time > 0):
        #    data.loc[)()]

        if len(self.exclude_time) > 0:
            data.loc[data.index.strftime('%H:%M').isin(self.exclude_time), 'valid'] = 0

        if len(self.trade_time) > 0:
            data.loc[(data.index.strftime('%H:%M').isin(self.trade_time)) & (data['signal'] != 0), 'valid'] = 1
            data.loc[(~data.index.strftime('%H:%M').isin(self.trade_time)) & (data['signal'] != 0), 'valid'] = 0

        if self.trading_window_start is not None: 
            data.loc[(data.index.hour < self.trading_window_start) & (data['signal'] != 0), 'valid'] = 0
        
        if self.trading_window_end is not None: 
            data.loc[(data.index.hour >= self.trading_window_end) & (data['signal'] != 0), 'valid'] = 0
        
        # modifies signal depending on trade validity
        data['signal'] = data['signal'] * data['valid']
        

        # filters signals only if long or short orders are allowed
        data.loc[data['signal'] == -1, 'signal'] = 0 if self.orders == 'long' else -1
        data.loc[data['signal'] == 1, 'signal'] = 0 if self.orders == 'short' else 1
        
        

        data['pct_change'] = data['close'].pct_change() * 100
        #data['strategy_returns'] = data['pct_change'] * data['signal']
        
        data['commission'] = np.where(data['signal'] != 0, self.commission, 0)
        
        data['candle_body'] = data['close'] - data['open']

        data['lot'] = self.lot
        
        data['close_price'] = data['close'].shift(periods = -(self.hold_time - 1)) if self.hold_time > 0 else data['close']
        
        data['trade_diff'] = 0
        data.loc[data['signal'] !=0, 'trade_diff'] = abs(data['close_price'] - data['open'])

        data.loc[(data['close_price'] > data['open']) & (data['signal'] == -1), 'trade_diff']  = data['trade_diff'] * -1
        data.loc[(data['close_price'] < data['open']) & (data['signal'] == 1), 'trade_diff'] = data['trade_diff'] * -1
        data['match'] = 0 
        data.loc[data['trade_diff'] > 0, 'match'] = 1
        data.loc[data['trade_diff'] < 0, 'match'] = -1
        

        data['lowest'] = data['low'].rolling(self.hold_time).min().shift(1-self.hold_time)
        data['highest'] = data['high'].rolling(self.hold_time).max().shift(1-self.hold_time)
        data['lowest_trade_diff'] = np.where(data['signal'] !=0, data['lowest'] - data['open'], 0)
        data['highest_trade_diff'] = np.where(data['signal'] !=0, data['highest'] - data['open'], 0)
        
        #THIS IS WORKING
        # amount to subtract from sprade diff due to spread. evaluates impact of spread to overall performance
        #self.spread_factor = self.spread * self.trade_point
        #data['spread'] = self.spread 
        #data['spread_factor'] = np.where(data['signal'] != 0, self.spread_factor, 0)

        if 'spread' not in data.columns:
            data['spread'] = self.spread 

        data['spread_factor'] = 0
        data.loc[data['signal'] != 0, 'spread_factor'] = data['spread'] * self.trade_point
        
        # not working properly
        data['closing_spread_factor'] = data['spread_factor'].shift(periods = -(self.hold_time - 1)) if self.hold_time > 0 else data['spread_factor']
        
     
        # CORRECT PROFIT CALCULATION USING MT5 DATA
        #ans = (close_price - open_price) * (1 / pt) * (vol) * (trade_tick)
        
        ## === RAW PROFIT === ##
        data['raw_profit'] = data['trade_diff'] * (1 / self.trade_point) * data['lot'] * self.trade_tick
        data.loc[(data['raw_profit'] < 0) & (data['raw_profit'] < -self.max_loss), 'raw_profit'] = -self.max_loss
        data['raw_stop_diff'] = 0
        data['raw_stop_diff'] = np.where(data['signal'] == 1, abs(data['lowest_trade_diff']), abs(data['raw_stop_diff']))
        data['raw_stop_diff'] = np.where(data['signal'] == -1, abs(data['highest_trade_diff']), abs(data['raw_stop_diff']))
        data['raw_dd'] = -abs(data['raw_stop_diff']) * (1 / self.trade_point) * data['lot'] * self.trade_tick
        data.loc[(data['raw_dd'] < -self.max_loss), 'raw_profit'] = -self.max_loss
        
        data['running_raw'] = data['raw_profit'].cumsum()
        data['raw_balance'] = data['running_raw'] + self.starting_balance
        data.loc[data.index == data.index[0], 'raw_balance'] = self.starting_balance
        ## === RAW PROFIT === ##
        
        ## === SPREAD ADJUSTED PROFIT === ##
        data['spread_adj_trade_diff'] = data['trade_diff'] - data['spread_factor'] 
        data['spread_adjusted_profit'] = (data['spread_adj_trade_diff']) * (1 / self.trade_point) * data['lot'] * self.trade_tick
        data.loc[(data['spread_adjusted_profit'] < 0) & (data['spread_adjusted_profit'] < -self.max_loss), 'spread_adjusted_profit'] = -self.max_loss
        data['spread_adj_trade_points'] = abs(data['spread_adj_trade_diff']) * (1 / self.trade_point)
        data['spread_adj_stop_diff'] = 0
        data['spread_adj_stop_diff'] = np.where(data['signal'] == 1, abs(data['lowest_trade_diff']), abs(data['spread_adj_stop_diff']))
        data['spread_adj_stop_diff'] = np.where(data['signal'] == -1, abs(data['highest_trade_diff']), abs(data['spread_adj_stop_diff']))
        data['spread_adj_dd'] = -abs(data['spread_adj_stop_diff']) * (1 / self.trade_point) * data['lot'] * self.trade_tick
        data.loc[(data['spread_adj_dd'] < -self.max_loss), 'spread_adjusted_profit'] = -self.max_loss
        data['running_spread_adj'] = data['spread_adjusted_profit'].cumsum()
        data['spread_adj_balance'] = data['running_spread_adj'] + self.starting_balance
        ## === SPREAD ADJUSTED PROFIT === ##
        
        ## === NET PROFIT === ##
        data['net_profit'] = np.where(data['signal'] != 0, data['spread_adjusted_profit'] - self.commission, 0)
        data['running_profit'] = data['net_profit'].cumsum()
        data['running_profit_pct'] = data['running_profit'].pct_change() * 100
        data['balance'] = data['running_profit'] + self.starting_balance
        data.loc[data.index == data.index[0], 'balance'] = self.starting_balance
        data['composition'] = ((data['raw_profit'] - data['net_profit']) / data['raw_profit']) * 100
        ## === NET PROFIT === ##
        
        balance_cols = ['balance','spread_adj_balance','raw_balance']
        
        for b in balance_cols:
            data.loc[data[b] < 0, b] = 0
        
        
        data['running_pct_gain'] = (data['balance'].pct_change() * 100).cumsum()
        data['peak_balance'] = data['balance'].cummax()
        data['drawdown'] = (1 - (data['balance'] / data['peak_balance'])) * 100

         ## TIME INTERVALS
        data['hour'] = data.index.hour
        data['min'] = data.index.minute
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['day_of_week'] = data.index.day_of_week
        
        data = data.fillna(0)
        data = data.replace([np.inf, -np.inf], 0).fillna(0)
        data = data[:-self.hold_time]
        return data
        
        
    def summary(self, metrics: Evaluation, test_type: str):
        """
        Creates a summary dataframe which summarizes evaluation metrics for the current testing configuration.
        
        Parameters
        ----------
        metrics: Evaluation
            evaluation metrics object to build
            
        test_type: str
            test type used for naming dataframe columns
        """
        summary_df = metrics.evaluation_dataframe()
        summary_df.columns = [f'{self.symbol}_{self.timeframe}_{test_type}']
        
        
        return summary_df
    
    def plot_returns_comparison(self, dataset: str = 'train', compare: str = 'balance'):
        """
        Generates a plot evaluating the overall performance cost of broker commissions. 
        
        Parameters
        ----------
        
        data: pd.DataFrame
            dataset to plot: train, test, combined
            
        compare: str
            comparison type: balance or returns
        """
        
        data = self.select_dataset(dataset)
        
        if compare == 'balance':
            compare_data = ['spread_adj_balance','balance','raw_balance']
            labels = ['Spread Adjusted Balance', 'Transaction Cost and Spread Adjusted', 'Raw Balance']
            ylabel = 'Balance ($)'
            
        elif compare == 'returns':
            compare_data = ['running_raw','running_profit', 'running_spread_ex']
            labels = ['Non Adjusted returns', 'Transaction Cost Adjusted Returns', 'Spread Excluded Returns']
            ylabel = 'Returns ($)'
            
        else:
            raise ValueError('Invalid Comparison Type. Use: balance, returns')
            
        data[compare_data].plot(figsize = self.default_figsize)
        
        plt.legend(labels = labels)
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.title(f'{dataset.title()} Set Performance Comparison: Raw vs Transaction Cost Adjusted')
        plt.show()
        
    def plot_periodic_statistics(self, groupby = 'hour', metric = 'mean'):
        """
        Generates a plot of statistics by period/interval
        
        Parameters
        ----------
        groupby: str
            Column name used as reference for aggregation
            
            Default: hour
            
            Options: hour, day, day_of_week, month
            
        metric: str
            Aggeration operation
            
            Default: mean
            
            Options: mean, sum, std, count
        
        """
        def group_by(d):
            return d.groupby(groupby)[['net_profit']]
        
        def m(d):
            if metric == 'mean':
                return d.mean()
            elif metric == 'sum':
                return d.sum()
            elif metric == 'std':
                return d.std()
            elif metric == 'count':
                return d.count()
            else:
                raise ValueError('Invalid Metric. Use mean, sum, std, count')
        
        train_grouped = group_by(self.train_data.loc[self.train_data['signal'] != 0])
        test_grouped = group_by(self.test_data.loc[self.test_data['signal'] != 0])
        combined_grouped = group_by(self.combined.loc[self.combined['signal'] != 0])
            
        train_build, test_build, combined_build = m(train_grouped), m(test_grouped), m(combined_grouped)
        main_df = pd.concat([train_build,test_build], axis = 1)
        main_df.columns = ['train_net','test_net']
        main_df.plot(kind = 'bar', figsize = self.default_figsize, alpha=0.6, edgecolor='black')
        
        plt.xlabel(groupby.replace('_',' ').title())
        plt.ylabel(f'Returns {metric.replace("_"," ").title()}')
        plt.legend(labels = ['IS','OOS'])
        plt.title(f'Interval Performance: By: {groupby.title()} Metric: {metric.title()}')
        plt.show()
        
        best_results = main_df.loc[main_df['train_net'] > 0].sort_values(by = 'train_net', ascending = False)
        
        return main_df
    
    
    
    def train_test_comparison(self):
        """
        Compares In-Sample and Out-Of-Sample datasets and generates a dataframe 
        
        Returns
        -------
        comparison: pd.DataFrame
            Comparison dataframe containing difference on performance metrics
        
        train_test_df: pd.DataFrame
            Generates a side by side raw comparison on IS and OOS performance results
        """
        
        comparison = pd.concat([self.train_summary, self.train_filtered_summary, self.test_filtered_summary], axis = 1)
        self.train_test_df = comparison.copy()
        comparison = comparison.loc[['win_rate','avg_profit_per_win','avg_l_per_loss','profit_factor','max_dd_pct','avg_rrr']]
        comparison['diff'] = 1 - (comparison[comparison.columns[1]] / comparison[comparison.columns[0]])
        
        return comparison, self.train_test_df
    
    def plot_performance_comparison(self):
        """
        Generates a figure on In-Sample and Out-Of-Sample Performance comparison, comparing balance and peak balance, as well as
        interval filtered data.
        """
        
        test_start = self.test_evaluation.start_date
        start_bal = self.train_evaluation.starting_balance
        plt.figure(figsize=self.default_figsize)
        plt.axvline(x = test_start, ls = '--')
        plt.axhline(y = start_bal, ls = 'dotted', color = 'g')
        plt.plot(self.combined.index, self.combined[['balance','peak_balance']])
        plt.plot(self.combined_filtered.index, self.combined_filtered[['balance','peak_balance']])
        plt.legend(labels = [f'Sample Split {test_start}','Starting Balance','Balance','Peak Balance','Filtered Balance', 'Filtered Peak Balance'])
        plt.title('IS and OOS Performance Comparison')
        plt.ylabel('Equity ($)')
        plt.show()
        
    def plot_transaction_cost_composition(self, dataset:str = 'train'):
        """
        Generates a plot comparing raw profits, spread adjusted profits, and cost adjusted profits.
        
        Parameters
        ----------
        dataset: str 
            used for selecting the dataset for plotting.
        """
        
        data = self.select_dataset(dataset)
        sig = data.loc[data['signal'] != 0]
        
        grouped = (sig.groupby(self.grouping(sig))[['raw_profit','spread_adjusted_profit', 'commission', 'net_profit']].sum() / self.starting_balance) * 100
        
        grouped.plot(kind = 'bar', figsize = self.default_figsize, alpha=0.6, edgecolor='black')
        plt.legend(labels = ['Raw Profit','Spread Adjusted Profit', 'Transaction Costs', 'Cost Adjusted Profit'])
        plt.title(f'{dataset.title()} Set Transaction Cost Composition')
        plt.ylabel('Cumulative Gain (%)')
        plt.show()
        
    def plot_interval_filter_comparison(self, dataset:str = 'test'):
        """
        Plots an equity curve comparing results of a raw dataset, compared with a dataset with interval filtering. 
        
        Compares the unfiltered test data, and filtered test data. 
        """
        
        unfiltered = self.test_data.copy()
        filtered = self.test_filtered.copy()
        
        cols = ['balance','peak_balance']
        plt.figure(figsize=self.default_figsize)
        plt.axhline(y = self.starting_balance, ls = 'dotted', color = 'g')
        plt.plot(unfiltered.index, unfiltered[cols])
        plt.plot(filtered.index, filtered[cols])
        plt.legend(labels = ['Starting Balance', 'Unfiltered Balance', 'Unfiltered Peak Balance', 'Filtered Balance', 'Filtered Peak Balance'])
        plt.title(f'{dataset.title()} Set Interval Filtering Comparison')
        plt.ylabel('Equity ($)')
        plt.show()
        
    def plot_intraday_performance(self, dataset:str = 'train', metric: str = 'net_profit', func: str = 'mean'):
        """
        Plots intraday performance by net profit. 
        """

        data = self.select_dataset(dataset)

        sig = data.loc[data['signal'] != 0]
        grouped = sig.groupby(self.grouping(sig))[metric]

        if func == 'mean':
            calc = grouped.mean()

        elif func == 'sum':
            calc = grouped.sum()
        elif func == 'std':
            calc = grouped.std()

        calc.plot(kind = 'bar', figsize = self.default_figsize, alpha=0.6, edgecolor='black')
        plt.title('Intraday Performance')
        plt.xlabel('Time')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.show()

    def best_interval_by_profit(self, metric: str = 'net_profit'):
        """
        Builds a dataframe of best time intervals. 
        
        Ranks the time intervals by getting an aggregated mean of net profit. 
        
        Time intervals that are the most profitable on average are ranked first. 
        
        Returns
        -------
        best_results: pd.DataFrame
            dataframe of best performing time intervals. 
              
            Returns empty dataframe if no time intervals are profitable
        """
        
        dataset = self.train_data.copy()
        sig = dataset.loc[dataset['signal'] != 0]
        grouped = sig.groupby(self.grouping(sig))[['net_profit']].mean()
        best_results_df = grouped.loc[grouped['net_profit'] > 0].sort_values(by = 'net_profit', ascending = False)
        
        return best_results_df
    
    def best_interval_by_cost_delta(self, num_elements: int):
        """
        Builds a dataframe of best performing time intervals. 

        
        Ranks the time intervals by getting an aggregated sum of net_profit vs net transaction costs. 
        
        Time intervals that outperforms net transaction costs are ranked first. 
        
        Returns
        -------
        best_results_df: pd.DataFrame
            dataframe of best performing time intervals.
            
            Returns empty dataframe if no time intervals are profitable. 
        """
        
        dataset = self.train_data.copy()

        filter_by = dataset.index.time if self.timeframe in self.intraday else dataset.index.dayofweek
        
        cost_df = dataset.groupby(self.grouping(dataset))[['raw_profit','spread_adjusted_profit', 'commission', 'net_profit']].sum()
        cost_df['cost_diff'] = cost_df['net_profit'] - cost_df['commission']
        best_results_df = cost_df[['cost_diff']].loc[(cost_df['cost_diff'] > 0) & (cost_df['net_profit'] > 0)].sort_values(by='cost_diff', ascending = False)
        self.cost_df = cost_df.copy()
        
        return best_results_df[:num_elements]
    
    def plot_returns_distribution(self, dataset:str = 'train'):
        """
        Plots a distribution of cost adjusted profit. 
        
        Parameters
        ----------
        dataset: str
            used for selecting dataset to use for plotting.
        ----------
        """
        
        data = self.select_dataset(dataset)
        data_to_plot = (data.loc[(data['signal'] != 0) & (data['valid'] == 1)][['net_profit']] / self.starting_balance) * 100

        frmt = '{:.3g}'
        jb, jb_p, skew, kurt=tuple([j.item() for j in jarque_bera(data_to_plot)])
        jb_string = f'JB: {jb:.2f}\np-value: {frmt.format(jb_p)}\nSkew: {skew:.2f}\nKurt: {kurt:.2f}'
        
        sns.displot(data_to_plot, kde=True, legend=False, height=5, aspect=1.5, alpha=0.4)

        plt.xlabel('Profit ($)')
        plt.title(f'{dataset.title()} Set Profit Distribution\n{jb_string}', fontsize = 12)

    def plot_equity_curve(self):
        """
        Plots the equity curve and drawdown percentage.
        """
        dataset = self.combined_filtered

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15,8), sharex=True, gridspec_kw={'height_ratios':[2,1]})

        plt.subplots_adjust(left=0.1, right=0.9, top = 0.95, bottom = 0.1, hspace=0.1)
        fig.suptitle('Equity Curve and Drawdown', color='white')

        bal = dataset['balance']
        bal.plot(ax=ax1, kind = 'area', color ='springgreen', alpha=0.2, stacked =False)
        ax1.set_ylabel('Equity ($)')

        dd = dataset['drawdown'] * -1
        dd.plot(ax=ax2, kind='area', color = 'red', alpha =0.3)
        ax2.set_ylabel('Drawdown (%)')

        plt.show()

    
    def plot_cumulative_gain_by_month(self):
        """
        Plots cumulative gain by month
        """
        dataset = self.combined_filtered.copy()
        monthly_returns = pd.DataFrame((dataset.groupby([dataset.index.year, dataset.index.month])['net_profit'].sum() / self.starting_balance) * 100)
        monthly_returns.index = monthly_returns.index.rename(['Year','Mon_int'])
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_returns['Month'] = monthly_returns.index.get_level_values('Mon_int').map({
            k+1:v for k,v in enumerate(months)
        })

        monthly_returns = monthly_returns.reset_index()
        monthly_returns['Date'] = monthly_returns['Month'] + " " + monthly_returns['Year'].astype('str')
        monthly_returns = monthly_returns.set_index('Date')
        monthly_returns = monthly_returns.drop(['Year','Mon_int','Month'], axis = 1)

        monthly_returns.plot(kind = 'bar', figsize=(14,6), color = 'springgreen', alpha = 0.4, edgecolor = 'black')

        plt.legend(labels = ['Returns (%)'])
        plt.ylabel('Returns (%)')
        plt.xlabel('Date')
        plt.grid(axis='x')
        plt.title('Cumulative Gain By Month')
        plt.show()

    def plot_cumulative_gain_by_year(self):
        """
        Plots annual gain
        """
        dataset = self.combined_filtered.copy()
        annual_returns = pd.DataFrame((dataset.groupby([dataset.index.year])['net_profit'].sum() / self.starting_balance) * 100)
        annual_returns.index = annual_returns.index.rename('Year')
        annual_returns.plot(kind = 'bar', figsize=(14,6), color = 'springgreen', alpha = 0.4, edgecolor = 'black')

        plt.legend(labels = ['Returns (%)'])
        plt.ylabel('Returns (%)')
        plt.xlabel('Date')
        plt.title('Cumulative Gain By Year')
        plt.grid(axis = 'x')
        plt.show()

        
    def select_dataset(self, dataset: str):
        """
        Helper function for selecting dataset. Used for plotting figures
        
        Returns
        -------
        Selected Dataset
        """
        
        if dataset == 'train':
            return self.train_data.copy()
        
        elif dataset == 'test':
            return self.test_data.copy()
        
        elif dataset == 'train_filtered':
            return self.train_filtered.copy()
        
        elif dataset == 'test_filtered':
            return self.test_filtered.copy()
        
        elif dataset == 'combined':
            return self.combined.copy()
        
        elif dataset == 'combined_filtered':
            return self.combined_filtered.copy()
        
        else:
            raise ValueError('Invalid Dataset type. Use: train, test, train_filtered, test_filtered, combined, combined_filtered')

    @staticmethod
    def filter_default(timeframe: str):
        """
        Helper function for identifying grouping timeframes

        Parameters
        ----------
        timeframe: str
            timeframe in test
        """
        minutes_tf = ['m1','m5','m15','m30']
        hours_tf = ['h1','h4']
        htf = ['d1','w1','mn1']

        if timeframe in minutes_tf:
            return 'time'
        elif timeframe in hours_tf:
            return 'time'
        elif timeframe in htf:
            return 'day_of_week'
        else:
            raise ValueError('Invalid Timeframe')

    def grouping(self, reference_dataset):
        if self.filter_by == 'time':
            return reference_dataset.index.time 
        elif self.filter_by == 'day_of_week':
            return reference_dataset.index.time


