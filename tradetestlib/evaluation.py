import pandas as pd
import numpy as np

class Evaluation:
    """
    Creates an Evaluation object of the specified dataset. Used for generating a summary of alpha performance,
    for optimization and hyperparameter tuning.
    
    Parameters
    ----------
    data: pd.DataFrame
        Input data to evaluate 
        
    hyperparameters: dict
        Tunable hyperparameters
        
    Methods
    -------
    evaluate:
        generates evaluation and performance metrics of the dataset 
        
    evaluation_dataframe
        generates a dataframe summarizing performance metrics
        
        
    Data
    ----
    
    start_date: 
        start date of the test
        
    end_date
        end date of the test
    
    days:
        calendar days duration of testing period
        
    starting_balance
        deposit capital
    
    end_balance
        resulting balance after testing period
        
    spread
        bid-ask spread
        
    mean_profit_points
        mean profitable points difference from opening price to trade closing price
        
    mean_loss_points
        mean losing points difference from opening price to trade closing price
        
    pts_to_spread
        ratio of profitable points to spread 
    
        determines alpha performance compared to spread
    
        ideal value: > 1 
        
    lot:
        trade lot size
        
        tunable hyperparameter
    
    holdtime:
        trade hold time in intervals
        
        tunable hyperparameter
        
    prop_max_loss_pct:
        maximum allowable loss as a percentage of initial deposit
        
        determines risk appetite that yields the best performance
        
        tunable hyperparameter
        
        
    max_bal:
        maximum balance recorded during trading period
        
    min_bal:
        minimum balance recorded during trading period
        
    max_bal_pct:
        maximum balance recorded as percentage of initial deposit
        
    min_bal_pct
        minimum balance recorded as percentage of initial deposit
        
    wins:
        num of winning trades
        
    losses:
        num of losing trades
        
    total: 
        total trades opened
        
    win_rate:
        wins / total
        
    avg_win_usd:
        average profit from winning trades in USD
        
    avg_loss_usd:
        average exposure from losing trades in USD
        
    max_profit_usd:
        maximum profit recorded in USD
        
    max_loss_usd: 
        maximum loss incurred in USD
        
    max_profit_pct:
        maximum profit as a percentage of initial deposit
    
    max_loss_pct:
        maximum loss as a percentage of initial deposit
        
    gross_profit
        gross profit during testing period
        
    net_profit:
        net profit during testing period
        
    net_profit_pct:
        net_profit as a percetage of initial deposit 
        
    daily_return:
        average daily return during testing period
        
    monthly_return:
        average monthly return during testing period
        
    long_positions:
        long positions opened
        
    short_positions:
        short positions opened
        
    pct_long:
        percentage of long positions during testing period
        
    pct_short:
        percentage of short positions during testing period
        
    long_wins:
        number of profitable long positions
        
    short_wins:
        number of profitable short positions
        
    long_avg_win:
        average gain in USD for long positins
    
    short_avg_win:
        average gain in USD for short positions
        
    sharpe_ratio:
        sharpe ratio during the testing period
        
    profit_factor:
        profit factor during the testing period
        
    max_dd_pct:
        maximum recorded drawdown as a percentage of starting balance
        
    avg_rrr:
        average risk to reward ratio 
        
    commission_composition:
        average percentage of commission to average profitable trades
        
        
    """
    def __init__(self, data: pd.DataFrame, hyperparameters: dict):
    
        self.data = data.copy()
        self.hyperparameters = hyperparameters
        self.evaluate()
    
    
    def evaluate(self):
        """
        Generates evaluation and calculates performance metrics of the dataset. 
        """
        # Data to evaluate
        data = self.data
        
        ## MASK
        profit_mask = data['net_profit'] > 0
        loss_mask = data['net_profit'] < 0
        long_signal_mask = data['signal'] == 1
        short_signal_mask = data['signal'] == -1
        
        # Test start date, end date, and calendar days
        self.start_date = data[:1].index.item().date()
        self.end_date = data[-1:].index.item().date()
        days = (self.end_date - self.start_date).days
        years = ((data.index[-1:] - data.index[:1]) / np.timedelta64(1, 'Y')).item()
        
        # Test start and final balance
        self.starting_balance = data[:1]['balance'].item()
        self.end_balance = data[-1:]['balance'].item()
        
        # Recorded spread (mean spread for opened trades)
        self.spread = data.loc[profit_mask]['spread'].mean()
        
        # Mean points
        self.mean_profit_points = data.loc[profit_mask]['spread_adj_trade_points'].mean()
        self.mean_loss_points = data.loc[loss_mask]['spread_adj_trade_points'].mean()
        
        # Points to spread ratio
        self.pts_to_spread = self.mean_profit_points / self.spread if self.spread > 0 else np.inf
        
        # Hyperparameters: lotsize, hold time, max loss pct
        self.lot = self.hyperparameters['lot']
        self.holdtime = self.hyperparameters['holdtime']
        self.prop_max_loss_pct = self.hyperparameters['max_loss_pct']
        
        # Max and Min balance recorded (USD and pct)
        self.max_bal, self.min_bal = data['balance'].max(), data['balance'].min()
        self.max_bal_pct = ((self.max_bal / self.starting_balance) - 1) * 100
        self.min_bal_pct = (1 - (self.min_bal / self.starting_balance)) * 100
        
        # Num. of winning/losing trades, win rate
        self.wins = data.loc[profit_mask]['net_profit'].count()
        self.losses = data.loc[loss_mask]['net_profit'].count()
        self.total = data.loc[(data['signal'] != 0) & (data['valid'] != 0)]['net_profit'].count()
        self.win_rate = (self.wins / self.total) * 100
        
        # Trade result statistics: Average p/l, max p/l, flat amount and pct
        self.avg_win_usd = data.loc[profit_mask]['net_profit'].mean()
        self.median_win_usd = data.loc[profit_mask]['net_profit'].median()
        self.avg_loss_usd = data.loc[loss_mask]['net_profit'].mean()
        self.median_loss_usd = data.loc[loss_mask]['net_profit'].median()
        self.max_profit_usd = data['net_profit'].max()
        self.max_loss_usd = data['net_profit'].min()        
        self.max_profit_pct = (self.max_profit_usd / self.starting_balance) * 100
        self.max_loss_pct = (self.max_loss_usd / self.starting_balance) * 100
        self.returns_vol = data.loc[data['net_profit'] != 0]['net_profit'].std()
        
        # Gross and net profit
        self.gross_profit = data.loc[data['net_profit'] > 0]['net_profit'].sum()
        self.net_profit = data['net_profit'].sum()
        self.net_profit_pct = (self.net_profit / self.starting_balance) * 100
        
        # Periodic Return (for calculating sharpe ratio)
        self.daily_return = self.net_profit_pct / days
        self.monthly_return = self.daily_return * 30
        self.annual_return = self.monthly_return * 12
        
        # Order statistics (long and short), amount, and performance
        self.long_positions = data.loc[(long_signal_mask) & (data['net_profit'] != 0)]['net_profit'].count()
        self.short_positions = data.loc[(short_signal_mask) & (data['net_profit'] != 0)]['net_profit'].count()
        self.pct_long = (self.long_positions / self.total) * 100
        self.pct_short = (self.short_positions / self.total) * 100
        self.long_wins = data.loc[(long_signal_mask) & (profit_mask)]['net_profit'].count()
        self.short_wins = data.loc[(short_signal_mask) &(profit_mask)]['net_profit'].count()
        self.long_wr = (self.long_wins / self.long_positions) * 100
        self.short_wr = (self.short_wins / self.short_positions) * 100
        self.long_avg_win = data.loc[(long_signal_mask) & (profit_mask)]['net_profit'].mean()
        self.short_avg_win = data.loc[(short_signal_mask) & (profit_mask)]['net_profit'].mean()
        
        # sharpe ratio - approximated from current 10y yield
        tbill_rate = 4.47 # 10 year as of 11/25/2023
        sdev_ret = (data.loc[data['net_profit'] != 0, 'net_profit'] / self.starting_balance).std()
        
        risk_free_rate = tbill_rate / 100
        
        roi = self.annual_return / 100
        #self.sharpe_ratio = (roi - risk_free_rate) / sdev_ret

        # expectancy
        # (average gain * win%) - (average loss * loss%)
        self.expectancy = ((self.avg_win_usd * (self.win_rate/100))) - (abs(self.avg_loss_usd) * (1 - (self.win_rate / 100)))
        
        # cagr - compound annual growth rate
        self.cagr = (((self.end_balance / self.starting_balance) ** (1 / years)) - 1) * 100

        # Overall performance: profit factor, maxdd, avg rrr 
        
        self.profit_factor = abs((self.avg_win_usd * self.win_rate) / ((1 - self.win_rate) * self.avg_loss_usd))
        self.max_dd_pct = data['drawdown'].max()
        self.avg_rrr = abs(self.avg_win_usd / self.avg_loss_usd)
        
        # Commission composition: Percentage lost due to transaction costs
        self.commission_composition = ((data['commission'].max()*2) / self.avg_win_usd) * 100
        
        
        self.evaluation_data = {
            'start_date' : self.start_date, 
            'end_date' : self.end_date, 
            'starting_balance' : self.starting_balance, 
            'end_balance' : self.end_balance, 
            'mean_profit_points' : self.mean_profit_points, 
            'mean_loss_points' : self.mean_loss_points, 
            'pts_to_spread' : self.pts_to_spread,
            'lot' : self.lot,
            'holdtime' : self.holdtime,
            'max_loss_pct' : self.prop_max_loss_pct, 
            'max_bal' : self.max_bal,
            'min_bal' : self.min_bal,
            'max_bal_pct' : self.max_bal_pct, 
            'min_bal_pct' : self.min_bal_pct,
            'spread' : self.spread,
            'wins' : self.wins, 
            'losses' : self.losses, 
            'total' : self.total, 
            'win_rate' : self.win_rate,
            'avg_profit_per_win' : self.avg_win_usd, 
            'avg_l_per_loss' : self.avg_loss_usd, 
            'gross_profit' : self.gross_profit,
            'net_profit' : self.net_profit,
            'net_profit_pct' : self.net_profit_pct, 
            'returns_vol' : self.returns_vol,
            'daily_return' : self.daily_return,
            'monthly_return' : self.monthly_return,
            'annual_return' : self.annual_return,
            'max_profit_usd' : self.max_profit_usd,
            'max_profit_pct' : self.max_profit_pct,
            'max_loss_usd' : self.max_loss_usd,
            'max_loss_pct' : self.max_loss_pct,
            'median_profit_usd' : self.median_win_usd,
            'median_loss_usd' : self.median_loss_usd,
            'num_longs' : self.long_positions,
            'pct_longs' : self.pct_long,
            'long_wins' : self.long_wins,
            'long_wr' : self.long_wr,
            'long_avg_win' : self.long_avg_win,
            'num_shorts' : self.short_positions,
            'pct_short' : self.pct_short,
            'short_wins' : self.short_wins,
            'short_wr' : self.short_wr,
            'short_avg_win' : self.short_avg_win,
            'profit_factor' : self.profit_factor,  
            #'sharpe_ratio' : self.sharpe_ratio,
            'expectancy' : self.expectancy,
            'cagr' : self.cagr,
            'max_dd_pct' : self.max_dd_pct,
            'avg_rrr' : self.avg_rrr,
            'commission_composition' : self.commission_composition
        }
      
    def evaluation_dataframe(self):
        """
        Returns
        -------
        Evaluation dataframe summarizing alpha performance
        """
        
        return pd.DataFrame.from_dict(self.evaluation_data, orient = 'index')