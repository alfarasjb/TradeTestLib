import MetaTrader5 as mt5
import pandas as pd 
import os
from datetime import datetime as dt, timedelta


class MT5_Data:
    
    def __init__(self):
        if mt5.account_info() is None:
            print('Launching MT5')
            self.launch_mt5()
            
    def launch_mt5(self):
        if not mt5.initialize('C:/Program Files/MetaTrader 5 IC Markets (SC)/terminal64.exe'):
            return False
        else:
            print('MT5 Launched')
            return True

    def request_price_data(self, symbol: str, 
                           tf: str, 
                           request_type:str = 'pos', 
                           start_date = None, end_date = None, 
                           start_index: int = 0, 
                           num_bars:int = 99000, 
                           export: bool = False):
        """
        Requests price data from MT5, and exports to csv
        
        Parameters
        ----------
        symbol: str
            Symbol to fetch 
            
        tf: str
            Timeframe

        request_type: str
            Default: 'pos'

            option for requesting data from mt5 (pos, date, range)

            pos - fetch based on position
            date - fetch based on start date
            range - fetch based on date range

        start_date: dt
			Default: None

			start date for requesting data from mt5. used when selecting
			'date' or 'rates' as request_type.

		end_date: dt
			Default: None

			end date for requesting data from mt5. used when 'rates' is 
			selected as request_type

		start_index: int
			Default: 0

			start index for requesting data from mt5. used when 'pos' is 
			selected as request_type
		
		num_bars: int
			Default: 99000

			number of bars to receive. used when selecting 'pos' or 'date'
			as request_type.

        Returns
        -------
        data: pd.DataFrame
            Returns a dataframe of requested data
        """
        timeframe_converter = {
            'm1' : mt5.TIMEFRAME_M1,
            'm5' : mt5.TIMEFRAME_M5,
            'm15' : mt5.TIMEFRAME_M15,
            'm30' : mt5.TIMEFRAME_M30,
            'h1' : mt5.TIMEFRAME_H1,
            'h4' : mt5.TIMEFRAME_H4,
            'd1' : mt5.TIMEFRAME_D1,
            'w1' : mt5.TIMEFRAME_W1,
            'mn1' : mt5.TIMEFRAME_MN1,
        }
        
        try:
            if request_type == 'pos':
                rates = mt5.copy_rates_from_pos(symbol, timeframe_converter[tf], start_index, num_bars)

            elif request_type == 'date':
                rates = mt5.copy_rates_from(symbol, timeframe_converter[tf], start_date, num_bars)

            elif request_type == 'range':
                rates = mt5.copy_rates_range(symbol, timeframe_converter[tf], start_date, end_date)
            
            else: 
                raise ValueError('Invalid Request')
            

            #rates = mt5.copy_rates_from_pos(symbol, timeframe_converter[tf], 1, 99000)
            data = pd.DataFrame(data = rates)
            data['time'] = pd.to_datetime(data['time'], unit = 's')
            data = data.loc[:, ['time', 'open', 'high', 'low', 'close','spread']]
            data.columns = ['Date','Open','High','Low','Close','Spread']
            data = data.set_index('Date', drop = True)

            if export:
                dir_path = f'.//history//{symbol}'
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                path = f'{dir_path}//{symbol}_{tf}.csv'
                data.to_csv(path)

            return data
        
        except KeyError: 
            print(f'No data available for: {symbol} {tf}')

        
        
        
        
    @staticmethod
    def get_mt5_symbols(category: str = 'Exotics'):
        """
        Fetches a list of symbols under the specified category.
        
        Parameters
        ----------
        category: str
            Category of symbols to select
            
        Returns
        -------
        symbols_list: list
            Returns a list of symbols under the specified category.
        """
        mt5_symbols = mt5.symbols_get()
        symbols_list = []
        
        for sym in mt5_symbols:
            sym_dict = sym._asdict()
            path = sym_dict['path']
            name = sym_dict['name']
            if category not in path:
                continue
                
            symbols_list.append(name)
            
        return symbols_list