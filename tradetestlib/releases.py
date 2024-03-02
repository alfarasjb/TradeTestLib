from datetime import datetime as dt, timedelta 
from urllib.request import urlretrieve 
from tqdm import tqdm 
import os
import pandas as pd

class Releases: 


    def __init__(self, directory:str = "", use_default_path: bool = False):
        self.path = directory if not use_default_path else os.environ.get('RELEASES_PATH')

        self.src_filename = "news_dates.csv"
        self.src_path = os.path.join(self.path, self.src_filename)

        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    
    def download(self, url, filename):
        """ 
        Download from url. 

        Parameters
        ----------
        url: str
            source url 
        
        filename: str 
            downloaded filename
        """

        filename = f"{filename}.csv" if not filename.endswith("csv") else filename 
        file_path = os.path.join(self.path, filename) 

        try: 
            location, msg = urlretrieve(url = url, filename = file_path)
            print(f"Downloaded File to: {location}")
        except: 
            print("Download Failed.")

        return location

    def fetch_latest(self):
        """ 
        Download latest weekly data 
        """
        url = "https://www.robots4forex.com/news/week.php" 
        latest = (dt.now() - timedelta(days = dt.now().weekday())).date().strftime("%Y-%m-%d")
        filename = f"{latest}_news_raw.csv"
        path = os.path.join(self.path, filename)


        directory = os.path.dirname(path) if os.path.dirname(path) != "" else None 
        files = os.listdir(directory) 
        
        location = self.download(url, filename) if filename not in files else filename 
        df = self.parse_raw_df(location)

        return df

    def parse_raw_df(self, file):
        """ 
        Parses raw df into a readable format. 
        """
        df = pd.read_csv(file, header = None)
        df.columns = ['date','time','country','impact','description', '_','_','actual','forecast','previous']
        df['date'] = df['date'] + ' ' + df['time']
        df = df.set_index('date', drop = True)
        df.index = pd.to_datetime(df.index)
        df = df.drop(['time'], axis = 1)
        df.index = df.index + pd.Timedelta(value = 2, unit = 'hours')
        df = df.drop(['_','_'], axis = 1)
        df = df.fillna(0)

        return df 

    def update_src(self):
        """ 
        Updates main file and directories 

        """
        url = "https://www.robots4forex.com/news/news.php"
        filename = "raw.csv"
        location = self.download(url, filename)

        df = self.parse_raw_df(location)

        news_dates_path = os.path.join(self.path, self.src_filename)
        df.to_csv(self.src_path)

        print(f"Updated Source: {news_dates_path}")

        updated_symbols = self.update_src_directories()
        print(f"{updated_symbols} Symbols Updated.")

        return df 

    def get_news(self, symbol: str = None, start_date = None, end_date = None, country = None, impact = None): 
        
        """ 
        Gets events dataframe. 

        Can be filtered by symbol, start date, end date, country, and impact. 
        """
        
        df = pd.read_csv(self.src_path, index_col='date')
        df.index = pd.to_datetime(df.index)

        countries = df['country'].unique().tolist()
        impacts = df['impact'].unique().tolist()
        
        try: 
            iter_country = iter(country)
            if type(country) is not list:
                raise TypeError
            for c in country:
                if c not in countries:
                    raise ValueError(f"Invalid Country: {c}")
            
            df = df.loc[df['country'].isin(country)]
        except TypeError:
            if country is not None and country not in countries:
                raise ValueError(f"Invalid Country: {country}")

            df = df.loc[df['country'] == country] if country is not None else df


        try: 
            t = iter(impact)


            for i in impact: 
                if i not in impacts: 
                    raise ValueError(f"Invalid Impact: {i}")
            df = df.loc[df['impact'].isin(impact)]

        except TypeError: 
            if impact is not None and impact not in impacts:
                raise ValueError(f"Invalid Impact: {impact}")
            df = df.loc[df['impact'] == impact] if impact is not None else df 


        if symbol is not None:        
            countries = [symbol[:3], symbol[3:]]
            df = df.loc[df['country'].isin(countries)] 
        
        df = df.loc[df.index.date >= start_date] if start_date is not None else df 
        df = df.loc[df.index.date <= end_date] if end_date is not None else df 

        return df


    def update_src_directories(self, symbols = list()):
        """ 
        Updates source directories sorted by symbol. 
        """
        df = self.get_news()
        years = df.index.year.unique().tolist()

        directory = os.path.join(self.path, "events")

        if len(symbols) == 0: 
            if not os.path.isdir(directory):
                os.mkdir(directory)
            symbols = os.listdir(directory)

        if len(symbols) == 0:
            print("No symbols found.")
            return 0
        

        for symbol in tqdm(symbols): 
            curr = [symbol[:3], symbol[3:]]
            for year in years: 
                d = df.loc[(df['country'].isin(curr)) & (df.index.year == year)]
                path = os.path.join(directory, symbol)
                if not os.path.isdir(path):
                    os.mkdir(path)

                filename = f"{symbol}_{year}.csv"
                filepath = os.path.join(path, filename)
                d.to_csv(filepath)

        return len(symbols)