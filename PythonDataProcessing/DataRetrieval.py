from Historic_Crypto import HistoricalData
from datetime import datetime
import yfinance as yf
import talib as ta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas.core.frame import DataFrame

pd.options.mode.chained_assignment = None

'''
All data goes from earliest to latest and is stored in a data frame.
You can access individual values by doing: data['High'], data['Low']...
'''

class Scaler():
    def __init__(self,input_features,label_features):
        self.data_scaler = MinMaxScaler()
        self.label_scaler = MinMaxScaler()
        self.input_features = input_features
        self.label_features = label_features

    def set_train_size(self,train_size):
        self.train_size = train_size

    def fit_scalers(self,data_source):
        '''
        Creates the data and label scalers that are used to normalize data. This is created on the training dataset.
        :param data_source: The training dataset used to create the normalization models.
        '''
        self.data_scaler.fit(data_source[self.input_features].iloc[:self.train_size])
        self.label_scaler.fit(data_source[self.label_features].iloc[:self.train_size])

    def normalize_stock_data(self, data_source):
        '''
        This uses pre-fit datascalers to normalize the entire dataset.
        :param data_source: The entire dataset to be normalized.
        '''
        scaled_data = self.data_scaler.transform(data_source[self.input_features])
        scaled_lbls = self.label_scaler.transform(data_source[self.label_features])
        return scaled_data, scaled_lbls




def get_crypto_data(symbol, **kwargs):
    '''
    This functions takes a symbol for a crypto and returns it's historic data.

    Input:
        symbol (Required) - Takes the exchanged pairs: ETH-USD or BTC_ETH, or BTC-USD
        interval (optional) - Takes the interval of prices: 1d, 1h, 5m or 1m.
        start_date (optional) - The earliest historic data you want in form YYYY-MM-DD-HH-SS: 2000-01-01-00-00
        end_date (optional) - The latest historic data you want in form YYYY-MM-DD-HH-SS: 2000-01-01-00-00

    Output:
        A data frame with the High, Low, Open, Close and Volume prices. This is the historic prices. By default it returns
        all daily prices unless otherwise specified.

    Examples:
        get_crypto_data('BTC-USD',interval='1h',start_date='2021-04-01-00-00')
        get_crypto_data('BTC-USD',interval='1h',start_date='2021-04-01-00-00',end_date='2021-06-01-00-00')
        get_crypto_data('ETH-USD')
    '''
    interval = kwargs.get('interval', '1d')
    start_date = kwargs.get('start_date', '2000-01-01-00-00')
    end_date = kwargs.get('end_date', datetime.today().strftime('%Y-%m-%d-%H-%M'))
    seconds = 0
    if interval == '1d':
        seconds = 86400
    elif interval == '1h':
        seconds = 3600
    elif interval == '5m':
        seconds = 300
    elif interval == '1m':
        seconds = 60
    data = HistoricalData(symbol, seconds, start_date, end_date).retrieve_data()
    data = data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", 'volume': 'Volume'})
    return data


def get_stock_data(symbol, **kwargs):
    '''
    This functions takes a symbol for a stock and returns it's historic data.

    Input:
        symbol (Required) - Takes the exchanged pairs: ETH-USD or BTC_ETH, or BTC-USD
        interval (optional) - Takes the interval of prices: 1d, 1h, 5m or 1m.
        start_date (optional) - The earliest historic data you want in form YYYY-MM-DD: 2000-01-01
        end_date (optional) - The latest historic data you want in form YYYY-MM-DD: 2000-01-01
        period (optional) - Instead of specifying an end_date you can specify a period: 1d, 1y, 1m, 2y

    Output:
        A data frame with the High, Low, Open, Close and Volume prices. This is the historic prices. By default it returns
        all daily prices unless otherwise specified.

    ** Note: yFinance is limited in what it can return in terms of intraday data so be aware.

    Examples:
        data_source = DR.get_stock_data('DIA',interval='1d',period='2y')
        data_source = DR.get_stock_data('AAPL',start_date='2021-04-01',end_date='2021-06-01')
    '''
    interval = kwargs.get('interval', '1d')
    start_date = kwargs.get('start_date', '2000-01-01')
    end_date = kwargs.get('end_date', datetime.today().strftime('%Y-%m-%d'))
    period = kwargs.get('period', None)

    stock = yf.Ticker(symbol)
    if period != None:
        return stock.history(period=period, interval=interval)

    return stock.history(interval=interval, start=start_date, end=end_date)

def add_features(input_features,label_features,data):
    new_features = []
    new_features.extend(set(input_features) - set(data.columns))
    new_features.extend(set(label_features) - set(data.columns))

    for new_feature in new_features:
        try:
            data = add_technical_indicators[new_feature](data)
        except KeyError:
            print(f'No function exists to add the dimension {new_feature}')
            raise
    return data

def add_SMA(data):
    '''
    This functions takes a data frame of historic data and adds the SMA to it.

    Input:
        data (required) - A dataframe of historic data that SMA will be added to.

    Output:
        A data frame with the SMA added.
    Examples:
        add_SMA(data_source)
    '''
    # Adding SMA
    sma = ta.SMA(data['Close'])
    data['SMA'] = sma
    data = data.iloc[29:, :]
    return data


def add_EMA(data):
    '''
    This functions takes a data frame of historic data and adds the EMA to it.

    Input:
        data (required) - A dataframe of historic data that EMA will be added to.

    Output:
        A data frame with the EMA added.
    Examples:
        add_EMA(data_source)
    '''
    # Adding EMA
    ema = ta.EMA(data['Close'])
    data['EMA'] = ema
    data = data.iloc[29:, :]
    return data

def sliding_windows( data, labels, seq_length):
    x = []
    y = []
    for i in range(data.shape[0]-seq_length-1):
        _x = data[i: (i+seq_length)]
        _y = labels[i+seq_length]
        x.append(_x)
        y.append(_y)
    return x, y

add_technical_indicators = {
    'EMA': add_EMA,
    'SMA': add_SMA
}