from Historic_Crypto import HistoricalData
from datetime import datetime
import yfinance as yf
import talib as ta

'''
All data goes from earliest to latest and is stored in a data frame.
You can access individual values by doing: data['High'], data['Low']...
'''


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
def get_crypto_data(symbol,**kwargs):
    interval = kwargs.get('interval','1d')
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
    data = HistoricalData(symbol,seconds,start_date,end_date).retrieve_data()
    data = data.rename(columns={"open": "Open", "high": "High","low": "Low", "close": "Close",'volume':'Volume'})
    return data


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
def get_stock_data(symbol,**kwargs):
    interval = kwargs.get('interval','1d')
    start_date = kwargs.get('start_date', '2000-01-01')
    end_date = kwargs.get('end_date', datetime.today().strftime('%Y-%m-%d'))
    period = kwargs.get('period', None)

    stock = yf.Ticker(symbol)
    if period != None:
        return stock.history(period=period,interval=interval)

    return stock.history(interval=interval,start=start_date,end=end_date)

'''
This functions takes a data frame of historic data and adds the SMA to it.

Input:
    data (required) - A dataframe of historic data that SMA will be added to.

Output:
    A data frame with the SMA added.
Examples:
    add_SMA(data_source)
'''
def add_SMA(data):
    # Adding SMA
    sma = ta.SMA(data['Close'])
    data['SMA'] = sma
    data= data.iloc[29:, :]
    return data

'''
This functions takes a data frame of historic data and adds the EMA to it.

Input:
    data (required) - A dataframe of historic data that EMA will be added to.

Output:
    A data frame with the EMA added.
Examples:
    add_EMA(data_source)
'''
def add_EMA(data):
    # Adding EMA
    ema = ta.EMA(data['Close'])
    data['EMA'] = ema
    data= data.iloc[29:, :]
    return data

