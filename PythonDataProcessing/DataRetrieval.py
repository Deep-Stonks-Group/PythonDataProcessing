from Historic_Crypto import HistoricalData
from datetime import datetime
import yfinance as yf
import talib as ta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas.core.frame import DataFrame
from datetime import timedelta

pd.options.mode.chained_assignment = None


class LSTM_DATA_HANDLE():
    '''
    Handles all gathering. Retrieves the data from the corresponding APIs, normalizes the data and formats it so that it
    is ready to go for the LSTM.
    '''

    def __init__(self, ticker, **kwargs):
        '''
        Initialization of the data handler.

          Input:
              *ticker* (Required) - String ticker that you want to train on. \n
              **kwargs**
                  *input_features* (Optional) - Array of features to be used for input. \n
                  *label_features* (Optional) - Array of feature to be used for label. \n
                  *indicator_key* (Optional) - String indicating which column in data you want to use for indicator. \n
                  *seq_length* (Optional) - Integer length of sequence. \n
                  *interval* (Optional) - String interval of prices: 1d, 1h, 5m or 1m. \n
                  *period* (Optional) - String specify a length of data to be retrieved: 1d, 1y, 1m, 2y \n
                  *start_date* (Optional) - The earliest historic data you want in form YYYY-MM-DD: 2000-01-01 \n
                  *end_date* (Optional) - The latest historic data you want in form YYYY-MM-DD: 2000-01-01 \n
                  *training_set_coeff* (Optional) - Decimal percent of dataset used for training. \n
                  *is_crypto* (Optional) - Boolean which indicates if ticker is a crypto. \n
        '''

        self.data_scaler = MinMaxScaler()
        self.label_scaler = MinMaxScaler()

        # Data/Feature Shape
        self.ticker = ticker
        self.input_features = kwargs.get('input_features', ['High', 'Low', 'Close', 'Volume', 'EMA'])
        self.label_features = kwargs.get('label_features', ['SMA'])
        self.indicator_key = kwargs.get('indicator_key', 'Close')
        self.seq_length = kwargs.get('seq_length', 10)
        self.interval = kwargs.get('interval', '1d')
        self.period = kwargs.get('period', '2y')
        self.training_set_coeff = kwargs.get('training_set_coeff', 0.8)
        self.is_crypto = kwargs.get('is_crypto', False)
        if self.is_crypto:
            self.start_date = kwargs.get('start_date', '2021-04-01-00-00')
            self.end_date = kwargs.get('end_date', datetime.today().strftime('%Y-%m-%d-%H-%M'))
        else:
            self.start_date = kwargs.get('start_date', None)
            self.end_date = kwargs.get('end_date', None)

    def retrieve_data(self):
        '''
          This function uses DataRtrieval to load the data, normalize it and format it to be used by the LSTM.

          Output:
              *x* - The final input data m x seq_length x n. Where m is the length of the entire dataset and n is
              the number of features. \n
              *y* - The corresponding m x 1 final labels for the dataset where m is the length of the entire dataset. \n
        '''
        if not self.is_crypto:
            data = get_stock_data(self.ticker, interval=self.interval,
                                  period=self.period, start_date=self.start_date, end_date=self.end_date)
        else:
            data = get_crypto_data(self.ticker, interval=self.interval, start_date=self.start_date,
                                   end_date=self.end_date)
        data = add_features(self.input_features, self.label_features, data, key=self.indicator_key)
        self.train_size = int(len(data) * self.training_set_coeff)
        self.set_train_size = self.train_size
        self.fit_scalers(data)
        scaled_data, scaled_labels = self.normalize_stock_data(data)
        x, y = sliding_windows(scaled_data, scaled_labels, self.seq_length)
        return x, y

    def fit_scalers(self, data_source):
        '''
          Creates the data and label scalers that are used to normalize data. This is created on the training dataset.

          Input:
              *data_source* (Required) - The training dataset used to create the normalization models.
        '''
        self.data_scaler.fit(data_source[self.input_features].iloc[:self.train_size])
        self.label_scaler.fit(data_source[self.label_features].iloc[:self.train_size])

    def normalize_stock_data(self, data_source):
        '''
            This uses pre-fit datascalers to normalize the entire dataset.

            Input:
                *data_source* (Required) - The training dataset used to create the normalization models.

            Output:
                *scaled_data* -
                *scaled_lbls* -

        '''
        scaled_data = self.data_scaler.transform(data_source[self.input_features])
        scaled_lbls = self.label_scaler.transform(data_source[self.label_features])
        return scaled_data, scaled_lbls


'''
All data goes from earliest to latest and is stored in a data frame.
You can access individual values by doing: data['High'], data['Low']...
'''

def get_all_currencies():
    # curr_lst = ['1INCH','ADA','ALGO','AMP','ANKR','ATOM','BAL','BAND','BAT','BCH','BNT','BOND','BTC',
    #             'CGLD','CHZ','COMP','CRV','CTSI','DAI','DASH','DOGE','DOT','ENJ','EOS','ETC','ETH','FORTH',
    #             'FIL','GRT','GTC','ICP','KEEP','KNC','LINK','LPT','LRC','LTC','MANA','MATIC','MIR','MKR',
    #             'MLN','NMR','NKN','NU','OGN','OMG','OXT','QNT','REN','REP','RLC','SUSHI','SKL','SNX','SOL','STORJ',
    #             'TRB','UMA','UNI','WBTC','XLM','XTZ','YFI','ZEC','ZRX']

    #Phemex Currency Left
    curr_lst = ['ADA', 'ALGO', 'ATOM','BAT', 'BOND', 'BTC',
                'CHZ', 'COMP', 'DOGE', 'DOT', 'ENJ', 'EOS', 'ETH',
                'FIL', 'GRT', 'GTC', 'LINK', 'LTC', 'MANA', 'MKR',
                'NU', 'SUSHI', 'SNX',
                'SOL',
                'UMA', 'UNI', 'XLM', 'XTZ', 'YFI', 'ZEC']
    return curr_lst

def get_all_stocks():
    list_of_tickers = gt.get_tickers()
    return list_of_tickers

def get_last_crypto_price(symbol,interval):
    today = datetime.today()
    start_date = today - timedelta(days=3)
    start_date = start_date.strftime('%Y-%m-%d-00-00')
    prices = get_crypto_data(symbol, interval=interval, start_date=start_date)['Close']
    close_price = prices.tail(1)
    close_price = close_price.values[0]
    return close_price

def get_last_stock_price(symbol,interval):
    prices = get_stock_data(symbol, interval=interval, period='5d')['Close']
    close_price = prices.tail(2)
    close_price = close_price.values[0]
    return close_price

def get_crypto_data(symbol, **kwargs):
    '''
    This functions takes a symbol for a crypto and returns it's historic data.

    Input:
        *symbol* (Required) - Takes the exchanged pairs: ETH-USD or BTC_ETH, or BTC-USD \n
        *interval* (optional) - Takes the interval of prices: 1d, 1h, 5m or 1m. \n
        *start_date* (optional) - The earliest historic data you want in form YYYY-MM-DD-HH-SS: 2000-01-01-00-00 \n
        *end_date* (optional) - The latest historic data you want in form YYYY-MM-DD-HH-SS: 2000-01-01-00-00 \n

    Output:
        *data* - A data frame with the High, Low, Open, Close and Volume prices. This is the historic prices. By default it returns
        all daily prices unless otherwise specified.

    Examples:
        get_crypto_data('BTC-USD',interval='1h',start_date='2021-04-01-00-00') \n
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
    data = HistoricalData(symbol, seconds, start_date, end_date,verbose=False).retrieve_data()
    data = data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", 'volume': 'Volume'})
    return data


def get_stock_data(symbol, **kwargs):
    '''
    This functions takes a symbol for a stock and returns it's historic data.

    Input:
        *symbol* (Required) - Takes the exchanged pairs: ETH-USD or BTC_ETH, or BTC-USD \n
        *interval* (optional) - Takes the interval of prices: 1d, 1h, 5m or 1m.\n
        *start_date* (optional) - The earliest historic data you want in form YYYY-MM-DD: 2000-01-01 \n
        *end_date* (optional) - The latest historic data you want in form YYYY-MM-DD: 2000-01-01 \n
        *period* (optional) - Instead of specifying an end_date you can specify a period: 1d,5d,1mo,3mo,6mo,1y,2y,
        5y,10y,ytd,max

    Output:
        *data* - A data frame with the High, Low, Open, Close and Volume prices. This is the historic prices. By default it returns
        all daily prices unless otherwise specified.

    ** Note: yFinance is limited in what it can return in terms of intraday data so be aware.

    Examples:
        data_source = DR.get_stock_data('DIA',interval='1d',period='2y') \n
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


def add_features(input_features, label_features, data, key='Close'):
    '''
       This function is responsible for adding adding the neccesarry indicators/features to the data.

       Input:
            *input_features* (Required) -  List of features to be used for input. \n
            *label_features* (Required) - List of features to be used for labels. \n
            *data* (Required) - Dataframe of historical stock data. \n
            *key* (optional) - Key used for technical indicators.

       Output:
           *data* - A data frame with the High, Low, Open, Close, Volume prices in addition to the added features.

       Examples:
           data = add_features(['High','Low','Close','Volume','EMA'],['SMA'],data) \n
           data = add_features(['High','Low','Close','Volume','EMA'],['SMA'],data,key='High')
       '''
    new_features = []
    new_features.extend(set(input_features) - set(data.columns))
    new_features.extend(set(label_features) - set(data.columns))
    for new_feature in new_features:
        try:
            data = add_technical_indicators[new_feature](data, key)
        except KeyError:
            print(f'No function exists to add the dimension {new_feature}')
            raise
    return data

def add_SMA(data, key='Close'):
    '''
    This functions takes a data frame of historic data and adds the SMA to it.

    Input:
        *data* (required) - A dataframe of historic data that SMA will be added to. \n
        *key* (optional) - The column which you want to apply the SMA.

    Output:
        *data* - A data frame with the SMA added.

    Examples:
        add_SMA(data_source)
    '''
    # Adding SMA
    sma = ta.SMA(data[key])
    data['SMA'] = sma
    data = data.iloc[29:, :]
    return data

def add_EMA(data, key='Close'):
    '''
    This functions takes a data frame of historic data and adds the EMA to it.

    Input:
        *data* (required) - A dataframe of historic data that EMA will be added to. \n
        *key* (optional) - The column which you want to apply the SMA.

    Output:
        *data* - A data frame with the EMA added.

    Examples:
        add_EMA(data_source)
    '''
    # Adding EMA
    ema = ta.EMA(data[key])
    data['EMA'] = ema
    data = data.iloc[29:, :]
    return data


def sliding_windows(data, labels, seq_length):
    '''
    This functions takes a data array and converts it to an x,y array that is seq_length long.
    This creates a single seq_length input for an LSTM.

    Input:
        *data* (required) - An m x n numpy array of data where m is the length of the entire dataset. And n is the number
        of features. \n
        *labels* (required) - The corresponding m x 1 labels for the dataset where m is the length of the entire dataset. \n
        *seq_length* (required) - Represents minutes/hours/days that are included in a single LSTM input.

    Output:
        *x* - The formatted data m x seq_length x n. Where m is the length of the entire dataset and n is the number of
        features. \n
        *y* - The corresponding m x 1 labels for the dataset where m is the length of the entire dataset. \n

    Examples:
        x, y = sliding_windows(data, labels, 10)
    '''
    x = []
    y = []
    for i in range(data.shape[0] - seq_length - 1):
        _x = data[i: (i + seq_length)]
        _y = labels[i + seq_length]
        x.append(_x)
        y.append(_y)
    return x, y


add_technical_indicators = {
    'EMA': add_EMA,
    'SMA': add_SMA
}