# /* Copyright (C) Kannan Sekar Annu Radha - All Rights Reserved
#  * Unauthorized copying of this file, via any medium is strictly prohibited
#  * Proprietary and confidential
#  * Written by Kannan Sekar Annu Radha <kannansekara@gmail.com>, October 2021
#  */ 


import os
import datetime
from re import X
import matplotlib as mpl
from pandas.core.frame import DataFrame
import websocket, json, pprint, talib 
import config
import numpy as np
import pandas as pd
import seaborn as sns
from binance.client import Client
from binance.enums import *
from keras.layers import LSTM, Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import rcParams
from pandas_datareader import data as wb
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import logging
import threading
import time
import websockets
import asyncio
from datetime import datetime
from unicorn_binance_websocket_api.unicorn_binance_websocket_api_manager import BinanceWebSocketApiManager
import csv
from keras.models import load_model

client = Client(config.API_KEY, config.API_SECRET, tld='us')

eth_quantity = 0.07284003
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
TRADE_SYMBOL = 'ETHUSD'
TRADE_QUANTITY = 0.07

SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"


in_position = False

client = Client(config.API_KEY, config.API_SECRET, tld='us')

closes = []
closes_times = []
now = []
zip2 = []
df = pd.DataFrame()
df['price']  = closes
df['time']  = closes_times


def order(side, quantity, symbol,order_type=ORDER_TYPE_MARKET):
    try:
        print("sending order")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False

    return True


class RunThread(threading.Thread):
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        super().__init__()

    def run(self):
        self.result = asyncio.run(self.func(*self.args, **self.kwargs))

def run_async(func, *args, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        thread = RunThread(func, args, kwargs)
        thread.start()
        thread.join()
        # return thread.result
        return thread.result

    else:
        return asyncio.run(func(*args, **kwargs))


class BinanceWs:

    def __init__(self, channels, markets):
        market = 'ethusdt'
        tf = 'kline_1m'
        self.binance_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.com-futures")
        stream = self.binance_websocket_api_manager.create_stream(tf, market)
        self.binance_websocket_api_manager.subscribe_to_stream(stream, channels, markets)

    async def run(self):
        in_position = False
        while True:
            #one day ago works // "1 Dec, 2017", "1 Jan, 2018"  "1 day ago UTC"  run this to get historical data
            dfold = client.get_historical_klines("ETHUSDT",  Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
            # print(dfold)
            dfold = pd.DataFrame(dfold, columns=['timeunixms', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qoute assest volume', 'number of trades', 'taker buy base assest volume', 'taker buy qoute assest volume', 'ignore'])
            # print(dfold.head())
            dfold.insert(0, 'time', pd.to_datetime(dfold['timeunixms'], unit='ms'))
            # print(dfold.head())
                        # dfold.insert(0, 'time, pd.to_datetime(dfold['datetime'], unit='ms') = pd.to_datetime(dfold['time'], unit='ms')

            dfold.set_index(dfold['time'], inplace=True)
            dfold.insert(1, 'price', dfold['close'])
            dfold.to_csv('dfold.csv')
            # dfold = pd.read_csv('dfold.csv')
            # print(dfold.head())

            received_stream_data_json = self.binance_websocket_api_manager.pop_stream_data_from_stream_buffer()
            if received_stream_data_json:
                json_data = json.loads(received_stream_data_json)
                # print(json_data)
                candle_data = json_data.get('data', {})

                candle = candle_data.get('k', {})
                symbol = candle.get('s', {})
                timeframe = candle.get('i', {})
                close_prices = candle.get('c', {})
                open_prices = candle.get('o', {})
                closetime = candle_data.get('E', {})

                is_candle_closed = candle.get('x', {})
                # print(candle_data)
                if is_candle_closed:
                    print("candle closed at {}, {}".format(close_prices,pd.to_datetime(closetime, unit='ms')))
                    # closes.append(float(close_prices))
                    closes.append(close_prices)
                    # print("closes")
                    # print(closes)
                    # closes_times.append(float(closetime))
                    closes_times.append(pd.to_datetime(closetime, unit='ms'))
                    now.append(closetime)
                    # print(closes)
                    # print(closes_times)
                    zip2 = zip(closes_times, closes,now)
                    # print(zip2)
                    df = pd.DataFrame(zip2, columns= ['time', 'price', 'timeunixms'])
                    df.set_index(df['time'], inplace=True)
                    # print(df.head())

                    
                    # df['price'] = np.asarray(closes) 
                    # df['time'] = np.asarray(now)
                    # now.append(datetime.now())
  
                    # print(df.head())
                    df.to_csv('binanceData.csv')

                    # plt.figure(figsize = (18,9))
                    # # print("df shape{}".format(df.shape[0]))
                    # plt.plot(range(df.shape[0]),(df['price']))
                    # plt.xticks(range(0,df.shape[0],500),df['time'].loc[::500],rotation=45)
                    # plt.xlabel('time',fontsize=18)
                    # plt.ylabel('price',fontsize=18)
                    # plt.show()
                    frames = [dfold, df]
                    dfmega = pd.concat(frames)
                    dfmega.to_csv('mega.csv')
                    # print(dfmega.head())
                    # dfmega.iloc[0: , 0, 2]

                    # training_set = dfmega.iloc[:800, 0:2].values
                    # test_set = dfmega.iloc[800:, 0:2].values
                    # print(training_set)
                    # data=dfmega.sort_index(ascending=True,axis=0)
                    sc=MinMaxScaler(feature_range=(0,1))
                    # new_dataset=pd.DataFrame(index=range(0,len(dfmega)),columns=['Date','Close'])
                    new_dataset = dfmega[['timeunixms', 'price']].copy()
                    new_dataset = new_dataset.rename(columns={'timeunixms':'Date', 'price':'Close'})

                    # for i in range(0,len(dfmega)):
                    #     # new_dataset["Date"][i]=dfmega['time'][i]
                    #     new_dataset["Date"][i]=dfmega['timeunixms'][i]
                    #     # print(dfmega['timeunixms'][i])
                    #     new_dataset["Close"][i]=dfmega["price"][i]
                    #     print(dfmega["price"][i])

                    # print(new_dataset.head())

                    new_dataset.set_index(new_dataset["Date"], inplace = True)
                    # new_dataset.to_csv('new.csv')

                    # new_dataset.drop("Date",axis=1,inplace=True)

                    # final_dataset=new_dataset.values

                    train_data=new_dataset.iloc[0:800,:].values
                    # print(train_data)
                    valid_data=new_dataset.iloc[800:,:].values

                    training_set = new_dataset.iloc[0:800,:].values
                    # print(training_set)
                    test_set = new_dataset.iloc[800:,:].values

                    training_set_scaled=sc.fit_transform(training_set)
                    # Feature Scaling
                    # sc = MinMaxScaler(feature_range = (0, 1))
                    # training_set_scaled = sc.fit_transform(training_set)
                    # Creating a data structure with 60 time-steps and 1 output
                    X_train = []
                    y_train = []
                    for i in range(60, 800):
                        X_train.append(training_set_scaled[i-60:i, 0])
                        y_train.append(training_set_scaled[i, 0])
                    X_train, y_train = np.array(X_train), np.array(y_train)
                    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                    #(740, 60, 1)
                    model = Sequential()
                    #Adding the first LSTM layer and some Dropout regularisation
                    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
                    model.add(Dropout(0.2))
                    # Adding a second LSTM layer and some Dropout regularisation
                    model.add(LSTM(units = 50, return_sequences = True))
                    model.add(Dropout(0.2))
                    # Adding a third LSTM layer and some Dropout regularisation
                    model.add(LSTM(units = 50, return_sequences = True))
                    model.add(Dropout(0.2))
                    # Adding a fourth LSTM layer and some Dropout regularisation
                    model.add(LSTM(units = 50))
                    model.add(Dropout(0.2))
                    # Adding the output layer
                    model.add(Dense(units = 1))

                    # Compiling the RNN
                    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
                    # model.summary()

                    # Fitting the RNN to the Training set
                    model.fit(X_train, y_train, epochs = 100, batch_size = 32)
                    
                    # model.save('kbot2')


                    # dataset_train = dfmega.iloc[:800, 0:2]
                    # dataset_test = dfmega.iloc[800:, 0:2]

                    dataset_train = new_dataset.iloc[0:800,0:]
                    dataset_test = new_dataset.iloc[800:,0:]
                    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
                    dataset_total.to_csv('datasettotal.csv')
                    # print(dataset_total.head())
                    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
                    inputs = inputs.reshape(-1,2)
                    # print(inputs)
                    inputs = sc.transform(inputs)
                    X_test = []
                    for i in range(60, 519):
                        X_test.append(inputs[i-60:i, 0])
                    X_test = np.array(X_test)

                    # X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1] , 1))
                    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1] , 1))

                    # print(X_test.shape)
                    # (459, 60, 1)

                    predicted_stock_price = model.predict(X_test)
                    train_predict = predicted_stock_price
                    train_predict_new = np.zeros(shape=(len(train_predict), 2))
                    train_predict_new[:,0:] = train_predict[:,0:]
                    trainPredict = sc.inverse_transform(train_predict_new)[:,0:]
                    # print(trainPredict)

                    # predicted_stock_price = sc.inverse_transform(predicted_stock_price)

                    x = dataset_total['Date']
                    y = dataset_total['Close']
                    x2 = dataset_total['Date']
                    y2 = trainPredict
                    dfpredict = pd.DataFrame(trainPredict, columns=['Date', 'Close'])
                    # for i in dfpredict['Date']:
                    #     dfpredict['time'] = pd.to_datetime(i, unit='ms')
                    dfpredict['time'] = dfpredict['Date'].apply(lambda x: pd.to_datetime(x, unit='ms'))
                        # print(pd.to_datetime(i, unit='ms'))
                    # print(dfpredict.head())
                    dfpredict.set_index(dfpredict['Date'], inplace=True)
                    dfpredict.to_csv('predictions.csv')
                    x2 = dfpredict['Date']

                    #plot values $$$$$$$$$$$$$$$$$$$$$$$$$$$$$   need help fixing these graphsET
                    tickvalues = range(0,len(x))
                    plt.figure(figsize = (20,5))
                    plt.title('ETH Stock Price')
                    plt.xticks(ticks = tickvalues ,label = 'real eth price', rotation = 'vertical')
                    plt.xlabel('Time')
                    plt.ylabel('ETH Price')
                    plt.plot(x,y)
                    # plt.plot(x2,y2, color='red')
                    plt.show()

                    data = np.genfromtxt("predictions.csv", delimiter=",", names=["Date", "Close"])
                    plt.title('ETH Stock Price Prediction')
                    plt.xlabel('Time')
                    plt.ylabel('ETH Stock Price')
                    plt.plot(data['Date'], data['Close'])
                    plt.show()

                    model = load_model('kbot')

                    # define input
                    X = X_test

                    # make predictions
                    yhat = model.predict(X)
                    # print(yhat) 
                    dfy = pd.DataFrame(yhat)
                    # print(dfy.head())

                    #plot values $$$$$$$$$$$$$$$$$$$$$$$$$$$$$

                    futureElement = dfpredict['Close'].iloc[-1]
                    print(futureElement)
                    # print(type(futureElement))
                    # print(type(df['price'].iloc[-1]))

                    futureElements = []
                    futureElements.append(futureElement)
                    if futureElement >= float(df['price'].iloc[-1]):
                        if in_position:
                            print("It is oversold, but you already own it, nothing to do.")
                        else:
                            print("Oversold! Buy! Buy! Buy!")
                                # put binance buy order logic here
                            order_succeeded = order(SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)
                            if order_succeeded:
                                in_position = True
                    # 1.1*float(df['price'].iloc[-1]     1.1*float(min(df['price']) still working on when to sell
                    if futureElements[-1] >= 1.1*float(min(df['price'])):
                        if in_position:
                            print("Overbought! Sell! Sell! Sell!")
                                        # put binance sell logic here
                            order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)
                            if order_succeeded:
                                in_position = False      


async def main():
    tasks = []

    # channels = ['kline_1m', 'kline_5m', 'kline_15m', 'kline_30m', 'kline_1h', 'kline_12h', 'miniTicker']
    # markets = {'btcusdt', 'ethusdt', 'ltcusdt'}
    channels = ['kline_1m']
    markets = {'ethusdt'}   

    print(f'Main starting streams ... ')
    kl_socket = BinanceWs(channels=channels, markets=markets)
    task = await kl_socket.run()
    tasks.append(task)
    print(tasks)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # asyncio.run(main())

    run_async(main) 


