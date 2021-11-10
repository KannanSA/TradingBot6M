# pip freeze > requirements.txt
# conda env export | grep -v "^prefix: " > environment.yml




import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
df=pd.read_csv("TSLA.csv")
print("Number of rows and columns:", df.shape)
df.head(5)

training_set = df.iloc[:800, 1:2].values
test_set = df.iloc[800:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
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

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Getting the predicted stock price of 2017
dataset_train = df.iloc[:800, 1:2]
dataset_test = df.iloc[800:, 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 519):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)
# (459, 60, 1)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(df.loc[800:, "Date"],dataset_test.values, color = "red", label = "Real TESLA Stock Price")
plt.plot(df.loc[800:, "Date"],predicted_stock_price, color = "blue", label = "Predicted TESLA Stock Price")
plt.xticks(np.arange(0,459,50))
plt.title('TESLA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TESLA Stock Price')
plt.legend()
plt.show()


                    # plt.plot(dataset_total.loc[:, "Date"],dataset_total.values, color = "red", label = "Real ETH Stock Price")
                    # # plt.plot(dataset_total.loc[:, "Date"],trainPredict, color = "blue", label = "Predicted ETH Stock Price")
                    # plt.xticks(np.arange(0,459,50))
                    # plt.title('ETH Stock Price Prediction')
                    # plt.xlabel('Time')
                    # plt.ylabel('ETH Stock Price')
                    # plt.legend()
                    # plt.show()


                    # Visualising the results
                    # plt.plot(dataset_total.loc[800:, "Date"],dataset_total.values, color = "red", label = "Real ETH  Price")
                    # plt.plot(dataset_total.loc[800:, "Date"],trainPredict, color = "blue", label = "Predicted ETH  Price")
                    # plt.xticks(np.arange(0,459,50))
                    # plt.title('ETH Stock Price Prediction')
                    # plt.xlabel('Time')
                    # plt.ylabel('ETH Price')
                    # plt.legend()
                    # plt.show()
                                        

                # do stuff with data ... 


# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),(df['Close']))
# plt.xticks(range(0,df.shape[0],500),df['Close_time'].loc[::500],rotation=45)
# plt.xlabel('Close_time',fontsize=18)
# plt.ylabel('Close',fontsize=18)
# plt.show()

# df = pd.DataFrame()
# # df['price']  = closes
# # df['time']  = closes_times

# training_set = df['time'], df['price']


# for i in df['Close_time']:
#     x = np.asarray(i).astype('float32')
# # x = np.stack(x, axis=0) 
# for i in df['Close']:
#     y = np.asarray(i).astype('float32')


# data = (x.reshape(-1,1), y.reshape(-1,1))

# X_train = x.reshape(-1, 1,1)
# X_test  = x.reshape(-1, 1,1)
# y_train = y.reshape(-1, 1,1)
# y_test = y.reshape(-1, 1,1)



# model = Sequential()
# model.add(LSTM(100, input_shape=(1, 1), return_sequences=True))
# model.add(LSTM(5, input_shape=(1, 1), return_sequences=True))
# model.add(Dense(24))
# model.compile(optimizer='adam', loss='mse', metrics= ['mae', 'mape', 'acc'])

# # model.compile(loss="mean_absolute_error", optimizer="adam", metrics= ['accuracy'])
# model.summary()

# history = model.fit(X_train,y_train,epochs=100, validation_data=(X_test,y_test))



# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),(df['price']))
# plt.xticks(range(0,df.shape[0],500),df['time'].loc[::500],rotation=45)
# plt.xlabel('time',fontsize=18)
# plt.ylabel('price',fontsize=18)
# plt.show()


# model = Sequential()
# model.add(LSTM(32, input_shape=(10, 1)))
# model.add(Dense(1))
# model.summary()



# in_position = False

# client = Client(config.API_KEY, config.API_SECRET, tld='us')

# def order(side, quantity, symbol,order_type=ORDER_TYPE_MARKET):
#     try:
#         print("sending order")
#         order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
#         print(order)
#     except Exception as e:
#         print("an exception occured - {}".format(e))
#         return False

#     return True

    
# def on_open(ws):
#     print('opened connection')

# def on_close(ws):
#     print('closed connection')

# def on_message(ws, message):
#     global closes, in_position, closes_times
    
#     print('received message')
#     json_message = json.loads(message)
#     # pprint.pprint(json_message)

#     candle = json_message['k']

#     is_candle_closed = candle['x']
#     close = candle['c']
#     closetime = candle['E']

#     if is_candle_closed:
#         # df = pd.DataFrame()
#         # df['price']  = closes
#         # df['time']  = closes_times
#         print("candle closed at {}".format(close))
#         closes.append(float(close))
#         print("closes")
#         print(closes)
#         closes_times.append(float(closetime))
#         df['price']  = np.asarray(closes)
#         df['time']  = np.asarray(closes_times)
#         print(df.head())

#         if len(closes) > RSI_PERIOD:
#             np_closes = numpy.array(closes)
#             rsi = talib.RSI(np_closes, RSI_PERIOD)
#             print("all rsis calculated so far")
#             print(rsi)
#             last_rsi = rsi[-1]
#             print("the current rsi is {}".format(last_rsi))

#             if last_rsi > RSI_OVERBOUGHT:
#                 if in_position:
#                     print("Overbought! Sell! Sell! Sell!")
#                     # put binance sell logic here
#                     order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)
#                     if order_succeeded:
#                         in_position = False
#                 else:
#                     print("It is overbought, but we don't own any. Nothing to do.")
            
#             if last_rsi < RSI_OVERSOLD:
#                 if in_position:
#                     print("It is oversold, but you already own it, nothing to do.")
#                 else:
#                     print("Oversold! Buy! Buy! Buy!")
#                     # put binance buy order logic here
#                     order_succeeded = order(SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)
#                     if order_succeeded:
#                         in_position = True

                
# ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
# ws.run_forever()













# $$$$$$$$ waste code 



# start = '2019-02-28'
# end = '2020-02-28'

# tickers = ['AAPL']

# thelen = len(tickers)

# price_data = []
# for ticker in tickers:
#     prices = wb.DataReader(ticker, start = start, end = end, data_source='yahoo')[['Open','Adj Close']]
#     price_data.append(prices.assign(ticker=ticker)[['ticker', 'Open', 'Adj Close']])

# #names = np.reshape(price_data, (len(price_data), 1))

# df = pd.concat(price_data)
# df.reset_index(inplace=True)

# for col in df.columns: 
#     print(col) 

# #used for setting the output figure size
# rcParams['figure.figsize'] = 20,10
# #to normalize the given input data
# scaler = MinMaxScaler(feature_range=(0, 1))
# #to read input data set (place the file name inside  ' ') as shown below
# df.head()

# df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
# #df.index = names['Date']
# plt.figure(figsize=(16,8))
# plt.plot(df['Adj Close'], label='Closing Price')


# ntrain = 80
# df_train = df.head(int(len(df)*(ntrain/100)))
# ntest = -80
# df_test = df.tail(int(len(df)*(ntest/100)))


# #importing the packages 
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM

# #dataframe creation
# seriesdata = df.sort_index(ascending=True, axis=0)
# new_seriesdata = pd.DataFrame(index=range(0,len(df)),columns=['Date','Adj Close'])
# length_of_data=len(seriesdata)
# for i in range(0,length_of_data):
#     new_seriesdata['Date'][i] = seriesdata['Date'][i]
#     new_seriesdata['Adj Close'][i] = seriesdata['Adj Close'][i]
# #setting the index again
# new_seriesdata.index = new_seriesdata.Date
# new_seriesdata.drop('Date', axis=1, inplace=True)
# #creating train and test sets this comprises the entire data’s present in the dataset
# myseriesdataset = new_seriesdata.values
# totrain = myseriesdataset[0:255,:]
# tovalid = myseriesdataset[255:,:]
# #converting dataset into x_train and y_train
# scalerdata = MinMaxScaler(feature_range=(0, 1))
# scale_data = scalerdata.fit_transform(myseriesdataset)
# x_totrain, y_totrain = [], []
# length_of_totrain=len(totrain)
# for i in range(60,length_of_totrain):
#     x_totrain.append(scale_data[i-60:i,0])
#     y_totrain.append(scale_data[i,0])
# x_totrain, y_totrain = np.array(x_totrain), np.array(y_totrain)
# x_totrain = np.reshape(x_totrain, (x_totrain.shape[0],x_totrain.shape[1],1))

# #LSTM neural network
# lstm_model = Sequential()
# lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_totrain.shape[1],1)))
# lstm_model.add(LSTM(units=50))
# lstm_model.add(Dense(1))
# lstm_model.compile(loss='mean_squared_error', optimizer='adadelta')
# lstm_model.fit(x_totrain, y_totrain, epochs=3, batch_size=1, verbose=2)
# #predicting next data stock price
# myinputs = new_seriesdata[len(new_seriesdata) - (len(tovalid)+1) - 60:].values
# myinputs = myinputs.reshape(-1,1)
# myinputs  = scalerdata.transform(myinputs)
# tostore_test_result = []
# for i in range(60,myinputs.shape[0]):
#     tostore_test_result.append(myinputs[i-60:i,0])
# tostore_test_result = np.array(tostore_test_result)
# tostore_test_result = np.reshape(tostore_test_result,(tostore_test_result.shape[0],tostore_test_result.shape[1],1))
# myclosing_priceresult = lstm_model.predict(tostore_test_result)
# myclosing_priceresult = scalerdata.inverse_transform(myclosing_priceresult)



# totrain = df_train
# tovalid = df_test

# #predicting next data stock price
# myinputs = new_seriesdata[len(new_seriesdata) - (len(tovalid)+1) - 60:].values

# #  Printing the next day’s predicted stock price. 
# print(len(tostore_test_result));
# print(myclosing_priceresult);


# class myThread (threading.Thread):
#    def __init__(self, threadID, name, counter):
#       threading.Thread.__init__(self)
#       self.threadID = threadID
#       self.name = name
#       self.counter = counter
#    def run(self):
#       print( "Starting " + self.name)
#       # Get lock to synchronize threads
#       threadLock.acquire()
#       print_time(self.name, self.counter, 3)
#       # Free lock to release next thread
#       threadLock.release()

# def print_time(threadName, delay, counter):
#    while counter:
#       time.sleep(delay)
#       print ("%s: %s" % (threadName, time.ctime(time.time())))
#       counter -= 1

# threadLock = threading.Lock()
# threads = []

# # Create new threads
# thread1 = myThread(1, "Thread-1", 1)
# thread2 = myThread(2, "Thread-2", 2)

# # Start new Threads
# thread1.start()
# thread2.start()

# # Add threads to thread list
# threads.append(thread1)
# threads.append(thread2)

# # Wait for all threads to complete
# for t in threads:
#     t.join()
# print ("Exiting Main Thread")


# connection = websockets.connect(uri=SOCKET)

# # The client is also as an asynchronous context manager.
# async with connection as websocket:
#     # # Sends a message.
#     # await websocket.send(
#     #     '{"method": "GetDocument", "params": {"id": "ad36g4y5"}, "id": 1}')

#     # Receives the replies.
#     async for message in websocket:
#         print(message)
        
#     # Closes the connection.
#     await websocket.close()

                    # trainPredict = model.predict(X_test)
                    # train_predict = predicted_stock_price

                    # trainPredict_dataset_like = np.zeros(shape=(len(trainPredict), 1) )
                    # # put the predicted values in the right field
                    # trainPredict_dataset_like[:,0:] = trainPredict[:,0:]
                    # # inverse transform and then select the right field
                    # trainPredict = sc.inverse_transform(trainPredict_dataset_like)[:,0]