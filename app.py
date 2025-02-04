import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from minepy import MINE #loading class to select features using MIC (Maximal Information Coefficient)
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, RepeatVector, Bidirectional, LSTM, GRU, AveragePooling2D
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
import pickle
import matplotlib.pyplot as plt #use to visualize dataset vallues
from keras.layers import *
from keras.models import *
from keras import backend as K
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = 'welcome'

#defining self attention layer
class attention(Layer):
    def __init__(self, return_sequences=True, name=None, **kwargs):
        super(attention,self).__init__(name=name)
        self.return_sequences = return_sequences
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")

        super(attention,self).build(input_shape)

    def call(self, x):

        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

        #class to normalize dataset values
scaler = MinMaxScaler(feature_range = (0, 1))
scaler1 = MinMaxScaler(feature_range = (0, 1))

#loading and displaying Aneshtesia clinical dataset
dataset = pd.read_csv("Dataset/household_power_consumption.csv", sep=";", nrows=10000)
#converting sub meter values as float data     
dataset['Sub_metering_1'] = dataset['Sub_metering_1'].astype(float)
dataset['Sub_metering_2'] = dataset['Sub_metering_2'].astype(float)
dataset['Sub_metering_3'] = dataset['Sub_metering_3'].astype(float)
dataset.fillna(0, inplace = True)
dataset

#applying dataset processing such as converting date and time into numeric values and then summing all 3
#submeters consumption as single target value to forecast future electricity
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['year'] = dataset['Date'].dt.year
dataset['month'] = dataset['Date'].dt.month
dataset['day'] = dataset['Date'].dt.day
dataset['Time'] = pd.to_datetime(dataset['Time'])
dataset['hour'] = dataset['Time'].dt.hour
dataset['minute'] = dataset['Time'].dt.minute
dataset['second'] = dataset['Time'].dt.second
dataset['label'] = dataset['Sub_metering_1'] + dataset['Sub_metering_2'] + dataset['Sub_metering_3']
dataset.drop(['Date', 'Time', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis = 1,inplace=True)
dataset.fillna(0, inplace = True)
dataset
#applying MIC (Maximal Information Coefficient) algorithm to select least similar or correlated features to avoid 
#poor prediction values
Y = dataset['label'].ravel() #getting target column
dataset.drop(['label'], axis = 1,inplace=True)
columns = dataset.columns
X = dataset.values #get dataset features
print("Total features exists in Dataset before applying MIC features Selection algorithm : "+str(X.shape[1]))
mic_scores = []
mine = MINE()
for i in range(0, len(columns)-1):#loop and compute mic score for each features
    mine.compute_score(X[:,i], Y)
    mic_scores.append((columns[i], mine.mic()))
# Sort features by MIC score
mic_scores.sort(key=lambda x: x[1], reverse=True)
# Select top features
top_features = [feature for feature, _ in mic_scores[:8]]  # Select top 2 features
X = dataset[top_features]
print("Total features exists in Dataset before applying MIC features Selection algorithm : "+str(X.shape[1]))
X = dataset.values

#normalizing selected features using MINMAX scaler
Y = Y.reshape(-1, 1)
scaler = MinMaxScaler((0,1))
scaler1 = MinMaxScaler((0,1))
X = dataset.values
X = scaler.fit_transform(X)
Y = scaler1.fit_transform(Y)
print("Normalized Features = "+str(X))

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print("Train & Test Dataset Split")
print("80% records used to train algorithms : "+str(X_train.shape[0]))
print("20% records features used to test algorithms : "+str(X_test.shape[0]))

#defining global variables to save algorithm performnace metrics
rsquare = []
rmse = []
mae = []

#training propose CNN-BiLSTM-SA algorithm on training features and then evaluate performance on 20% test features
#this algorithm is a combination of CNN, BI-LSTM and SA (self attention)
X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
propose_model = Sequential()
#adding CNN layer
propose_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
propose_model.add(MaxPooling2D(pool_size = (1, 1)))
propose_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
propose_model.add(MaxPooling2D(pool_size = (1, 1)))
propose_model.add(Flatten())
propose_model.add(RepeatVector(3))
propose_model.add(attention(return_sequences=True,name='attention')) # ========adding Attention layer
#adding bidirectional LSTM as CRNN layer
propose_model.add(Bidirectional(LSTM(64, activation = 'relu')))#==================adding BILSTM
propose_model.add(RepeatVector(3))
propose_model.add(Bidirectional(LSTM(64, activation = 'relu')))#==================adding BILSTM
#defining output classification layer with 256 neurons 
propose_model.add(Dense(units = 256, activation = 'relu'))
propose_model.add(Dropout(0.3))
propose_model.add(Dense(units = 1))
propose_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
if os.path.exists("model/propose_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/propose_weights.hdf5', verbose = 1, save_best_only = True)
    hist = propose_model.fit(X_train1, y_train, batch_size = 16, epochs = 50, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/propose_hist.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    propose_model.load_weights("model/propose_weights.hdf5")
#perform prediction on test data
predict = propose_model.predict(X_test1)

extension_model = Sequential()
extension_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
extension_model.add(MaxPooling2D(pool_size = (1, 1)))
extension_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
extension_model.add(MaxPooling2D(pool_size = (1, 1)))
extension_model.add(Flatten())
extension_model.add(RepeatVector(3))
extension_model.add(attention(return_sequences=True,name='attention')) 
extension_model.add(Bidirectional(GRU(64, activation = 'relu')))
extension_model.add(RepeatVector(3))
extension_model.add(Bidirectional(GRU(64, activation = 'relu'))) 
extension_model.add(Dense(units = 256, activation = 'relu'))
extension_model.add(Dropout(0.3))
extension_model.add(Dense(units = 1))
extension_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
if os.path.exists("model/extension_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
    hist = extension_model.fit(X_train1, y_train, batch_size = 16, epochs = 50, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/extension_hist.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    extension_model.load_weights("model/extension_weights.hdf5")
#perform prediction on test data
predict = extension_model.predict(X_test1)

#reading test data and then predicting dosage
testData = pd.read_csv("Dataset/testData.csv", sep=";")#read test data
data = testData.values
#handling and removing missing values        
testData.fillna(0, inplace = True)
testData['Date'] = pd.to_datetime(testData['Date'])#convert date and time to year, month, day, hour, second and minutes
testData['year'] = testData['Date'].dt.year
testData['month'] = testData['Date'].dt.month
testData['day'] = testData['Date'].dt.day
testData['Time'] = pd.to_datetime(testData['Time'])
testData['hour'] = testData['Time'].dt.hour
testData['minute'] = testData['Time'].dt.minute
testData['second'] = testData['Time'].dt.second
testData.drop(['Date', 'Time'], axis = 1,inplace=True)
testData.fillna(0, inplace = True)
X = testData[top_features]#select MIC top features
testData = testData.values
testData = scaler.transform(testData)#normalize dataset values
testData = np.reshape(testData, (testData.shape[0], testData.shape[1], 1, 1))
predict = extension_model.predict(testData)#predict electricity consumption using extension model
predict = predict.reshape(-1, 1)
predict = scaler1.inverse_transform(predict)#reverse normalize predicted SOC to normal integer value
for i in range(len(predict)):
    print("Test Data = "+str(data[i])+" Predicted Electricity Consumption ===> "+str(abs(predict[i,0])))
    print()

@app.route('/Predict', methods=['GET', 'POST'])
def predictView():
    return render_template('Predict.html', msg='')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/index', methods=['GET', 'POST'])
def index1():
    return render_template('index.html', msg='')


def getModel():
    extension_model = Sequential()
    extension_model.add(Convolution2D(32, (1 , 1), input_shape = (10, 1, 1), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_model.add(Flatten())
    extension_model.add(RepeatVector(3))
    extension_model.add(attention(return_sequences=True,name='attention'))
    extension_model.add(Bidirectional(GRU(64, activation = 'relu')))
    extension_model.add(RepeatVector(3))
    extension_model.add(Bidirectional(GRU(64, activation = 'relu'))) 
    extension_model.add(Dense(units = 256, activation = 'relu'))
    extension_model.add(Dropout(0.3))
    extension_model.add(Dense(units = 1))
    extension_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    extension_model.load_weights("model/extension_weights.hdf5")
    return extension_model

@app.route('/AdminLogin', methods=['GET', 'POST'])
def AdminLogin():
    return render_template('AdminLogin.html', msg='')

@app.route('/AdminLoginAction', methods=['GET', 'POST'])
def AdminLoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('AdminScreen.html', msg="Welcome "+user)
        else:
            return render_template('AdminLogin.html', msg="Invalid login details")

@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        extension_model = getModel()
        testData = pd.read_csv("Dataset/testData.csv", sep=";")
        data = testData.values        
        testData.fillna(0, inplace = True)
        testData['Date'] = pd.to_datetime(testData['Date'])
        testData['year'] = testData['Date'].dt.year
        testData['month'] = testData['Date'].dt.month
        testData['day'] = testData['Date'].dt.day
        testData['Time'] = pd.to_datetime(testData['Time'])
        testData['hour'] = testData['Time'].dt.hour
        testData['minute'] = testData['Time'].dt.minute
        testData['second'] = testData['Time'].dt.second
        testData.drop(['Date', 'Time'], axis = 1,inplace=True)
        testData.fillna(0, inplace = True)
        X = testData[top_features]
        testData = testData.values
        testData = scaler.transform(testData)
        testData = np.reshape(testData, (testData.shape[0], testData.shape[1], 1, 1))
        predict = extension_model.predict(testData)
        predict = predict.reshape(-1, 1)
        predict = scaler1.inverse_transform(predict)
        output = ""
        for i in range(len(predict)):
            output += "Test Data = "+str(data[i])+" Predicted Electricity Consumption ===> "+str(predict[i,0])+"<br/><br/>"
        return render_template('Predict.html', msg=output)

if __name__ == '__main__':
    app.run()
