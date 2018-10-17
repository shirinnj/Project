# Project
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import datasets, linear_model

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_forward= 5):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back:i+look_back+look_forward, 0])
    return numpy.array(dataX), numpy.array(dataY)

def create_dataFrame(X,Y,Predict):
    new = pd.concat([pd.DataFrame(Predict,columns = ["Predict"]),pd.DataFrame(Y,columns =["Y"])],axis=1)
    new = new[["Y","Predict"]]
    return new

def create_final_output_test_and_train(INPUT_FOLDER,file_name,trainX,trainY,trainPredict,testX,testY,testPredict):
    res = create_dataFrame(testX,testY,testPredict)
    res_new = pd.DataFrame([scaler.inverse_transform(res[k]) for k in res.columns]).T
    res_new.columns =res.columns
    res.columns = [k + "_normalized" for k in res.columns]
    test_final = pd.concat([res,res_new],axis=1)
    test_final.to_csv(INPUT_FOLDER + file_name+"_test_result_final.csv", index=None)

    res_train = create_dataFrame(trainX,trainY,trainPredict)
    res_train_new = pd.DataFrame([scaler.inverse_transform(res_train[k]) for k in res_train.columns]).T
    res_train_new.columns =res_train.columns
    res_train.columns = [k + "_normalized" for k in res_train.columns]
    train_final = pd.concat([res_train,res_train_new],axis=1)
    train_final.to_csv(INPUT_FOLDER + file_name+"_train_result_final.csv", index =None)
    return train_final,test_final


if __name__=="__main__":
    
    INPUT_FOLDER="../data/entire data/LSTM/"
    file_name="pc3_2016_1_3_total"
    # fix random seed for reproducibility
    numpy.random.seed(7)
    
    # load the dataset
    dataframe = pd.read_csv(INPUT_FOLDER+file_name+".csv", usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))    
    
    # reshape into X=t and Y=t+1
    look_back = 24
    look_forward = 12
    trainX, trainY = create_dataset(train, look_back, look_forward)
    testX, testY = create_dataset(test, look_back, look_forward)
    
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=4, batch_size=1, verbose=2)
    
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)  
        
    train_final,test_final = create_final_output_test_and_train(INPUT_FOLDER,file_name,trainX,trainY,
                                                                trainPredict,testX,testY,testPredict)    

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    numpy.savetxt(INPUT_FOLDER+file_name+"_train_predict.csv",trainPredict)
    numpy.savetxt(INPUT_FOLDER+file_name+"_test_predict.csv",testPredict)
    numpy.savetxt(INPUT_FOLDER+file_name+"_train_Y.csv",trainY)
    numpy.savetxt(INPUT_FOLDER+file_name+"_test_Y.csv",testY)
    
    
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

    # plot baseline and predictions
    plt.suptitle(file_name, fontsize=12)
    plt.plot(scaler.inverse_transform(dataset), label="observed")
    plt.plot(trainPredictPlot, label="train_predict")
    plt.plot(testPredictPlot, label="test_predict")
    plt.legend()
    plt.show()
    
