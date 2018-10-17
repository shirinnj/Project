%%writefile lstm.py
import math
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
'''
LSTM model to predict taxi ridership demand
'''
# convert an array of values into a dataset matrix
def create_dataset(DATASET, LOOK_BACK=1, LOOK_FORWARD=5):
    DATA_X, DATA_Y = [], []
    for i in range(len(DATASET)- LOOK_BACK- 1):
        NEW = DATASET[i:(i+ LOOK_BACK), 0]
        DATA_X.append(NEW)
        DATA_Y.append(DATASET[i + LOOK_BACK:i+ LOOK_BACK+ LOOK_FORWARD, 0])
    return numpy.array(DATA_X), numpy.array(DATA_Y)

def create_dataFrame(X, Y, PREDICT):
    NEW = pd.concat([pd.DataFrame(PREDICT, columns=["Predict"]), pd.DataFrame(Y, columns=["Y"])], axis=1)
    NEW = NEW[["Y", "Predict"]]
    return NEW

def create_final_output(INPUT_FOLDER, FILE_NAME, TRAIN_X, TRAIN_Y, 
                                       TRAIN_PREDICT, TEST_X, TEST_Y, TEST_PREDICT):
    RES = create_dataFrame(TEST_X, TEST_Y, TEST_PREDICT)
    RES_NEW = pd.DataFrame([scaler.inverse_transform(RES[k]) for k in RES.columns]).T
    RES_NEW.columns = RES.columns
    RES.columns = [k + "_normalized" for k in RES.columns]
    TEST_FINAL = pd.concat([RES, RES_NEW], axis=1)
    TEST_FINAL.to_csv(INPUT_FOLDER + FILE_NAME+ "_test_result_final.csv", index=None)

    RES_TRAIN = create_dataFrame(TRAIN_X, TRAIN_Y, TRAIN_PREDICT)
    RES_TRAIN_NEW = pd.DataFrame([scaler.inverse_transform(RES_TRAIN[k]) 
                                  for k in RES_TRAIN.columns]).T
    RES_TRAIN_NEW.columns = RES_TRAIN.columns
    RES_TRAIN.columns = [k + "_normalized" for k in RES_TRAIN.columns]
    TRAIN_FINAL = pd.concat([RES_TRAIN, RES_TRAIN_NEW], axis=1)
    TRAIN_FINAL.to_csv(INPUT_FOLDER + FILE_NAME+"_train_result_final.csv", index=None)
    return TRAIN_FINAL, TEST_FINAL

if __name__ == "__main__":
    INPUT_FOLDER = "../data/entire data/LSTM/"
    FILE_NAME = "pc3_2016_1_3_total"
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    DATAFRAME = pd.read_csv(INPUT_FOLDER + FILE_NAME + ".csv", usecols=[1], engine='python', skipfooter=3)
    DATASET = DATAFRAME.values
    DATASET = DATASET.astype('float32')
    # normalize the dataset
    SCALAR = MinMaxScaler(feature_range=(0, 1))
    DATASET = SCALAR.fit_transform(DATASET)
    # split into train and test sets
    TRAIN_SIZE = int(len(DATASET) * 0.67)
    TEST_SIZE = len(DATASET) - TRAIN_SIZE
    TRAIN, TEST = DATASET[0:TRAIN_SIZE, :], DATASET[TRAIN_SIZE:len(DATASET), :]
    print(len(TRAIN), len(TEST))    
    # reshape into X=t and Y=t+1
    LOOK_BACK = 24
    LOOK_FORWARD = 12
    TRAIN_X, TRAIN_Y = create_dataset(TRAIN, LOOK_BACK, LOOK_FORWARD)
    TEST_X, TEST_Y = create_dataset(TEST, LOOK_BACK, LOOK_FORWARD)
    # reshape input to be [samples, time steps, features]
    TRAIN_X = numpy.reshape(TRAIN_X, (TRAIN_X.shape[0], 1, TRAIN_X.shape[1]))
    TEST_X = numpy.reshape(TEST_X, (TEST_X.shape[0], 1, TEST_X.shape[1]))
    # create and fit the LSTM network
    MODEL = Sequential()
    MODEL.add(LSTM(4, input_shape=(1, LOOK_BACK)))
    MODEL.add(Dense(1))
    MODEL.compile(loss = 'mean_squared_error', optimizer = 'adam')
    MODEL.fit(TRAIN_X, TRAIN_Y, epochs=4, batch_size=1, verbose=2)
    # make predictions
    TRAIN_PREDICT = MODEL.predict(TRAIN_X)
    TEST_PREDICT = MODEL.predict(TEST_X)  
    TRAIN_FINAL, TEST_FINAL = create_final_output(INPUT_FOLDER, FILE_NAME, TRAIN_X, TRAIN_Y,
                                                TRAIN_PREDICT, TEST_X, TEST_Y, TEST_PREDICT)    
    # invert predictions
    TRAIN_PREDICT = SCALAR.inverse_transform(TRAIN_PREDICT)
    TRAIN_Y = SCALAR.inverse_transform([TRAIN_Y])
    TEST_PREDICT = SCALAR.inverse_transform(TEST_PREDICT)
    TEST_Y = SCALAR.inverse_transform([TEST_Y])
    numpy.savetxt(INPUT_FOLDER+ FILE_NAME+ "_train_predict.csv", TRAIN_PREDICT)
    numpy.savetxt(INPUT_FOLDER+ FILE_NAME+ "_test_predict.csv", TEST_PREDICT)
    numpy.savetxt(INPUT_FOLDER+ FILE_NAME+ "_train_Y.csv", TRAIN_Y)
    numpy.savetxt(INPUT_FOLDER+ FILE_NAME+ "_test_Y.csv", TEST_Y)
    # calculate root mean squared error
    TRAIN_SCORE = math.sqrt(mean_squared_error(TRAIN_Y[0], TRAIN_PREDICT[:, 0]))
    print('Train Score: %.2f RMSE' % (TRAIN_SCORE))
    TEST_SCORE = math.sqrt(mean_squared_error(TEST_Y[0], TEST_PREDICT[:, 0]))
    print('Test Score: %.2f RMSE' % (TEST_SCORE))
    # shift train predictions for plotting
    TRAIN_PPREDICT_PLOT = numpy.empty_like(DATASET)
    TRAIN_PPREDICT_PLOT[:, :] = numpy.nan
    TRAIN_PPREDICT_PLOT[LOOK_BACK:len(TRAIN_PREDICT)+ LOOK_BACK, :] = TRAIN_PREDICT
    # shift test predictions for plotting
    TEST_PPREDICT_PLOT = numpy.empty_like(DATASET)
    TEST_PPREDICT_PLOT[:, :] = numpy.nan
    TEST_PPREDICT_PLOT[len(TRAIN_PREDICT)+ (LOOK_BACK*2)+ 1:len(DATASET)-1, :] = TEST_PREDICT
    # plot baseline and predictions
    plt.suptitle(FILE_NAME, fontsize=12)
    plt.plot(SCALAR.inverse_transform(DATASET), label="Observed")
    plt.plot(TRAIN_PPREDICT_PLOT, label="Train_predict")
    plt.plot(TEST_PPREDICT_PLOT, label="Test_predict")
    plt.legend()
    plt.show()
