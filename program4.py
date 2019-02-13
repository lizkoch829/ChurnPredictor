import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

##Liz Koch
##CS 461
##Program 4
##12/10/2018


file=pd.read_csv('program4input.txt.csv')

##make the string numbers into float
file['MonthlyCharges'] = pd.to_numeric(file['MonthlyCharges'],\
                                       errors='coerce')
file['TotalCharges'] = pd.to_numeric(file['TotalCharges'],\
                                     errors='coerce')
file['TotalCharges'].fillna(value = file['tenure'] * \
            file['MonthlyCharges'], inplace= True)

##read in all data except customer number and churn value
X=file.iloc[:,1:20].values
##churn value
Y=file.iloc[:,20].values


##transform all the strings into numbers that
##correspond to each string
##used code from https://stackoverflow.com/questions
##/44474570/sklearn-label-encoding-multiple-columns-pandas-dataframe

labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])
labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])
labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])
labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8])
labelencoder_X_9 = LabelEncoder()
X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])
labelencoder_X_10 = LabelEncoder()
X[:, 10] = labelencoder_X_10.fit_transform(X[:, 10])
labelencoder_X_11 = LabelEncoder()
X[:, 11] = labelencoder_X_11.fit_transform(X[:, 11])
labelencoder_X_12 = LabelEncoder()
X[:, 12] = labelencoder_X_12.fit_transform(X[:, 12])
labelencoder_X_13 = LabelEncoder()
X[:, 13] = labelencoder_X_13.fit_transform(X[:, 13])
labelencoder_X_14 = LabelEncoder()
X[:, 14] = labelencoder_X_14.fit_transform(X[:, 14])
labelencoder_X_15 = LabelEncoder()
X[:, 15] = labelencoder_X_15.fit_transform(X[:, 15])
labelencoder_X_16 = LabelEncoder()
X[:, 16] = labelencoder_X_15.fit_transform(X[:, 16])
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y[:])


##want to keep 30% for testing/validating
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size\
                        = 0.3, random_state = 0)

##standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
##input layer
model.add(Dense(output_dim = 10, init = 'uniform', activation \
     = 'relu',input_dim = 19))
##regularizer
model.add(Dropout(0.2))
##first hidden layer
model.add(Dense(output_dim = 10, init = 'uniform', activation\
                     = 'relu'))
model.add(Dropout(0.2))
##second hidden layer
model.add(Dense(output_dim = 10, init = 'uniform', activation\
                     = 'relu'))
model.add(Dropout(0.2))
##output layer
model.add(Dense(output_dim = 1, init = 'uniform', activation = \
                     'sigmoid'))
##compile network
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',\
                   metrics = ['accuracy'])

##fit model to training data
hist = model.fit(X_train, Y_train, batch_size = 40, nb_epoch = 10)
accuracy=hist.history['acc']

##test the remaining data
prediction = model.predict(X_test)
prediction = (prediction > 0.5)

##this creates a matrix that tells how the network performed
##with the format
## TT TF
## FT FF

confusion_mtrx = confusion_matrix(Y_test, prediction)
print(confusion_mtrx)




































##
