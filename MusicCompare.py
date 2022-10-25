# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 14:30:18 2022

@author: yehuh
"""
##fur elise => 16sec
##maiden prey => 23sec 
from  preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

PATH_GARBAGE_CAR = "./GarbageCar/"

# Second dimension of the feature is dim2
feature_dim_2 = 11

# Save data to array file first
save_data_to_array(DATA_PATH, max_len=feature_dim_2)

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test(path = DATA_PATH)

# # Feature dimension
feature_dim_1 = 20
channel = 1
epochs = 50
batch_size = 100
verbose = 1
labels = os.listdir(DATA_PATH)
label_indices = np.arange(0, len(labels))
num_classes = len(labels)

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model




# Reshaping to perform 2D convolution
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

model = get_model()
model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))

from keras.models import load_model
model.save('ASR.h5')  # creates a HDF5 file 'model.h5'

def getLabel(filepath):
    return os.listdir(filepath)
    

def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return model.predict(sample_reshaped)
    
def expectLab(filepath, model):
    pred_rslts = predict(filepath, model)
    
    same_melody_exist = False
    for preds in pred_rslts:
        for pred in preds:
            if(pred > 0.5):
                same_melody_exist = True
                break;
                
        if(same_melody_exist == True):
            break
    
    if(same_melody_exist == False):
        return"Label Not Exist!!"
    
    
    labs = get_labels(DATA_PATH)[0]
    expect_id = np.argmax(pred_rslts)
    return labs[expect_id]


#model = keras.models.load_model('ASR.h5')
result = predict('./SampleToBeTest/elise.wav', model)
#print(result)

print( expectLab('./SampleToBeTest/elise.wav', model))
#labels = getLabel(PATH_GARBAGE_CAR)


#print(EvaluateResults)
    
    
#print(predict('./SampleToBeTest/20221004.wav', model))
