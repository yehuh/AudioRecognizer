# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:25:13 2022

@author: yehuh
"""

from tensorflow import keras
from preprocess import *

import os
import numpy as np

feature_dim_1 = 20
feature_dim_2 = 11
channel = 1

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


model = keras.models.load_model('ASR.h5')

print(expectLab('./SampleToBeTest/Maid.wav', model))