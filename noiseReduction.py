# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:51:56 2022

@author: yehuh
"""
from scipy.io import wavfile
import noisereduce as nr
# load data
rate, data = wavfile.read("./SampleToBeTest/20221010.wav")

noise_rate, noise_data = wavfile.read("./SampleToBeTest/noise.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate, y_noise=noise_data,prop_decrease = 0.8)
wavfile.write("reduced_noise.wav", rate, reduced_noise)
