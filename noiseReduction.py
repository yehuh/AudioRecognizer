# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:51:56 2022

@author: yehuh
"""
from scipy.io import wavfile
import noisereduce as nr
import os

# load data
sample_path = './SampleToBeTest/'

files_with_noise = []
allFileList = os.listdir(sample_path)
for file in allFileList:
    if "Elise" in file or "Maid" in file:
        file_buff = file.replace(".wav","")
        
        print(str(file_buff))
        
        files_with_noise.append(file_buff)


for noise_file in files_with_noise:
    rate, data = wavfile.read(sample_path + str(noise_file)+".wav")
    noise_rate, noise_data = wavfile.read("./SampleToBeTest/noise.wav")
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate, y_noise=noise_data,prop_decrease = 0.8)
    wavfile.write(sample_path+str(noise_file)+"NoiseReduction.wav", rate, reduced_noise)
