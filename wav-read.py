#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:26:20 2019

@author: dawei
"""

from scipy.io import wavfile
import os
import numpy as np

import matplotlib.pyplot as plt

os.chdir('/home/dawei/research4')   # Set main dir
print(os.path.abspath(os.path.curdir))


def read_audio_data(file):
    rate, wav_data = wavfile.read(file)
    assert wav_data.dtype == np.int16, 'Not support: %r' % wav_data.dtype
    scaled_data = wav_data / 32768.0   # 16bit
    return scaled_data
    
file = '0001_T1_HSS.wav'
scaled_data = read_audio_data(file)

plt.plot(scaled_data)
plt.show(block = True)