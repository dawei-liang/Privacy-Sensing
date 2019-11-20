# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 21:54:06 2019

@author: david
"""

"""
Script to load, test and save a clip of audio
""" 

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy import signal
from statsmodels.tsa.stattools import acf
import librosa

import wav_read
import pca

#%%   
"""
load data, process and save
"""
new_sampling_rate=16000   # Resample to 16kHz (VGGish: 8kHz 1D-2D, 16KHz 2D-3D).
window_length_secs=0.06
hop_length_secs = window_length_secs # 
file = './mturk/scenarios/couple/couple.wav'   # path to load audio
filename1, filename2 = './mturk/scenarios/original.wav', './mturk/scenarios/processed.wav' # path to save audio
option = 4   # audio processing methods

window_length_samples = int(round(new_sampling_rate * window_length_secs))
hop_length_samples = int(round(new_sampling_rate * hop_length_secs))
fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

file_sampling_rate, scaled_data = wav_read.read_audio_data(file)
print('audio file length for check: %d sec' %(scaled_data.shape[0]//file_sampling_rate))
# audio channel, resampling
data = wav_read.audio_pre_processing(data=scaled_data, 
                            sr=file_sampling_rate, 
                            sr_new=new_sampling_rate)
# frame time-series signals
frames = wav_read.framing(data, hop_length=hop_length_samples, window_length=window_length_samples)
# privacy processing
start = time.time()
frames_new = wav_read.frames_processing(frames, option = option)
print("execution time : {}".format(time.time()-start))

# saving
processed_audio_signal = wav_read.reconstruct_time_series(frames_new, hop_length_samples)
wav_read.write_audio_data(filename1, new_sampling_rate, data)   # save raw signals
wav_read.write_audio_data(filename2, new_sampling_rate, processed_audio_signal)   # save processed signals

#%%
"""
PCA transform for privacy protection
"""
start = time.time()
# calculate spectrogram, all data points in each frame are used for fft
spectrogram_for_PCA = np.abs(np.fft.rfft(frames, frames.shape[1]))
# PCA transform and reconstruction
spectrogram_PCAed = pca.pca_transform(spectrogram_for_PCA, components = 10)
windowed_frames_PCAed = np.fft.irfft(spectrogram_PCAed, n = frames.shape[1], axis=-1)
print("execution time2 : {}".format(time.time()-start))

# reconstruct PCAed frames back to time domain
pca_audio_signal = wav_read.reconstruct_time_series(windowed_frames_PCAed, hop_length_samples)
#wav_read.write_audio_data('./mturk/scenarios/dinner/pca.wav', new_sampling_rate, pca_audio_signal)   # save PCAed signals


#%%
"""
plot spectrogram
"""
# create window func
window = wav_read.periodic_hann(window_length_samples)
# apply window to the frames, not used in fft
windowed_frames_original = frames * window   # origianl
windowed_frames_new = frames_new * window   # processed
# calculate spectrogram, all data points in each frame are used for fft
spectrogram_original = np.abs(np.fft.rfft(windowed_frames_original, int(fft_length)))
spectrogram_new = np.abs(np.fft.rfft(windowed_frames_new, int(fft_length)))

# plot sample spectrogram
#plt.figure()
#plt.plot(frames[10], label='1')
#plt.plot(frames_new[10], label='4')
#plt.xlabel('Time [62.5 uSec]')
#plt.legend(loc='upper left')
#plt.show(block = True)
#
plt.figure(1)
plt.pcolormesh(10 * np.log10(spectrogram_original.T[:,50:150]))
plt.ylabel('FFT')
plt.xlabel('0:50 Frames')
plt.show()
#
#plt.figure()
#plt.pcolormesh(10 * np.log10(spectrogram_new.T[:,0:50]))
#plt.ylabel('FFT')
#plt.xlabel('0:50 Frames')
#plt.show()


#plt.figure()
#
#freqs, times, Sx = signal.spectrogram(data, fs=new_sampling_rate, window='hanning',
#                                      nperseg=int(fft_length), noverlap=0,
#                                      detrend=False, scaling='spectrum')
#
#plt.pcolormesh(times[:50], freqs, 10 * np.log10(Sx[:,:50]), cmap='viridis')
#plt.ylabel('HZ')
#plt.xlabel('Time')
#plt.show()
#
#plt.figure()
#
#freqs, times, Sx = signal.spectrogram(processed_audio_signal, fs=new_sampling_rate, window='hanning',
#                                      nperseg=int(fft_length), noverlap=0,
#                                      detrend=False, scaling='spectrum')
#
#plt.pcolormesh(times[:50], freqs, 10 * np.log10(Sx[:,:50]), cmap='viridis')
#plt.ylabel('HZ')
#plt.xlabel('Time')
#plt.show()

#%%
'''temp, testing inverse FFT'''
#spectrogram2 = np.fft.rfft([1,0,0,0,1])
#spectrogram2_flip = np.flip(spectrogram2, axis=0)
#spectrogram2_flip = -spectrogram2
#windowed_frames2 = np.fft.irfft(spectrogram2_flip, 5)
#print(windowed_frames2)

# couple chatting in bedroom, home party, office talking, sexual acivities, answer phone at home


