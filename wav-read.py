#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:26:20 2019

@author: dawei
"""

from scipy.io import wavfile
import os
import numpy as np
import resampy
import copy as cp

import matplotlib.pyplot as plt
from scipy import signal

import pandas as pd

np.random.seed(0)

#%%
#os.chdir('/home/dawei/research4')   # Set main dir
#print(os.path.abspath(os.path.curdir))


def read_audio_data(file):
    rate, wav_data = wavfile.read(file)
    assert wav_data.dtype == np.int16, 'Not support: %r' % wav_data.dtype  # check input audio rate(int16)
    scaled_data = wav_data / 32768.0   # 16bit
    return rate, scaled_data

#%%

def write_audio_data(filename, rate, wav_data):
    wav_data = wav_data * 32768.0   # 16bit
    wav_data = wav_data.astype(np.int16)
    wavfile.write(filename, rate, wav_data)
    print('Saved')


#%%
def frame(data, window_length, hop_length):
  """Convert array into a sequence of successive possibly overlapping frames.
  An n-dimensional array of shape (num_samples, ...) is converted into an
  (n+1)-D array of shape (num_frames, window_length, ...), where each frame
  starts hop_length points after the preceding one.
  This is accomplished using stride_tricks, so the original data is not
  copied.  However, there is no zero-padding, so any incomplete frames at the
  end are not included.
  Args:
    data: np.array of dimension N >= 1.
    window_length: Number of samples in each frame.
    hop_length: Advance (in samples) between each window.
  Returns:
    (N+1)-D np.array with as many rows as there are complete frames that can be
    extracted.
  """
  num_samples = data.shape[0]
  num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
  shape = (num_frames, window_length) + data.shape[1:]
  ''' data.strides: # of bytes to skip to next value; 
      data.strides * hop_length = # of bytes to skip to next window'''
  strides = (data.strides[0] * hop_length,) + data.strides 
  # shape parameter: frame shape(# of frames * window size)
  return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

#%%

def reconstruct_time_series(frames, hop_length_samples):
    '''Reconstruct frames back to the original signals
    frames = [# of frames/windows, samples within the window]
    hop_length_samples = # of samples skipped during framing
    '''
    new_signal = []
    for i in range(len(frames)-1):
        for j in range(0, hop_length_samples):
            new_signal.append(frames[i, j])
    # Last frame
    for i in range(frames.shape[1]):
        new_signal.append(frames[-1,i])
        
    new_signal = np.asarray(new_signal)
    
    return new_signal
    
#%%

def periodic_hann(window_length):
  """Calculate a "periodic" Hann window.
  The classic Hann window is defined as a raised cosine that starts and
  ends on zero, and where every value appears twice, except the middle
  point for an odd-length window.  Matlab calls this a "symmetric" window
  and np.hanning() returns it.  However, for Fourier analysis, this
  actually represents just over one cycle of a period N-1 cosine, and
  thus is not compactly expressed on a length-N Fourier basis.  Instead,
  it's better to use a raised cosine that ends just before the final
  zero value - i.e. a complete cycle of a period-N cosine.  Matlab
  calls this a "periodic" window. This routine calculates it.
  Args:
    window_length: The number of points in the returned window.
  Returns:
    A 1D np.array containing the periodic hann window.
  """
  return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                             np.arange(window_length)))
  
#%%

def frames_processing(frames, option):
    '''0 horizontal flip within each frame, 
    1 vertial flip within each frame, 
    2 resampling, 
    3 interpolation and replacement,
    4 randomly flip frame orders'''
    
    if option == 0:
        frames_new = np.flip(frames,axis=1)
    elif option == 1:  
        frames_new = - frames
    elif option == 2:        
        frames_new = resampy.resample(frames, 16000, 8000)   # original sr, new sr
    elif option == 3:
        frames_new = cp.deepcopy(frames)
        percentage = 0.5   # % to be replaced
        idx = np.random.randint(low=50, high=frames.shape[0]-50, 
                                size=(round(percentage*frames.shape[0]), 1))
        for i in idx:
            neighbour = np.random.randint(low=i-50, high=i+50)
            frames_new[i] = frames[neighbour]   # replaced by neighbourhood
        print('new frames shape:', frames_new.shape)
    elif option ==4:
        frames_new = cp.deepcopy(frames)
        np.random.shuffle(frames_new)   # only 1st axis is shuffled

    return frames_new

#%%
    
file = './bathing.wav'
rate, scaled_data = read_audio_data(file)
option = 3   # process methods

# Convert to mono.
if scaled_data.shape[1] > 1:
  data = np.mean(scaled_data, axis=1)
# Resample to the rate assumed by VGGish.
if rate != 16000:
  data = resampy.resample(data, rate, 16000)

audio_sample_rate=8000
window_length_secs=0.025
hop_length_secs=0.010

window_length_samples = int(round(audio_sample_rate * window_length_secs))
hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
  
frames = frame(data, hop_length=hop_length_samples, window_length=window_length_samples)
frames_new = frames_processing(frames, option = option)
# create window func
window = periodic_hann(window_length_samples)
windowed_frames = frames * window
windowed_frames_new = frames_new * window
# calculate spectrogram
spectrogram = np.abs(np.fft.rfft(windowed_frames, int(fft_length)))
spectrogram_new = np.abs(np.fft.rfft(windowed_frames_new, int(fft_length)))

#%%
plt.figure()
plt.plot(frames[1000], label='1')
plt.plot(frames_new[1000], label='4')
plt.xlabel('Time [62.5 uSec]')
plt.legend(loc='upper left')
plt.show(block = True)

plt.figure()
plt.pcolormesh(spectrogram.T[:,0:50])
plt.ylabel('FFT')
plt.xlabel('0:50 Frames')
plt.show()

plt.figure()
plt.pcolormesh(spectrogram_new.T[:,0:50])
plt.ylabel('FFT')
plt.xlabel('0:50 Frames')
plt.show()

#%%

processed_audio_signal = reconstruct_time_series(frames_new, hop_length_samples)
filename1, filename2, rate, wav_data='./original.wav', './processed.wav', 16000, processed_audio_signal
write_audio_data(filename1, rate, data)   # save raw signals
write_audio_data(filename2, rate, processed_audio_signal)   # save processed signals

#%%
'''temp, testing inverse FFT'''
spectrogram2 = np.fft.rfft([1,0,0,0,1])
spectrogram2_flip = np.flip(spectrogram2, axis=0)
#spectrogram2_flip = -spectrogram2
windowed_frames2 = np.fft.irfft(spectrogram2_flip, 5)
print(windowed_frames2)
