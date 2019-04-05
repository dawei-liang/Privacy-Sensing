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

import matplotlib.pyplot as plt
from scipy import signal

import pandas as pd

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
    frames.shape = # of frames * window size
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
    
file = 'G://Research1/codes/audioset/test_data/summer_2018_freesound/scripted study/zhao/0/bathing.wav'
rate, scaled_data = read_audio_data(file)

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
#frames_flip = np.flip(frames,axis=1)  # horizontal flip
frames_flip = - frames   # vertical flip

window = periodic_hann(window_length_samples)
windowed_frames = frames * window
windowed_frames_flipped = frames_flip * window

spectrogram = np.abs(np.fft.rfft(windowed_frames, int(fft_length)))
spectrogram_flipped = np.abs(np.fft.rfft(windowed_frames_flipped, int(fft_length)))

#%%
plt.figure()
plt.subplot()
plt.plot(frames[0])
plt.xlabel('Time [Sec]')
plt.show(block = True)
plt.subplot()
plt.plot(frames_flip[0])
plt.xlabel('Time [Sec]')
plt.show(block = True)

plt.figure()
plt.pcolormesh(spectrogram.T[:,0:1000])
plt.ylabel('FFT coe')
plt.xlabel('Frames')
plt.show()

#%%

new_signal = reconstruct_time_series(frames, hop_length_samples)
filename1, filename2, rate, wav_data='./speech.wav', './flipped_speech.wav', 16000, new_signal
write_audio_data(filename1, rate, data)
write_audio_data(filename2, rate, wav_data)
