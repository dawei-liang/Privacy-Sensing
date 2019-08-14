#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:26:20 2019
Partially adopted from Google VGGish Github

@author: dawei
"""

"""
audio IO and privacy processing
"""

from scipy.io import wavfile
import os
import numpy as np
import resampy
import copy as cp

np.random.seed(0)
#os.chdir('/home/dawei/research4')   # Set main dir
#print(os.path.abspath(os.path.curdir))

#%%
def read_audio_data(file):
    '''read audio, only support 16-bit depth'''
    rate, wav_data = wavfile.read(file)
    assert wav_data.dtype == np.int16, 'Not support: %r' % wav_data.dtype  # check input audio rate(int16)
    scaled_data = wav_data / 32768.0   # 16bit standardization
    return rate, scaled_data

#%%
def write_audio_data(filename, rate, wav_data):
    '''write normalized audio signals with 16 bit depth to a wave file'''
    wav_data = wav_data * 32768.0   # 16bit
    wav_data = wav_data.astype(np.int16)
    wavfile.write(filename, rate, wav_data)
    print(filename + ' Saved')

#%%
def audio_pre_processing(data, sr, sr_new):
    # Convert to mono.
    try:
        if data.shape[1] > 1:
            data = np.mean(data, axis=1)
    except:
        pass
    # Resampling the data to specified rate
    if sr != sr_new:
      data = resampy.resample(data, sr, sr_new)
    return data

#%%
def framing(data, window_length, hop_length):
    """
    Convert 1D time series signals or N-Dimensional frames into a (N+1)-Dimensional array of frames.
    No zero padding, rounding at the end.
    Args:
        data: Input signals.
        window_length: Number of samples in each frame.
        hop_length: Advance (in samples) between each window.
    Returns:
        np.array with as many rows as there are complete frames that can be extracted.
    """
    
    num_samples = data.shape[0]
    frame_array = data[0:window_length]
    # create a new axis as # of frames
    frame_array = frame_array[np.newaxis]  
    start = hop_length
    for _ in range(num_samples):
        end = start + window_length
        if end <= num_samples:
            # framing at the 1st axis
            frame_temp = data[start:end]
            frame_temp = frame_temp[np.newaxis]
            frame_array = np.concatenate((frame_array, frame_temp), axis=0)
        start += hop_length
    return frame_array


#%%

def reconstruct_time_series(frames, hop_length_samples):
    """
    Reconstruct N-Dimensional framed array back to (N-1)-Dimensional frames or 1D time series signals
    Args:
        frames = [# of frames, window length1 in samples, (window length2, ...)]
        hop_length_samples = # of samples skipped between two frames
    return:
        (N-1)-Dimensional frames or 1D time series signals
    """
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
    """
    0 no processing
    1 horizontal flip within each frame, 
    2 vertial flip within each frame, 
    3 frequency resampling, 
    4 interpolation and replacement,
    5 order shuffling
    6 sub sampling
    7 cloning
    
    args:
        frames: 2D array of audio frames
        option: processing mode
    return:
        new 2D array of processed audio frames
    """
    print('process mode:', option)
    if option == 0:
        frames_new = frames
        
    if option == 1:
        frames_new = np.flip(frames,axis=1)
        
    elif option == 2:  
        frames_new = - frames
        
    elif option == 3:        
        frames_new = resampy.resample(frames, 16000, 8000)   # original sr, new sr
        
    elif option == 4:
        frames_new = cp.deepcopy(frames)
        percentage = 0.7   # % of frames to be replaced
        neighbor_range = 50   # range of nearby frames, one-sided
        # select target frames
        frames_to_be_replaced = np.random.choice(np.arange(0, frames.shape[0]), 
                                          size=round(percentage*(frames.shape[0]-neighbor_range*2)),
                                          replace=False)
        print('frames to be replaced:', np.sort(frames_to_be_replaced), 'size:', len(frames_to_be_replaced))
        for i in frames_to_be_replaced:
            # select a nearby frame, including the target frame itself
            # include two edge cases
            if i <= neighbor_range:
                neighbour = np.random.choice(np.arange(0, i+neighbor_range))
            elif (i+neighbor_range) >= frames.shape[0]:
                neighbour = np.random.choice(np.arange(i-neighbor_range, frames.shape[0]-1))
            else:
                neighbour = np.random.choice(np.arange(i-neighbor_range, i+neighbor_range))   
            # replacement
            frames_new[i] = frames[neighbour]   
            
    elif option == 5:
        frames_new = cp.deepcopy(frames)
        np.random.shuffle(frames_new)   # only 1st axis (frame index) is shuffled
        
    elif option == 6:
        percentage = 0.7   # % of frames to drop
         # select target frames
        frames_to_drop = np.random.choice(frames.shape[0], 
                                          size=round(percentage*frames.shape[0]),
                                          replace=False)
        print('frames to drop:', np.sort(frames_to_drop), 'size:', len(frames_to_drop))
        # delete items by idx
        frames_new = np.delete(frames, frames_to_drop, axis=0)
        
    elif option == 7:
        percentage = 0.5   # % of frames to be copied
        frames_new = cp.deepcopy(frames)
        frames_new = frames_new.tolist()
        # select target frames
        frames_to_add = np.random.choice(frames.shape[0], 
                                         size=round(percentage*frames.shape[0]),
                                         replace=False)
        print('frames to be added:', np.sort(frames_to_add), 'size:', len(frames_to_add))
        # randomly copy and add frames to the new set
        for i in frames_to_add:
            index_to_insert = np.random.randint(0, len(frames_new)-1)
            frames_new.insert(index_to_insert, frames[i])
        frames_new = np.asarray(frames_new)

    print('new frames shape:', frames_new.shape)
    return frames_new
