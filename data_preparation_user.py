# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:00:54 2019

@author: david
"""

"""
Load user audio and save original/processed data as individual activity class (class-based).
"""

import os
import numpy as np
from scipy.io import wavfile
import csv

import wav_read
import check_dirs

#%%
classes = ['bathing',
          'flushing',
          'brushing',
          'shaver',
          'frying',
          'chopping',
          'micro',
          'boiling', 
          'blender',
          'TV', 
          'piano', 
          'vacuum',  
          'washing',
          'chatting', 
          'strolling']
#classes=['washing','chatting','strolling']
markname = './processed_user_data/6/0.9/dropping_'   # path to save
option = 6

rootdir = "./scripted study/"   # dir to load audio clips
new_sampling_rate=16000   # Hz
window_length_secs=0.06   # sec
hop_length_secs = window_length_secs #   sec

#%% 
def list_files(wav_list, target_list, rootdir, target): 
    """
    Return a list of target wave path and the file labels.
    Args:
        wav, labels: empty lists of wave files and activity labels
        rootdir: directory to loop for the files
        target: target activity labels
    Return:
        lists of wave files and labels
    """
    for subdir, dirs, files in os.walk(rootdir):
        for name in files:
            if target in name and name.endswith('.wav'):
                wav_list.append(os.path.join(subdir, name))
                target_list.append(target)
    return wav_list, target_list

#%%
def save_audio_data(data, labels, save_path):
    """
    save audio signals as CSV, not actually used
    """
    with open(save_path + 'processed-audio'+'.csv','w') as f:
        wr = csv.writer(f,lineterminator='\n', delimiter=',')
        wr.writerows((data, labels))
    f.close()
    
#%%
"""
loading
"""

# Calculate window parameters
window_length_samples = int(round(new_sampling_rate * window_length_secs))
hop_length_samples = int(round(new_sampling_rate * hop_length_secs))
print('window_length_samples: %d, hop_length_samples: %d' %(window_length_samples, hop_length_samples))

wav_list = []   # list of wav path
target_list = []   # list of target class names
for k in range(len(classes)):
    target = classes[k]  
    # Return all target file dirs of class k
    wav_list, target_list = list_files(wav_list, target_list, rootdir, target)  
print('check if target file and label size are the same: %d, %d' %(len(wav_list), len(target_list)))

audio_data = []    
audio_labels = []
# Load wave data and assign labels
for i, file in enumerate(wav_list):
    file_sampling_rate, scaled_data = wav_read.read_audio_data(file=file)
    data = wav_read.audio_pre_processing(data=scaled_data, 
                                         sr=file_sampling_rate, 
                                         sr_new=new_sampling_rate)
    audio_data.append(data)
    audio_labels.append(target_list[i])
    if i % 10 == 0:
        print('loading and formatting/re-sampling %d out of %d files' %(i+1,len(wav_list)))
print('check if loaded data and label size are the same: %d, %d' %(len(audio_data), len(audio_labels)))
    
#%%
"""
processing and saving
"""

processed_audio_signal = []
check_dirs.check_dir(markname.replace(markname.split('/')[-1], ''))

# Process with proposed methods and reconstruct as time-series signals
for i in range(len(audio_data)):
    # frame time-series signals
    frames = wav_read.framing(audio_data[i], 
                              hop_length=hop_length_samples, 
                              window_length=window_length_samples)
    # privacy processing
    print('processing file: %d out of %d' % (i+1, len(audio_data)))
    frames_new = wav_read.frames_processing(frames, option = option)
    # save all frame arrays in one list
    processed_audio_signal.append(wav_read.reconstruct_time_series(frames_new, hop_length_samples))
print('check if reconstructed signal and label size are the same: %d, %d' %(len(processed_audio_signal), len(audio_labels)))

# Loop for each label
for class_label in classes:
    # reset data container
    class_data_processed = np.empty((0,))
    for i in range(len(audio_labels)):
        # match between class labels and file labels
        if class_label == audio_labels[i]:
            class_data_processed = np.hstack((class_data_processed, processed_audio_signal[i]))
    # check if a class is missing
    if class_data_processed.shape[0]<1:
        continue
    print('saving audio for label:', class_label)
    wav_read.write_audio_data(filename=markname + class_label + '.wav', 
                              rate=new_sampling_rate, 
                              wav_data=class_data_processed)   # save processed signals   