#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:47:02 2019

@author: dawei
"""

"""
Load audio from individual class form, split (5-fold) and concatenate it as fold-based sequences. 
Save privacy processed sequences in the format of mfcc features with labels.

Audio is loaded from original (method 0) class-based data. Sicne it has been pre-processed by data_preparation_user.py, 
there is no need to use wave_rad.audio_pre_processing again.
"""

from numpy.random import seed
import os
import numpy as np
from python_speech_features import mfcc
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
# Fix random seed
#seed(0)


#%%
"""
data loading 
"""  
dir_load_audio = './processed_user_data_class-based/0/'   
save_data_dir = './processed_user_data_fold-based/neighbouring_range_test/1000/test3/'
new_sampling_rate=16000   # Hz
window_length_secs=0.06   # sec
hop_length_secs = window_length_secs #   sec
mfcc_size = 13
option = 4
process_both_training_validation = True

# load data and labels
audio_file_list = [x for x in os.listdir(dir_load_audio) if x.endswith('.wav')]   # Load wave files
audio_data, labels = [], []
for class_item in classes:
    print('loading class ', class_item)
    labels.append(class_item)   # activity classes in use
    for file_item in audio_file_list:
        if class_item in file_item:
            # load and scale the audio
            path = os.path.join(dir_load_audio, file_item)
            file_sampling_rate, scaled_data = wav_read.read_audio_data(file=path)
            assert file_sampling_rate == new_sampling_rate, 'loaded sampling rate is not the same as new_sampling_rate'
            audio_data.append(scaled_data)
            
print('number of loaded audio classes and labels: ', len(audio_data), len(labels))

#%%
""" 
feature extraction and cross-validation split
"""
# Define subsets and labels
full_data, full_labels = np.empty((0,mfcc_size)), []
training_data1, training_labels1 = np.empty((0,mfcc_size)), []
training_data2, training_labels2 = np.empty((0,mfcc_size)), []
training_data3, training_labels3 = np.empty((0,mfcc_size)), []
training_data4, training_labels4 = np.empty((0,mfcc_size)), []
training_data5, training_labels5 = np.empty((0,mfcc_size)), []
valid_data1, valid_labels1 = np.empty((0,mfcc_size)), []
valid_data2, valid_labels2 = np.empty((0,mfcc_size)), []
valid_data3, valid_labels3 = np.empty((0,mfcc_size)), []
valid_data4, valid_labels4 = np.empty((0,mfcc_size)), []
valid_data5, valid_labels5 = np.empty((0,mfcc_size)), []

# loop for each activity class k
for label_i in range(len(labels)):   
    mfcc_feat = mfcc(signal=audio_data[label_i], 
                     samplerate=file_sampling_rate,
                     winlen=window_length_secs,
                     winstep=hop_length_secs,
                     numcep=mfcc_size,
                     nfilt=26,
                     nfft=960)   # mfcc of each class, shape: [class size, mfcc size]
    # full data is used to check original vs shuffling, not used for fold split!
    full_data = np.vstack((full_data, mfcc_feat))
    full_labels.append([label_i] * mfcc_feat.shape[0])
    # split 5 folds for each class of feature sets
    n1 = len(mfcc_feat)*1//5
    n2 = len(mfcc_feat)*2//5
    n3 = len(mfcc_feat)*3//5
    n4 = len(mfcc_feat)*4//5
    n = len(mfcc_feat)
    # 1st fold
    training_data1 = np.vstack((training_data1, mfcc_feat[0:n4]))
    training_labels1.append([label_i] * n4)   # list of labels of the features
    valid_data1 = np.vstack((valid_data1, mfcc_feat[n4:]))
    valid_labels1.append([label_i] * (n - n4))   
    # 2nd fold
    training_data2 = np.vstack((training_data2, mfcc_feat[0:n3]))
    training_data2 = np.vstack((training_data2, mfcc_feat[n4:]))
    training_labels2.append([label_i] * (n3 + n - n4)) 
    valid_data2 = np.vstack((valid_data2, mfcc_feat[n3:n4]))
    valid_labels2.append([label_i] * (n4 - n3)) 
    # 3rd fold
    training_data3 = np.vstack((training_data3, mfcc_feat[0:n2]))
    training_data3 = np.vstack((training_data3, mfcc_feat[n3:]))
    training_labels3.append([label_i] * (n2 + n - n3)) 
    valid_data3 = np.vstack((valid_data3, mfcc_feat[n2:n3]))
    valid_labels3.append([label_i] * (n3 - n2))
    # 4th fold
    training_data4 = np.vstack((training_data4, mfcc_feat[0:n1]))
    training_data4 = np.vstack((training_data4, mfcc_feat[n2:]))
    training_labels4.append([label_i] * (n1 + n - n2)) 
    valid_data4 = np.vstack((valid_data4, mfcc_feat[n1:n2]))
    valid_labels4.append([label_i] * (n2 - n1))
    # 5th fold
    training_data5 = np.vstack((training_data5, mfcc_feat[n1:]))
    training_labels5.append([label_i] * (n - n1))  
    valid_data5 = np.vstack((valid_data5, mfcc_feat[0:n1]))
    valid_labels5.append([label_i] * n1)
# flatten lists of labels
flattened_full_labels = [item for sublist in full_labels for item in sublist]
flattened_training_labels1 = [item for sublist in training_labels1 for item in sublist]
flattened_valid_labels1 = [item for sublist in valid_labels1 for item in sublist]
flattened_training_labels2 = [item for sublist in training_labels2 for item in sublist]
flattened_valid_labels2 = [item for sublist in valid_labels2 for item in sublist]
flattened_training_labels3 = [item for sublist in training_labels3 for item in sublist]
flattened_valid_labels3 = [item for sublist in valid_labels3 for item in sublist]
flattened_training_labels4 = [item for sublist in training_labels4 for item in sublist]
flattened_valid_labels4 = [item for sublist in valid_labels4 for item in sublist]
flattened_training_labels5 = [item for sublist in training_labels5 for item in sublist]
flattened_valid_labels5 = [item for sublist in valid_labels5 for item in sublist]

# concatenate data and labels for each fold, shape: [# of frmaes, (size of mfcc + 1)]
training_data_fold1 = np.hstack((training_data1, np.asarray(flattened_training_labels1).reshape((len(flattened_training_labels1),1))))
training_data_fold2 = np.hstack((training_data2, np.asarray(flattened_training_labels2).reshape((len(flattened_training_labels2),1))))
training_data_fold3 = np.hstack((training_data3, np.asarray(flattened_training_labels3).reshape((len(flattened_training_labels3),1))))
training_data_fold4 = np.hstack((training_data4, np.asarray(flattened_training_labels4).reshape((len(flattened_training_labels4),1))))
training_data_fold5 = np.hstack((training_data5, np.asarray(flattened_training_labels5).reshape((len(flattened_training_labels5),1))))
valid_data_fold1 = np.hstack((valid_data1, np.asarray(flattened_valid_labels1).reshape((len(flattened_valid_labels1),1))))
valid_data_fold2 = np.hstack((valid_data2, np.asarray(flattened_valid_labels2).reshape((len(flattened_valid_labels2),1))))
valid_data_fold3 = np.hstack((valid_data3, np.asarray(flattened_valid_labels3).reshape((len(flattened_valid_labels3),1))))
valid_data_fold4 = np.hstack((valid_data4, np.asarray(flattened_valid_labels4).reshape((len(flattened_valid_labels4),1))))
valid_data_fold5 = np.hstack((valid_data5, np.asarray(flattened_valid_labels5).reshape((len(flattened_valid_labels5),1))))

#%%
"""
frames processing with the privacy method
"""
# Calculate window parameters
window_length_samples = int(round(new_sampling_rate * window_length_secs))
hop_length_samples = int(round(new_sampling_rate * hop_length_secs))
print('window_length_samples: %d, hop_length_samples: %d' %(window_length_samples, hop_length_samples))
list_of_folded_data = [training_data_fold1, 
                      training_data_fold2, 
                      training_data_fold3, 
                      training_data_fold4, 
                      training_data_fold5,
                      valid_data_fold1,
                      valid_data_fold2,
                      valid_data_fold3,
                      valid_data_fold4,
                      valid_data_fold5]   # sequence of folds
# Process the mfcc frames with proposed methods (no need to frame mfcc frames again!)
list_of_processed_mfcc = []
# Loop for each fold
for i in range(len(list_of_folded_data)):
    print('processing fold: %d out of %d' % (i+1, len(list_of_folded_data)))
    # If process training folds only, then all validatioin folds will remain unchanged (method 0)
    if not process_both_training_validation and i > 4:
        print('process training folds only, valid fold d% unchanged' %(i-4))
        frames_new = wav_read.frames_processing(list_of_folded_data[i], option = 0)
    else:
        frames_new = wav_read.frames_processing(list_of_folded_data[i], option = option)
    # save all frame arrays in one list
    list_of_processed_mfcc.append(frames_new)
print('check if reconstructed signal and label size are the same: %d, %d' %(len(list_of_processed_mfcc), len(list_of_folded_data)))

#%%
"""
save folds
"""
names = ['training_fold1',
         'training_fold2',
         'training_fold3',
         'training_fold4',
         'training_fold5',
         'validation_fold1',
         'validation_fold2',
         'validation_fold3',
         'validation_fold4',
         'validation_fold5']
check_dirs.check_dir(save_data_dir)
for i in range(len(list_of_processed_mfcc)):
    print('saving mfcc features and labels for fold:', i)
    with open(save_data_dir + names[i] + '.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(list_of_processed_mfcc[i])
        csvfile.close()
print('folded mfcc saved.')
