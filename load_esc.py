# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:17:04 2020

@author: david
"""
"""
Load the esc wav files, degrade, and save the degraded mfcc frames(csv) or full wav clips(wav) 
(controlled by extract_mfcc).

Options:
if extract_mfcc: Load esc dataset, extract mfcc features, process with the privacy methods (fold level), 
and save the labeled feature array as csv

if not extract_mfcc: Load esc dataset, process with the privacy methods (file level), 
and save the reconstructed wav files


loaded file name rules: fold + index in freesound + dataset marks + class label

"""

import pandas as pd
import numpy as np
import librosa
import csv
from shutil import copyfile as cp
import os

import wav_read
import check_dirs

#%%


root = './ESC-dataset/ESC-50-master/ESC-50-master/'   # root to load data and meta table
option = 7
save_mfcc_dir = './esc_features/mfcc/' + 'option' + str(option) + '-0.5/'   # dir to save mfcc features, only used if extract_mfcc == True
save_degraded_segment_dir = './processed_audio/option' + str(option) + '-0.9/' # dir to save degraded audio segments, only used if extract_mfcc == False
extract_mfcc = False

new_sampling_rate=16000   # Hz
window_length_secs=0.05   # sec    # 0.08 for mfcc, 0.05 for spectrogram
hop_length_secs = window_length_secs #   sec
window_length = int(window_length_secs * new_sampling_rate)   # in samples
hop_length = int(hop_length_secs * new_sampling_rate)
mfcc = 19

# Load data
meta = pd.read_csv(root + 'meta/esc50.csv')
folds = meta['fold'].drop_duplicates().values 
#target = ['crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps',
 #         'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping']
target = ['helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train',
          'church_bells', 'airplane', 'fireworks', 'hand_saw']

for fold in folds:
    training_data, test_data = np.empty((0, mfcc + 1)), np.empty((0, mfcc + 1))   # fold-based data (+1 to add labels)
    for target_class in target:
        print(target_class)
        # get series of files for one class
        file_name_training = meta[(meta['fold'] != fold) & (meta['category'] == target_class)]['filename']   
        file_name_test = meta[(meta['fold'] == fold) & (meta['category'] == target_class)]['filename']   
        file_list_training = file_name_training.tolist()
        file_list_test = file_name_test.tolist()
        
        # for a given class, transform all its training audio files to mfcc(np) or degraded clips(wav)
        for file in file_list_training:
            #print('loading for training:', file, 'fold: ', fold)

            #rate, scaled_data = wav_read.read_audio_data(root + '/audio/' + file)
            #audio_data_resampled = wav_read.audio_pre_processing(scaled_data, rate, new_sampling_rate)
            audio_data_resampled, sr = librosa.core.load(root + '/audio/' + file)
            if not extract_mfcc:
                audio_frames = wav_read.framing(audio_data_resampled, 
                                                window_length, 
                                                hop_length)
                print('processing audio: %s fold: %d option: %d' % (file, fold, option))
                frames_processed = wav_read.frames_processing(audio_frames, option)
                re_signals = wav_read.reconstruct_time_series(frames_processed, hop_length)
                save_degraded_segment_dir_fold = save_degraded_segment_dir + 'fold' + str(fold)
                check_dirs.check_dir(save_degraded_segment_dir_fold)
                wav_read.write_audio_data(save_degraded_segment_dir_fold + \
                                          '/' + str(target.index(target_class)) + '_training_' + file + '.wav', 
                                          new_sampling_rate, 
                                          re_signals)
            if extract_mfcc:
                mfcc_feat = librosa.feature.mfcc(audio_data_resampled, sr=new_sampling_rate, 
                                                 win_length=window_length, hop_length=hop_length, 
                                                 n_mfcc=mfcc).T
                # add target label to the feature array
                training_label = np.asarray([target.index(target_class)] * len(mfcc_feat))
                mfcc_feat = np.hstack((mfcc_feat, training_label.reshape(len(training_label), 1)))
                # stack fold data (with labels)
                training_data = np.vstack((training_data, mfcc_feat))
            
        # for a given class, transform all its test audio files to mfcc(np) or degraded clips(wav)    
        for file in file_list_test:
            #print('loading for test:', file, 'fold: ', fold)
            #cp(root + '/audio/' + file, './check/' + file)
            #rate2, scaled_data2 = wav_read.read_audio_data(root + '/audio/' + file)
            #audio_data_resampled2 = wav_read.audio_pre_processing(scaled_data2, rate2, new_sampling_rate)
            audio_data_resampled2, sr = librosa.core.load(root + '/audio/' + file)
            if not extract_mfcc:
                audio_frames2 = wav_read.framing(audio_data_resampled2, 
                                                window_length, 
                                                hop_length)
                print('processing audio: %s fold: %d option: %d' % (file, fold, option))
                frames_processed2 = wav_read.frames_processing(audio_frames2, option)
                re_signals2 = wav_read.reconstruct_time_series(frames_processed2, hop_length)
                wav_read.write_audio_data(save_degraded_segment_dir_fold + \
                                          '/' + str(target.index(target_class)) + '_test_' + file + '.wav', 
                                          new_sampling_rate, 
                                          re_signals2)
            if extract_mfcc:
                mfcc_feat2 = librosa.feature.mfcc(audio_data_resampled2, sr=new_sampling_rate, 
                                                 win_length=window_length, hop_length=hop_length, 
                                                 n_mfcc=mfcc).T
                 # add target label to to the feature array
                test_label2 = np.asarray([target.index(target_class)] * len(mfcc_feat2))
                mfcc_feat2 = np.hstack((mfcc_feat2, test_label2.reshape(len(test_label2), 1)))
                # stack fold data (with labels)
                test_data = np.vstack((test_data, mfcc_feat2))
       
       
    # degrade mfcc features and save as fold-level training/test files in csv
    if extract_mfcc:
        print('processing mfcc for fold: %d' % fold)
        frames_training_processed = wav_read.frames_processing(training_data, option)
        frames_test_processed = wav_read.frames_processing(test_data, option)    
        # save mfcc training set
        print('saving mfcc features and labels for fold:', fold)
        save_mfcc_dir_fold = save_mfcc_dir + 'fold' + str(fold)
        check_dirs.check_dir(save_mfcc_dir_fold)
        with open(os.path.join(save_mfcc_dir_fold, 'training.csv'), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerows(frames_training_processed)
            csvfile.close()
            print('fold %d mfcc saved.' % fold)
        # save mfcc test set
        print('saving mfcc features and labels for fold:', fold)
        check_dirs.check_dir(save_mfcc_dir_fold)
        with open(os.path.join(save_mfcc_dir_fold, 'test.csv'), 'w') as csvfile2:
            spamwriter = csv.writer(csvfile2, delimiter=',')
            spamwriter.writerows(frames_test_processed)
            csvfile2.close()
            print('fold %d mfcc saved.' % fold)
            
            
            
        
