# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 23:30:41 2019

@author: david
"""
import os
from python_speech_features import mfcc
from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
import matplotlib.pyplot as plt

import keras as K
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import SGD, Adadelta
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

import wav_read

#%%
#classes = ['bathing',
#          'flushing',
#          'brushing',
#          'shaver',
#          'frying',
#          'chopping',
#          'micro',
#          'boiling', 
#          'blender',
#          'TV', 
#          'piano', 
#          'vacuum',  
#          'washing',
#          'chatting', 
#          'strolling']
classes=['bathing','flushing']
# Fix random seed
seed(0)
set_random_seed(0)

#%%
    
def cnn_model_fn(class_size):
        model = Sequential()
        model.add(Conv1D(32, 8, strides=1, activation='relu', padding="same", input_shape=(13,1)))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))
        model.add(Conv1D(64, 8, strides=1, activation='relu', padding="same"))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))
        model.add(Conv1D(64, 4, strides=1, activation='relu', padding="same"))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))
        print(model.output.shape)
        model.add(Flatten())
        model.add(Dense(200, activation ='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(class_size, activation='softmax'))
            
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        loss = categorical_crossentropy
        model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])
        return model
    
#%%
"""
Save model and architecture
"""
def save_model(clf, count, save_model_dir):
    clf.save(save_model_dir + str(count) + 'temp.hdf5')   # Save model
    yaml_string = clf.to_yaml()
    with open(save_model_dir + \
              str(count) + 'temp.yaml', 'w') as f:   # Save architecture
        f.write(yaml_string)
    f.close()
    
#%%
"""
one-hot encoding and training data reshaping
"""
def reshape_data_labels(data, labels, mfcc_size, class_size):
    # Reshape training data as (#,size_mfcc,1) for CNN
    data = np.reshape(data, (data.shape[0], mfcc_size, 1))   
    # One-hot encoding for training labels: (#,size_labels)
    labels = np_utils.to_categorical(labels, class_size)
    return data, labels
    
#%%
"""
data loading and feature extraction
"""
dir_load_audio = './processed_user_data/'
save_model_dir = './models/'
new_sampling_rate=16000   # Hz
window_length_secs=0.1   # sec
hop_length_secs = 0.05 #   sec
mfcc_size = 13

# load data and labels
audio_file_list = [x for x in os.listdir(dir_load_audio) if x.endswith('.wav')]   # Load wave files
audio_data_processed, audio_data_original, labels = [], [], []
for class_item in classes:
    print('loading class ', class_item)
    labels.append(class_item)   # activity classes in use
    for file_item in audio_file_list:
        if class_item in file_item:
            # load and scale the audio
            path = os.path.join(dir_load_audio, file_item)
            file_sampling_rate, scaled_data = wav_read.read_audio_data(file=path)
            assert file_sampling_rate == new_sampling_rate, 'loaded sampling rate is not the same as new_sampling_rate'
            # seperate original sounds and privacy processed sounds
            if 'original' in file_item:
                audio_data_original.append(scaled_data)
            elif 'processed' in file_item:
                audio_data_processed.append(scaled_data)
            
print('loaded original audio number, loaded processed audio number, label size: ', 
      len(audio_data_original), len(audio_data_processed), len(labels))

# feature extraction
feature_label_list = []
mfcc_list_original, mfcc_list_processed = np.empty((0,mfcc_size)), np.empty((0,mfcc_size))
# loop for each activity class k
for label_i in range(len(labels)):   
    mfcc_feat_original = mfcc(signal=audio_data_original[label_i], 
                         samplerate=file_sampling_rate,
                         winlen=window_length_secs,
                         winstep=hop_length_secs,
                         numcep=mfcc_size,
                         nfilt=26)   # mfcc of original audio
    mfcc_list_original = np.vstack((mfcc_list_original, mfcc_feat_original))
    mfcc_feat_processed = mfcc(signal=audio_data_original[label_i], 
                         samplerate=file_sampling_rate,
                         winlen=window_length_secs,
                         winstep=hop_length_secs,
                         numcep=mfcc_size,
                         nfilt=26)   # mfcc of privacy processed audio
    mfcc_list_processed = np.vstack((mfcc_list_processed, mfcc_feat_processed))
    feature_label_list.append([label_i] * mfcc_feat_processed.shape[0])   # list of labels of the features
flattened_feature_label_list = [item for sublist in feature_label_list for item in sublist]
    
#%%
"""
training and test
"""
data_original, labels_original = reshape_data_labels(data=mfcc_list_original, 
                                                     labels=flattened_feature_label_list, 
                                                     mfcc_size=mfcc_size, 
                                                     class_size=len(labels))
data_processed, labels_processed = reshape_data_labels(data=mfcc_list_original, 
                                                       labels=flattened_feature_label_list, 
                                                       mfcc_size=mfcc_size, 
                                                       class_size=len(labels))
data_original, labels_original = shuffle(data_original, labels_original)
data_processed, labels_processed = shuffle(data_processed, labels_processed)
print('data shape to CNN:, labels shape to CNN: ', data_processed.shape, labels_processed.shape)
model = cnn_model_fn(class_size=len(labels))
model.fit(data_original[:3000], labels_original[:3000],   
            batch_size=32,
            epochs=5,
            verbose=2,
            validation_data = (data_original[3000:4000], labels_original[3000:4000]),
            shuffle=True,
            callbacks=[EarlyStopping(monitor='val_acc', patience=5, mode='auto')])
save_model(model, count=1, save_model_dir=save_model_dir)
print('Well trained and saved')

#%%
"""
evaluate the model
"""
scores1 = model.evaluate(data_original, labels_original, verbose=1)
scores2 = model.evaluate(data_processed, labels_processed, verbose=1)
print("%s: %.2f%% %.2f%%" % (model.metrics_names[1], scores1[1]*100, scores2[1]*100))        
