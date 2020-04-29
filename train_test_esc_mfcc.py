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
import pandas as pd

import keras as K
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, f1_score
import itertools

from sklearn.ensemble import RandomForestClassifier
import pickle

import check_dirs

#%%
'''global definition'''
classes = ['crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps',
          'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping']
# Fix random seed
seed(0)
set_random_seed(0)

#%%

def cnn_model_fn(class_size):
    """
    return a compiled model
    """
    model = Sequential()
    model.add(Conv1D(64, 4, strides=2, activation='relu', padding="same", input_shape=(19,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv1D(64, 4, strides=2, activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(64, activation ='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation ='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation ='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(class_size, activation='softmax'))
        
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    loss = categorical_crossentropy
    model.compile(loss=loss,
              optimizer=opt,
              metrics=['accuracy'])
    return model
    

#%%
def rf_model():        
    """
    return a built random forest classifier
    """
    clf = RandomForestClassifier(n_estimators=13, criterion="gini", random_state=0, verbose=1, n_jobs=-1)
    return clf
    
#%%
def reshape_data_labels(data, labels, feature_size, class_size):
    """
    one-hot encoding and training data reshaping
    args:
        data, labels
        feature_size: dimensions of the features
        class_size: number of classes
    return:
        reshaped data, labels
    """
    # Reshape training data as (#,feature_size,1) for CNN
    data = np.reshape(data, (data.shape[0], feature_size, 1))   
    # One-hot encoding for training labels: (#,size_labels)
    labels = np_utils.to_categorical(labels, class_size)
    return data, labels


#%%
'''plot confusion matrix'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    plot confusion matrix
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Activities')


#%%
'''Load data'''
option = 0    # 4, 5, 6, 7
training_fold = 5   
load_data_dir = './esc_features/mfcc/option' + str(option) + '/fold' + str(training_fold) + '/'

save_model_dir = './models_mfcc/'   # path to save temporary models
weights_path = './models_mfcc/final_models/method0-fold5-epoch_02-val_0.4430.hdf5'   # path to load FC weights (for pred)

training = False
method = 'FC'

training_frame = pd.read_csv(load_data_dir + 'training.csv', header=None).values  
valid_frame = pd.read_csv(load_data_dir + 'test.csv', header=None).values

training_data, training_labels = training_frame[:,:19], training_frame[:,19]
valid_data, valid_labels = valid_frame[:,:19], valid_frame[:,19]
 
#%%
'''training and test'''

training_data_to_cnn, training_labels_to_cnn = reshape_data_labels(data=training_data, 
                                                                         labels=training_labels, 
                                                                         feature_size=training_data.shape[1], 
                                                                         class_size=len(classes))
valid_data_to_cnn, valid_labels_to_cnn = reshape_data_labels(data=valid_data, 
                                                             labels=valid_labels, 
                                                             feature_size=valid_data.shape[1], 
                                                             class_size=len(classes))

print('training data shape:, labels shape: ', training_data_to_cnn.shape, training_labels_to_cnn.shape)

if training:
    # check save path
    check_dirs.check_dir(save_model_dir)
    # fully-connected layers
    if method == 'FC':
        model = cnn_model_fn(class_size=len(classes))
        model.fit(training_data_to_cnn, training_labels_to_cnn,   
                  batch_size=64,
                  epochs=100,
                  verbose=2,
                  validation_data = (valid_data_to_cnn, valid_labels_to_cnn),
                  shuffle=True,
                  callbacks=[EarlyStopping(monitor='val_acc', patience=10, mode='auto'),
                             K.callbacks.ModelCheckpoint(save_model_dir+"method7-fold3-epoch_{epoch:02d}-val_{val_acc:.4f}.hdf5", 
                                                         monitor='val_acc', 
                                                         verbose=0, 
                                                         save_best_only=True, 
                                                         save_weights_only=True, 
                                                         mode='auto', 
                                                         period=1),
                                                         K.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, min_lr = 0.00001)])
    # random forest
    elif method == 'RF':
        model = rf_model()
        model.fit(training_data, training_labels)
        pickle.dump(model, open(save_model_dir + 'rf.sav', 'wb'))

    print('Well trained and saved')
    print('Well trained and saved')

else:
    if method == 'FC':
        model = cnn_model_fn(class_size=len(classes))
        model.load_weights(weights_path)
        predictions = model.predict(valid_data_to_cnn)
        predictions = np.argmax(predictions, axis=1)
        
    elif method == 'RF':
        model = pickle.load(open(save_model_dir + 'rf.sav', 'rb'))
        predictions = model.predict(valid_data)
        
    acc = accuracy_score(valid_labels, predictions)
    f1 = f1_score(valid_labels, predictions, average='weighted')
    print("%s: %.4f%% %.4f%%" % ('acc, f1:', acc, f1))   
    
