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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import itertools

import check_dirs

#%%

#classes = ['Bathing',
#          'Flushing',
#          'Brushing',
#          'Doing shaver',
#          'Frying',
#          'Chopping',
#          'Microwave oven',
#          'Boiling', 
#          'Blender',
#          'Watching TV', 
#          'Listening musics', 
#          'Vacuum cleaning',  
#          'Washing',
#          'Chatting', 
#          'Strolling']
classes = ['A',
          'B',
          'C',
          'D',
          'E',
          'F',
          'G',
          'H', 
          'I',
          'J', 
          'K', 
          'L',  
          'M',
          'N', 
          'O']
#classes=['bathing','flushing']
# Fix random seed
seed(0)
set_random_seed(0)

#%%
    
def cnn_model_fn(class_size):
        model = Sequential()
        model.add(Conv1D(16, 8, strides=1, activation='relu', padding="same", input_shape=(13,1)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Conv1D(32, 8, strides=1, activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Conv1D(32, 8, strides=1, activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Flatten())
        #model.add(Dense(50, activation ='relu'))
        #model.add(Dropout(0.3))
        model.add(Dense(class_size, activation='softmax'))
            
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        loss = categorical_crossentropy
        model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])
        return model
    
#%%
"""
Save model and architecture, not used
"""
def save_model(clf, count, save_model_dir):
    check_dirs.check_dir(save_model_dir)
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
def reshape_data_labels(data, labels, feature_size, class_size):
    # Reshape training data as (#,feature_size,1) for CNN
    data = np.reshape(data, (data.shape[0], feature_size, 1))   
    # One-hot encoding for training labels: (#,size_labels)
    labels = np_utils.to_categorical(labels, class_size)
    return data, labels


#%%
"""
plot confusion matrix
"""

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

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
"""
Load data
"""
load_data_dir = './processed_user_data_fold-based/neighbouring_range_test/1000/test3/'
save_model_dir = './models/'
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
training_fold = 3   # 1 to 5
training = True

training_frame = pd.read_csv(load_data_dir + names[training_fold-1] + '.csv', header=None).values  
valid_frame = pd.read_csv(load_data_dir + names[training_fold + 4] + '.csv', header=None).values
training_data, training_labels = training_frame[:,:13], training_frame[:,13]
valid_data, valid_labels = valid_frame[:,:13], valid_frame[:,13]
 
#%%
"""
training and test
"""
#data_original, labels_original = reshape_data_labels(data=mfcc_list_original, 
#                                                     labels=flattened_feature_label_list, 
#                                                     mfcc_size=mfcc_size, 
#                                                     class_size=len(labels))
training_data, onehot_training_labels = reshape_data_labels(data=training_data, 
                                                            labels=training_labels, 
                                                            feature_size=training_data.shape[1], 
                                                            class_size=len(classes))
valid_data, onehot_valid_labels = reshape_data_labels(data=valid_data, 
                                                      labels=valid_labels, 
                                                      feature_size=valid_data.shape[1], 
                                                      class_size=len(classes))
#data_original, labels_original = shuffle(data_original, labels_original)
#data_processed, labels_processed = shuffle(data_processed, labels_processed)
print('training data shape:, labels shape: ', training_data.shape, onehot_training_labels.shape)

if training:
    # check save path
    check_dirs.check_dir(save_model_dir)
    model = cnn_model_fn(class_size=len(classes))
    model.fit(training_data, onehot_training_labels,   
              batch_size=128,
              epochs=20,
              verbose=2,
              validation_data = (valid_data, onehot_valid_labels),
              shuffle=True,
              callbacks=[EarlyStopping(monitor='val_acc', patience=5, mode='auto'),
                         K.callbacks.ModelCheckpoint(save_model_dir+"neighbouring_test-1000-fold3-epoch_{epoch:02d}-val_{val_acc:.4f}.hdf5", 
                                                     monitor='val_acc', 
                                                     verbose=0, 
                                                     save_best_only=True, 
                                                     save_weights_only=False, 
                                                     mode='auto', 
                                                     period=1)])
    #save_model(model, count=1, save_model_dir=save_model_dir)
    print('Well trained and saved')

else:
    model = cnn_model_fn(class_size=len(classes))
    model.load_weights('./final models both/0/method0-fold5-epoch_02-val_0.5606.hdf5')
    predictions = model.predict(valid_data)
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(valid_labels, predictions)
    f1 = f1_score(valid_labels, predictions, average='weighted')
    print("%s: %.4f%% %.4f%%" % ('acc, f1:', acc, f1))   
#%%    
    # confusion matrix
    predictions = np.reshape(predictions, (predictions.shape[0], 1))
    valid_labels = np.reshape(valid_labels, (valid_labels.shape[0], 1))
    # need to be commented when accumulating results
    #prediction_container, label_container = np.empty((0,1)), np.empty((0,1))   
    prediction_container, label_container = np.vstack((prediction_container, predictions)), np.vstack((label_container, valid_labels))
#%%
    C = confusion_matrix(label_container, prediction_container)
    plt.figure(num=1, figsize=(7,7))
    display_names = classes
    plot_confusion_matrix(C, classes=display_names, normalize=True,
                          title='Predicted Results')
    plt.show()     
