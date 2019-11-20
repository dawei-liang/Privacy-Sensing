#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:39:14 2019

@author: dawei
"""

from sklearn.decomposition import PCA

def pca_transform(spectrogram, components):
    pca = PCA(n_components=components)
    pca.fit(spectrogram[:(spectrogram.shape[0]//2)])
    eigenvectors = pca.transform(spectrogram)
    reduced_spectrogram = pca.inverse_transform(eigenvectors)
    return reduced_spectrogram