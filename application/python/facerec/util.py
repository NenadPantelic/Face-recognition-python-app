#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import numpy as np
import random
from scipy import ndimage

def read_image(filename):
    imarr = np.array([])
    try:
        im = Image.open(os.path.join(filename))
        im = im.convert("L") # u grayscale
        imarr = np.array(im, dtype=np.uint8)
    except IOError as e:
        print("I/O error: {0}".format(e))
    except:
        print("Nije moguce otvoriti sliku.")
    return imarr

def asRowMatrix(X):
    """
    Kreira matricu-red iz multidimenzionalnog skupa u listu l.
    
    X [list] lista sa multi-dimenzionalnim podacima.
    """
    if len(X) == 0:
        return np.array([])
    total = 1
    for i in range(0, np.ndim(X[0])):
        total = total * X[0].shape[i]
    mat = np.empty([0, total], dtype=X[0].dtype)
    for row in X:
        mat = np.append(mat, row.reshape(1,-1), axis=0) 
    return np.asmatrix(mat)

def asColumnMatrix(X):
    """
    Creates a column-matrix from multi-dimensional data items in list l.
    
    X [list] List with multi-dimensional data.
    """
    if len(X) == 0:
        return np.array([])
    total = 1
    for i in range(0, np.ndim(X[0])):
        total = total * X[0].shape[i]
    mat = np.empty([total, 0], dtype=X[0].dtype)
    for col in X:
        mat = np.append(mat, col.reshape(-1,1), axis=1) # same as hstack
    return np.asmatrix(mat)


def minmax_normalize(X, low, high, minX=None, maxX=None, dtype=np.float):
    """ min-max normalizacija matrice [low,high].
    
    Args:
        X [rows x columns] input 
        low [broj] donja granica
        high [broj] gornja granica
    """
    if minX is None:
        minX = np.min(X)
    if maxX is None:
        maxX = np.max(X)
    minX = float(minX)
    maxX = float(maxX)
    # normalizacija [0...1].    
    X = X - minX
    X = X / (maxX - minX)
    # skaliranje [low...high].
    X = X * (high-low)
    X = X + low
    return np.asarray(X, dtype=dtype)

def zscore(X):
    X = np.asanyarray(X)
    mean = X.mean()
    std = X.std() 
    X = (X-mean)/std
    return X, mean, std

def shuffle(X,y):
    idx = np.argsort([random.random() for i in xrange(y.shape[0])])
    return X[:,idx], y[idx]

def shuffle_array(X,y):
    """ Mesa dva niza!
    """
    idx = np.argsort([random.random() for i in xrange(len(y))])
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]
    return (X, y)
    


