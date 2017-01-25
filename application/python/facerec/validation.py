#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import numpy as np
import math as math
import random as random
import logging

from facerec.model import PredictableModel
from facerec.classifier import AbstractClassifier


from builtins import range



def shuffle(X, y):
    """ mesa dva niza po kolonama (len(X) == len(y))
        
        Args:
        
            X [dim x num_data] input 
            y [1 x num_data] klase

        Returns:

            izmesane nizove.
    """
    idx = np.argsort([random.random() for i in range(len(y))])
    y = np.asarray(y)
    X = [X[i] for i in idx]
    y = y[idx]
    return (X, y)
    
def slice_2d(X,rows,cols):
    """
    
    Vrsi odsecanje 2D liste u ravan niz.
    
    Args:
    
        X [num_rows x num_cols] multi-dimenzionalni podaci
        rows [list] redovi za slice
        cols [list] kolone za slice
    
    Example:
    
        >>> X=[[1,2,3,4],[5,6,7,8]]
        >>> # slice prva dva reda i prve kolone
        >>> Commons.slice(X, range(0,2), range(0,1)) # returns [1, 5]
        >>> Commons.slice(X, range(0,1), range(0,4)) # returns [1,2,3,4]
    """
    return [X[i][j] for j in cols for i in rows]

def precision(true_positives, false_positives):
    """preciznosst koja se racuna kao:
        
        true_positives/(true_positives+false_positives)
        
    """
    return accuracy(true_positives, 0, false_positives, 0)
    
def accuracy(true_positives, true_negatives, false_positives, false_negatives, description=None):
    """tacnost kao:
    
        (true_positives+true_negatives)/(true_positives+false_positives+true_negatives+false_negatives)
        
    """
    true_positives = float(true_positives)
    true_negatives = float(true_negatives)
    false_positives = float(false_positives)
    false_negatives = float(false_negatives)
    if (true_positives + true_negatives + false_positives + false_negatives) < 1e-15:
       return 0.0
    return (true_positives+true_negatives)/(true_positives+false_positives+true_negatives+false_negatives)

class ValidationResult(object):
    
    def __init__(self, true_positives, true_negatives, false_positives, false_negatives, description):
        self.true_positives = true_positives
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.description = description
        
    def __repr__(self):
        res_precision = precision(self.true_positives, self.false_positives) * 100
        res_accuracy = accuracy(self.true_positives, self.true_negatives, self.false_positives, self.false_negatives) * 100
        return "ValidationResult (Description=%s, Precision=%.2f%%, Accuracy=%.2f%%)" % (self.description, res_precision, res_accuracy)
    
class ValidationStrategy(object):
    
    def __init__(self, model):
       
        if not isinstance(model,PredictableModel):
            raise TypeError("Validacija radi samo sa tipom PredictableModel.")
        self.model = model
        self.validation_results = []
    
    def add(self, validation_result):
        self.validation_results.append(validation_result)
        
    def validate(self, X, y, description):

        raise NotImplementedError("Mora se implementiratu metod validacije!")
        
    
    def print_results(self):
        print(self.model)
        for validation_result in self.validation_results:
            print(validation_result)

    def __repr__(self):
        return "Validation Kernel (model=%s)" % (self.model)
        
class KFoldCrossValidation(ValidationStrategy):

    def __init__(self, model, k=10):
       

        super(KFoldCrossValidation, self).__init__(model=model)
        self.k = k
        self.logger = logging.getLogger("facerec.validation.KFoldCrossValidation")

    def validate(self, X, y, description="ExperimentName"):

        X,y = shuffle(X,y)
        c = len(np.unique(y))
        foldIndices = []
        n = np.iinfo(np.int).max
        for i in range(0,c):
            idx = np.where(y==i)[0]
            n = min(n, idx.shape[0])
            foldIndices.append(idx.tolist()); 

        if n < self.k:
            self.k = n

        foldSize = int(math.floor(n/self.k))
        
        true_positives, false_positives, true_negatives, false_negatives = (0,0,0,0)
        for i in range(0,self.k):
        
            self.logger.info("Processing fold %d/%d." % (i+1, self.k))
                
           
            l = int(i*foldSize)
            h = int((i+1)*foldSize)
            testIdx = slice_2d(foldIndices, cols=range(l,h), rows=range(0, c))
            trainIdx = slice_2d(foldIndices,cols=range(0,l), rows=range(0,c))
            trainIdx.extend(slice_2d(foldIndices,cols=range(h,n),rows=range(0,c)))
            
            
            Xtrain = [X[t] for t in trainIdx]
            ytrain = y[trainIdx]
                        
            self.model.compute(Xtrain, ytrain)
            
            
            for j in testIdx:
                prediction = self.model.predict(X[j])[0]
                if prediction == y[j]:
                    true_positives = true_positives + 1
                else:
                    false_positives = false_positives + 1
                    
        self.add(ValidationResult(true_positives, true_negatives, false_positives, false_negatives, description))
    
    def __repr__(self):
        return "k-Fold Cross Validation (model=%s, k=%s)" % (self.model, self.k)

class LeaveOneOutCrossValidation(ValidationStrategy):


    def __init__(self, model):

        super(LeaveOneOutCrossValidation, self).__init__(model=model)
        self.logger = logging.getLogger("facerec.validation.LeaveOneOutCrossValidation")
        
    def validate(self, X, y, description="ExperimentName"):
        
        #(X,y) = shuffle(X,y)
        true_positives, false_positives, true_negatives, false_negatives = (0,0,0,0)
        n = y.shape[0]
        for i in range(0,n):
            
            self.logger.info("Processing fold %d/%d." % (i+1, n))
            
           
            trainIdx = []
            trainIdx.extend(range(0,i))
            trainIdx.extend(range(i+1,n))
            
           
            Xtrain = [X[t] for t in trainIdx]
            ytrain = y[trainIdx]
            
            
            self.model.compute(Xtrain, ytrain)
            
            
            prediction = self.model.predict(X[i])[0]
            if prediction == y[i]:
                true_positives = true_positives + 1
            else:
                false_positives = false_positives + 1
                
        self.add(ValidationResult(true_positives, true_negatives, false_positives, false_negatives, description))
    
    def __repr__(self):
        return "Leave-One-Out Cross Validation (model=%s)" % (self.model)

class LeaveOneClassOutCrossValidation(ValidationStrategy):


    def __init__(self, model):

        super(LeaveOneClassOutCrossValidation, self).__init__(model=model)
        self.logger = logging.getLogger("facerec.validation.LeaveOneClassOutCrossValidation")
        
    def validate(self, X, y, g, description="ExperimentName"):
       
        true_positives, false_positives, true_negatives, false_negatives = (0,0,0,0)
        
        for i in range(0,len(np.unique(y))):
            self.logger.info("Validating Class %s." % i)
           
            trainIdx = np.where(y!=i)[0]
            testIdx = np.where(y==i)[0]
            
            Xtrain = [X[t] for t in trainIdx]
            gtrain = g[trainIdx]
            
            
            self.model.compute(Xtrain, gtrain)
            
            for j in testIdx:
                
                prediction = self.model.predict(X[j])[0]
                if prediction == g[j]:
                    true_positives = true_positives + 1
                else:
                    false_positives = false_positives + 1
        self.add(ValidationResult(true_positives, true_negatives, false_positives, false_negatives, description))
    
    def __repr__(self):
        return "Leave-One-Class-Out Cross Validation (model=%s)" % (self.model)

class SimpleValidation(ValidationStrategy):

    def __init__(self, model):
 
        super(SimpleValidation, self).__init__(model=model)
        self.logger = logging.getLogger("facerec.validation.SimpleValidation")

    def validate(self, Xtrain, ytrain, Xtest, ytest, description="ExperimentName"):
 
        self.logger.info("Validacija.")
       
        self.model.compute(Xtrain, ytrain)

        self.logger.debug("Model je obradjen.")

        true_positives, false_positives, true_negatives, false_negatives = (0,0,0,0)
        count = 0
        for i in ytest:
            self.logger.debug("Predicting %s/%s." % (count, len(ytest)))
            prediction = self.model.predict(Xtest[i])[0]
            if prediction == ytest[i]:
                true_positives = true_positives + 1
            else:
                false_positives = false_positives + 1
            count = count + 1
        self.add(ValidationResult(true_positives, true_negatives, false_positives, false_negatives, description))

    def __repr__(self):
        return "Simple Validation (model=%s)" % (self.model)
