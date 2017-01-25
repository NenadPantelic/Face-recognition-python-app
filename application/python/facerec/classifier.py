#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.
# Ovo je deo pomocnog koda koji je deo paketa bytefish autora Philipp Wagner-a

from facerec.distance import EuclideanDistance
from facerec.util import asRowMatrix
import logging
import numpy as np
import operator as op
from sklearn import svm

class AbstractClassifier(object):

    def compute(self,X,y):
        raise NotImplementedError("Every AbstractClassifier must implement the compute method.")
    
    def predict(self,X):
        raise NotImplementedError("Every AbstractClassifier must implement the predict method.")

    def update(self,X,y):
        raise NotImplementedError("This Classifier is cannot be updated.")

class NearestNeighbor(AbstractClassifier):
    """
    Implementiranje k-Nearest Neighbor modela sa metrikom za racunanje distance.
    """
    def __init__(self, dist_metric=EuclideanDistance(), k=1):
        AbstractClassifier.__init__(self)
        self.k = k
        self.dist_metric = dist_metric
        self.X = []
        self.y = np.array([], dtype=np.int32)

    def update(self, X, y):
        """
        Metoda za update klasifikatora
        """
        self.X.append(X)
        self.y = np.append(self.y, y)

    def compute(self, X, y):
        self.X = X
        self.y = np.asarray(y)
    
    def predict(self, q):
        """
       Predvidja model za zadati unos q. 
        
        input:
        
            q: niz koji unosimo kao argument
            
        output:
        
            Lista sa izlazom klasifikatora. Kao prvi elemnet vraca predvidjenu klasu. Takodje racuna distance kod kNN. 
            
            Primer:
            
                [ 0, 
                   { 'labele'    : [ 0,      0,      1      ],
                     'distance' : [ 10.132, 10.341, 13.314 ]
                   }
                ]
            
              
                    
        """
        distances = []
        for xi in self.X:
            xi = xi.reshape(-1,1)
            d = self.dist_metric(xi, q)
            distances.append(d)
        if len(distances) > len(self.y):
            raise Exception("More distances than classes. Is your distance metric correct?")
        distances = np.asarray(distances)
        # sortiramo u rastucem poretku:
        idx = np.argsort(distances)
        # Sortiramo labele i distance the labels and distances accordingly:
        sorted_y = self.y[idx]
        sorted_distances = distances[idx]
        # Uzimamo prvih k elemenata:
        sorted_y = sorted_y[0:self.k]
        sorted_distances = sorted_distances[0:self.k]
        # Histogram:
        hist = dict((key,val) for key, val in enumerate(np.bincount(sorted_y)) if val)
	#print hist
        # Izvucemo podeok sa najvecom f:
        predicted_label = max(iter(hist.items()), key=op.itemgetter(1))[0]
	#print predicted_label
        # A classifier should output a list with the label as first item and
        # generic data behind. The k-nearest neighbor classifier outputs the 
        # distance of the k first items. So imagine you have a 1-NN and you
        # want to perform a threshold against it, you should take the first
        # item 
	#print [predicted_label, { 'labels' : sorted_y, 'distances' : sorted_distances }]
        return [predicted_label, { 'labels' : sorted_y, 'distances' : sorted_distances }]
        
    def __repr__(self):
        return "NearestNeighbor (k=%s, dist_metric=%s)" % (self.k, repr(self.dist_metric))

class SVM(AbstractClassifier):
    """
    Pomocna klasa koja realizuje SVM ML algoritme
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None):
        AbstractClassifier.__init__(self)
        self.logger = logging.getLogger("facerec.classifier.SVM")
        # Initialize the SVM with given Parameters:
        self.svm = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, random_state=random_state)
        # Store parameters:
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state 


    
    def compute(self, X, y):
        X = asRowMatrix(X)
        y = np.asarray(y)
        self.svm.fit(X, y)
        self.y = y
    
    def predict(self, X):
        """
        
        Input:
        
            X: Slika, u formi niza.
        
        Returns:
        
            
            
                p_labels: lista predvidjenih labela
                p_acc: n-torka koja sadrzi preciznost procene, srednju kvadratnu gresku i koeficijent korelacije u regresiji
                p_vals: lista procenjenih verovatnoca
        """
        # transformisemosliku u niz-vektor:
        X = np.asarray(X).reshape(1,-1)
        # Procena verovatnoca:
        results = self.svm.predict_proba(y)[0]
        # Dobijamo verovatnoce:
        prob_per_class_dictionary = dict(zip(self.svm.classes_, results))
        # Sortiramo klase po verovatnoci:
        results_ordered_by_probability = map(lambda x: x[0], sorted(zip(self.svm.classes_, results), key=lambda x: x[1], reverse=True))
        # Prvi item je klasa koju predvidjamo
        predicted_label = int(results_ordered_by_probability[0])
	print [predicted_label, { 'results' : results }]

        return [predicted_label, { 'results' : results }]
    
    def __repr__(self):        
        return "Support Vector Machine (C=%s, kernel=%s, degree=%s, gamma=%s, coef0=%s, shrinking=%s, probability=%s, tol=%s, cache_size=%s, class_weight=%s, verbose=%s, max_iter=%s, decision_function_shape%s, random_state=%s)" % (self.C, self.kernel, self.degree, self.gamma, self.coef0, self.shrinking, self.probability, self.tol, self.cache_size, self.class_weight, self.verbose, self.max_iter, self.decision_function_shape, self.random_state)
