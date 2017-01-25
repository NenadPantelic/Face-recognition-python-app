#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import numpy as np

class AbstractFeature(object):

    def compute(self,X,y):
        pass
    
    def extract(self,X):
        pass
        
    def save(self):
        pass
    
    def load(self):
        pass
        
    def __repr__(self):
        return "AbstractFeature"

class Identity(AbstractFeature):
    """
    Najjednostavniji tip AbstractFeature, pogodno za SVM
    """
    def __init__(self):
        AbstractFeature.__init__(self)
        
    def compute(self,X,y):
        return X
    
    def extract(self,X):
        return X
    
    def __repr__(self):
        return "Identitet"


from facerec.util import asColumnMatrix
from facerec.operators import ChainOperator, CombineOperator
        
class PCA(AbstractFeature):
    def __init__(self, num_components=0):
        AbstractFeature.__init__(self)
        self._num_components = num_components
        
    def compute(self,X,y):
        # matrica kolona
        XC = asColumnMatrix(X)
        y = np.asarray(y)
        # broj komponenti
        if self._num_components <= 0 or (self._num_components > XC.shape[1]-1):
            self._num_components = XC.shape[1]-1
        self._mean = XC.mean(axis=1).reshape(-1,1)
        XC = XC - self._mean
        # menjamo strukturu velicine
        self._eigenvectors, self._eigenvalues, variances = np.linalg.svd(XC, full_matrices=False)
        # sortiramo karakteristicne vektore po njihovim kar.vrednstima u rastucem poretku
        idx = np.argsort(-self._eigenvalues)
        self._eigenvalues, self._eigenvectors = self._eigenvalues[idx], self._eigenvectors[:,idx]
        self._eigenvectors = self._eigenvectors[0:,0:self._num_components].copy()
        self._eigenvalues = self._eigenvalues[0:self._num_components].copy()
        # pravimo karakteristicne vrednosti
        self._eigenvalues = np.power(self._eigenvalues,2) / XC.shape[1]
        # ekstraktujemo karakteristike iz skupa
        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features
    
    def extract(self,X):
        X = np.asarray(X).reshape(-1,1)
        return self.project(X)
        
    def project(self, X):
        X = X - self._mean
        return np.dot(self._eigenvectors.T, X)

    def reconstruct(self, X):
        X = np.dot(self._eigenvectors, X)
        return X + self._mean

    @property
    def num_components(self):
        return self._num_components

    @property
    def eigenvalues(self):
        return self._eigenvalues
        
    @property
    def eigenvectors(self):
        return self._eigenvectors

    @property
    def mean(self):
        return self._mean
        
    def __repr__(self):
        return "PCA (broj komponenti=%d)" % (self._num_components)
        
class LDA(AbstractFeature):

    def __init__(self, num_components=0):
        AbstractFeature.__init__(self)
        self._num_components = num_components

    def compute(self, X, y):
        # matrica kolona
        XC = asColumnMatrix(X)
        y = np.asarray(y)
        # racuna dimenzije
        d = XC.shape[0]
        c = len(np.unique(y))        
        # postavlja regularan broj komponenti
        if self._num_components <= 0:
            self._num_components = c-1
        elif self._num_components > (c-1):
            self._num_components = c-1
        # racuna srednju vrednost
        meanTotal = XC.mean(axis=1).reshape(-1,1)
        Sw = np.zeros((d, d), dtype=np.float32)
        Sb = np.zeros((d, d), dtype=np.float32)
        for i in range(0,c):
            Xi = XC[:,np.where(y==i)[0]]
            meanClass = np.mean(Xi, axis = 1).reshape(-1,1)
            Sw = Sw + np.dot((Xi-meanClass), (Xi-meanClass).T)
            Sb = Sb + Xi.shape[1] * np.dot((meanClass - meanTotal), (meanClass - meanTotal).T)
        # racuna karakteristicne vrednosti matrica
        self._eigenvalues, self._eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
        # sortira te vektore po njihovim lambda vrednostima
        idx = np.argsort(-self._eigenvalues.real)
        self._eigenvalues, self._eigenvectors = self._eigenvalues[idx], self._eigenvectors[:,idx]
        self._eigenvalues = np.array(self._eigenvalues[0:self._num_components].real, dtype=np.float32, copy=True)
        self._eigenvectors = np.matrix(self._eigenvectors[0:,0:self._num_components].real, dtype=np.float32, copy=True)
        # karakteristike skupa podataka
        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features
        
    def project(self, X):
        return np.dot(self._eigenvectors.T, X)

    def reconstruct(self, X):
        return np.dot(self._eigenvectors, X)

    @property
    def num_components(self):
        return self._num_components

    @property
    def eigenvectors(self):
        return self._eigenvectors
    
    @property
    def eigenvalues(self):
        return self._eigenvalues
    
    def __repr__(self):
        return "LDA (num_components=%d)" % (self._num_components)
        
class Fisherfaces(AbstractFeature):

    def __init__(self, num_components=0):
        AbstractFeature.__init__(self)
        self._num_components = num_components
    
    def compute(self, X, y):
        #numpy rep
        Xc = asColumnMatrix(X)
        y = np.asarray(y)
        # statistika podataka
        n = len(y)
        c = len(np.unique(y))
        # definise karakteristike za ekstraktovanje
        pca = PCA(num_components = (n-c))
        lda = LDA(num_components = self._num_components)
        # fisherfaces PCA i LDA
        model = ChainOperator(pca,lda)
        model.compute(X,y)
        # skladisti kar.vrednosti i broj koriscenih komponenti
        self._eigenvalues = lda.eigenvalues
        self._num_components = lda.num_components
        # pca.eigenvectors*lda.eigenvectors
        self._eigenvectors = np.dot(pca.eigenvectors,lda.eigenvectors)
        # izvrsava Fisherfaces
        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features

    def extract(self,X):
        X = np.asarray(X).reshape(-1,1)
        return self.project(X)

    def project(self, X):
        return np.dot(self._eigenvectors.T, X)
    
    def reconstruct(self, X):
        return np.dot(self._eigenvectors, X)

    @property
    def num_components(self):
        return self._num_components
        
    @property
    def eigenvalues(self):
        return self._eigenvalues
    
    @property
    def eigenvectors(self):
        return self._eigenvectors

    def __repr__(self):
        return "Fisherfaces (num_components=%s)" % (self.num_components)

from facerec.lbp import LocalDescriptor, ExtendedLBP

class SpatialHistogram(AbstractFeature):
    def __init__(self, lbp_operator=ExtendedLBP(), sz = (8,8)):
        AbstractFeature.__init__(self)
        if not isinstance(lbp_operator, LocalDescriptor):
            raise TypeError("Operator facerec.lbp.LocalDescriptor je validan lbp_operator.")
        self.lbp_operator = lbp_operator
        self.sz = sz
        
    def compute(self,X,y):
        features = []
        for x in X:
            x = np.asarray(x)
            h = self.spatially_enhanced_histogram(x)
            features.append(h)
        return features
    
    def extract(self,X):
        X = np.asarray(X)
        return self.spatially_enhanced_histogram(X)

    def spatially_enhanced_histogram(self, X):
        # LBP
        L = self.lbp_operator(X)
        # piksel geometrija
        lbp_height, lbp_width = L.shape
        grid_rows, grid_cols = self.sz
        py = int(np.floor(lbp_height/grid_rows))
        px = int(np.floor(lbp_width/grid_cols))
        E = []
        for row in range(0,grid_rows):
            for col in range(0,grid_cols):
                C = L[row*py:(row+1)*py,col*px:(col+1)*px]
                H = np.histogram(C, bins=2**self.lbp_operator.neighbors, range=(0, 2**self.lbp_operator.neighbors), normed=True)[0]
                E.extend(H)
        return np.asarray(E)
    
    def __repr__(self):
        return "SpatialHistogram (operator=%s, grid=%s)" % (repr(self.lbp_operator), str(self.sz))
