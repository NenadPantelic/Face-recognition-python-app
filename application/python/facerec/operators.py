#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

from facerec.feature import AbstractFeature
import numpy as np

class FeatureOperator(AbstractFeature):
    """
     FeatureOperator je opeator za dva modela.
    
    Args:
        model1 [AbstractFeature]
        model2 [AbstractFeature]
    """
    def __init__(self,model1,model2):
        if (not isinstance(model1,AbstractFeature)) or (not isinstance(model2,AbstractFeature)):
            raise Exception("FeatureOperator radi samo na klasama koje implementiraju AbstractFeature!")
        self.model1 = model1
        self.model2 = model2
    
    def __repr__(self):
        return "FeatureOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"
    
class ChainOperator(FeatureOperator):
    """
    ChainOperator uvezuje dva ekstrakciona modula:
        model2.compute(model1.compute(X,y),y)
   
    
    Args:
        model1 [AbstractFeature]
        model2 [AbstractFeature]
    """
    def __init__(self,model1,model2):
        FeatureOperator.__init__(self,model1,model2)
        
    def compute(self,X,y):
        X = self.model1.compute(X,y)
        return self.model2.compute(X,y)
        
    def extract(self,X):
        X = self.model1.extract(X)
        return self.model2.extract(X)
    
    def __repr__(self):
        return "ChainOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"
        
class CombineOperator(FeatureOperator):
    """
    CombineOperator kombinuje izlaze dva multidimenzionalna modula.
        (model1.compute(X,y),model2.compute(X,y))
        
        
    Args:
        model1 [AbstractFeature]
        model2 [AbstractFeature]
        
    """
    def __init__(self,model1,model2):
        FeatureOperator.__init__(self, model1, model2)
        
    def compute(self,X,y):
        A = self.model1.compute(X,y)
        B = self.model2.compute(X,y)
        C = []
        for i in range(0, len(A)):
            ai = np.asarray(A[i]).reshape(1,-1)
            bi = np.asarray(B[i]).reshape(1,-1)
            C.append(np.hstack((ai,bi)))
        return C
    
    def extract(self,X):
        ai = self.model1.extract(X)
        bi = self.model2.extract(X)
        ai = np.asarray(ai).reshape(1,-1)
        bi = np.asarray(bi).reshape(1,-1)
        return np.hstack((ai,bi))

    def __repr__(self):
        return "CombineOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"
        
class CombineOperatorND(FeatureOperator):
    """
    CombineOperatorND kombinuje izlaze dva multidimenzionalna modula.
        (model1.compute(X,y),model2.compute(X,y))
        
    Args:
        model1 [AbstractFeature]
        model2 [AbstractFeature]
        hstack [bool]  horizontalno (True) i vertikalno ako je False
        
    """
    def __init__(self,model1,model2, hstack=True):
        FeatureOperator.__init__(self, model1, model2)
        self._hstack = hstack
        
    def compute(self,X,y):
        A = self.model1.compute(X,y)
        B = self.model2.compute(X,y)
        C = []
        for i in range(0, len(A)):
            if self._hstack:
                C.append(np.hstack((A[i],B[i])))
            else:
                C.append(np.vstack((A[i],B[i])))
        return C
    
    def extract(self,X):
        ai = self.model1.extract(X)
        bi = self.model2.extract(X)
        if self._hstack:
            return np.hstack((ai,bi))
        return np.vstack((ai,bi))

    def __repr__(self):
        return "CombineOperatorND(" + repr(self.model1) + "," + repr(self.model2) + ", hstack=" + str(self._hstack) + ")"
