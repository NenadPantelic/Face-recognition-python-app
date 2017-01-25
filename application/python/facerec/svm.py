#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

from facerec.classifier import SVM
from facerec.validation import KFoldCrossValidation
from facerec.model import PredictableModel
from svmutil import *
from itertools import product
import numpy as np
import logging


def range_f(begin, end, step):
    seq = []
    while True:
        if step == 0: break
        if step > 0 and begin > end: break
        if step < 0 and begin < end: break
        seq.append(begin)
        begin = begin + step
    return seq

def grid(grid_parameters):
    grid = []
    for parameter in grid_parameters:
        begin, end, step = parameter
        grid.append(range_f(begin, end, step))
    return product(*grid)

def grid_search(model, X, y, C_range=(-5,  15, 2), gamma_range=(3, -15, -2), k=5, num_cores=1):
    
    if not isinstance(model, PredictableModel):
        raise TypeError("GridSearch ocekuje PredictableModel.")
    if not isinstance(model.classifier, SVM):
        raise TypeError("Koristite facerec.classifier.SVM!")
    
    logger = logging.getLogger("facerec.svm.gridsearch")
    logger.info("Grid Search.")
    
    # najbolja kombinacija parametara; vodi se racuna i gama opsegu; postavlja se linearni kernel
    best_parameter = svm_parameter("-q")
    best_parameter.kernel_type = model.classifier.param.kernel_type
    best_parameter.nu = model.classifier.param.nu
    best_parameter.coef0 = model.classifier.param.coef0
    if (gamma_range is None) or (model.classifier.param.kernel_type == LINEAR):
        gamma_range = (0, 0, 1)
    
    # najbolja greska do sada
    best_accuracy = np.finfo('float').min
    
    # pravi mrezu (Dekartov proizvod opsega)        
    g = grid([C_range, gamma_range])
    results = []
    for p in g:
        C, gamma = p
        C, gamma = 2**C, 2**gamma
        model.classifier.param.C, model.classifier.param.gamma = C, gamma

        cv = KFoldCrossValidation(model=model,k=k)
        cv.validate(X,y)
        results.append([C, gamma, cv.accuracy])
        
        # najbolja kombinacija parametara
        if cv.accuracy > best_accuracy:
            logger.info("best_accuracy=%s" % (cv.accuracy))
            best_accuracy = cv.accuracy
            best_parameter.C, best_parameter.gamma = C, gamma
        
        logger.info("%d-CV Result = %.2f." % (k, cv.accuracy))
        
    # najbolji skup parametara ---> najbolji pronadjen
    return best_parameter, results
