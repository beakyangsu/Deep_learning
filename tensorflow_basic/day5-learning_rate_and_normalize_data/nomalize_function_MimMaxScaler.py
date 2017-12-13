# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:37:00 2017

@author: yangsu
"""

import numpy as np

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)