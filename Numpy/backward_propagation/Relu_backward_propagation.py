# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 21:03:00 2017

@author: yangsu
"""
import numpy as np
 
class sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1.0-self.out)
        return dx
    
