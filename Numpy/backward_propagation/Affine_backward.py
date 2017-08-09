# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 21:32:29 2017

@author: yangsu
"""

import numpy as np

class Affine :
    deif __init__(self):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
        
    delf forward(self, x) :
        self.x = x
        out = np.dot(x,self.w)+self.b
        return out
    
    def backward(self,dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0) 
        
        return dx