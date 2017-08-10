# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 19:21:14 2017

@author: yangsu
"""
import sys, os
sys.patch.append(os.pardir)
import numpy as np

from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size, weigh_init_std=0.01):
        
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward()
            
        return x
        
    def loss(self, x,t) :
        y=self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_w = lamda w : self.loss(x,t)
        
        grads={}
        grads['w1'] = numerical_gradient(loss_w, self.params['w1'])
        grads['w2'] = numerical_gradient(loss_w, self.params['w2'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        return grads
    
    def gradient(self, x, t):
        self.loss(x,t)
        
        dout = 1
        dout = self.lastLayser.backward(dout)
        
        layers = list(self.layers.valuse())
        layers.reverse()
        for layer in layers :
            dout = layer.backward(dout)
            
        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
        return grads