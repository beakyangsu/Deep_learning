# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 20:12:46 2017

@author: yangsu
"""
#use multiplication only

# problem1 :
#if you buy two apples, you calcuate how much is it when the tax is 1.1 and apple's price is 100  
#using forward propagation

#problem2 :
# what are grediants of apple_num, apple_price and tax when final price(price_with_tax) get higher a little bit. 
#using backward propagation


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy
        
    
apple_price_per_one = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple_price_per_one, apple_num)
price_with_tax = mul_tax_layer.forward(apple_price, tax)

print(price_with_tax)
#forward propagation


dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple_price_per_one, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple_num, dapple_price_per_one, dtax)
#backward propagation


