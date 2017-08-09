# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 20:12:46 2017

@author: yangsu
"""
#use multiplication, Add


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

class AddLayer :
    def __init__(self):
        pass
        
    def forward(self, x, y):
        return x + y
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    
apple_price_per_one = 100
apple_num = 2
orange_price_per_one = 150
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()


apple_price = mul_apple_layer.forward(apple_price_per_one, apple_num)
orange_price = mul_orange_layer.forward(orange_price_per_one, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price_with_tax = mul_tax_layer.forward(all_price, tax)

print(price_with_tax)
#forward propagation


dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple_price_per_one, dapple_num = mul_apple_layer.backward(dapple_price)
dorange_price_per_one, dorange_num = mul_apple_layer.backward(dorange_price)

print(dapple_num, dapple_price_per_one, dorange_num, dorange_price_per_one, dtax)
#backward propagation


