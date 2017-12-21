# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:26:49 2017

@author: yangsu
"""
import urllib.request 

url = "http://uta.pw/shodou/img/28/214.png"
savename = "test.png"

mem = urllib.request.urlopen(url).read()

with open(savename, mode = "wb") as f :
    f.write(mem)
#urllib.request.urlretrieve(url, savename)
print("saved!")
