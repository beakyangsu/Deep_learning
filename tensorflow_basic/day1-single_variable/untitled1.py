# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:59:29 2017

@author: yangsu
"""

import urllib.request

url = "http://api.aoikujira.com/ip/ini"
res = urllib.request.urlopen(url)
print(res)

data = res.read()
print(data)

text = data.decode("utf-8")
print("\ntext\n"+text)