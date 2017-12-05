# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 21:15:46 2017

@author: yangsu
"""

import sys
import urllib.request as req
import urllib.parse as parse


if len(sys.argv) <= 1 :
    print("USAGE: download-forecast-argv <Region Number>")
    sys.exit()
    
regionNumber = sys.argv[1]

API =  "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"
values = { 'stnId' : regionNumber }

params = parse.urlencode(values)
url = API + "?" + params
print("url= " , url)

data = req.urlopen(url).read()
text = data.decode("utf-8")
print("text= ", text)