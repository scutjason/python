#!/usr/bin/python
# encoding: utf-8
# -*- coding: utf8 -*-
"""
Created by 10171618.
File:               down_file.py
User:               xuyong
Create Date:        2017/3/24
Create Time:        19:40
 """
import urllib.request
import urllib.parse
import requests

url = 'http://10.118.202.164:11718/app/ipdr/server/export/test_0505.tar.gz'
file_name = urllib.parse.unquote(url).split('/')[-1]
save_path = "C:/Users/jason/Desktop/" + file_name

def cbk(a, b, c):
    """
    :param a: have downloaded module
    :param b: data block size
    :param c: http file size
    :return:
    """
    per = 100 * a * b / c
    if per > 100:
        per = 100
    print('%.2f%%' %per)

# there are three method to download from http with Python
# the first is use urllib.urlretrieve
print("downloading with urllib.request 1")
urllib.request.urlretrieve(url, save_path, cbk)

# the second mothod use urllib.request
print("downloading with urllib.request 2")
f = urllib.request.urlopen(url)
data=f.read()
with open(save_path, "wb") as code:
    code.write(data)

# the three mothod use requests
print("downloading with urllib.request 3")
r = requests.get(url)
with open(save_path, "wb") as code:
    code.write(r.content)
