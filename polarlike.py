#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 02:04:20 2019

@author: hoge
"""

import util
import numpy as np
import numba
import math
from sklearn.feature_extraction.text import CountVectorizer

data = np.array([
        ['本日の大阪の天気は晴れです',1],
        ['本日の大阪の天気は曇りです',0],
        ['本日の大阪の天気は雨です',-1],        
        ])

@numba.jit
def cvectizer(docs):
    vectizer = CountVectorizer(analyzer=util.terms)
    x = vectizer.fit_transform(docs)
    return x,vectizer

@numba.jit
def calcPosiNega(x,ratings):
    row_c,col_c = x.shape
    calcPosiNegaByRow = np.empty((row_c,col_c))
    calcPosiNegaByCol = np.empty(col_c)
    for row,rate in zip(range(row_c),ratings):
        for col in range(col_c):
            ans = rate * np.log(x[row,col] / np.sum(x[row,:]) +1 )
            ans = 0 if math.isnan(ans) else ans
            calcPosiNegaByRow[row][col] = ans
    calcPosiNegaByCol = np.sum(calcPosiNegaByRow,axis=0)
    return calcPosiNegaByCol



x,vectizer = cvectizer(data[:,0])


"""
出現回数(ベクトル)の確認
x.toarray()
array([[1, 1, 1, 0, 1, 0],
       [1, 1, 0, 1, 1, 0],
       [1, 1, 0, 0, 1, 1]], dtype=int64)
"""

"""
ベクトル算出時の単語
vectizer.get_feature_names()
['大阪', '天気', '晴れ', '曇り', '本日', '雨']
"""

#単語ごとの極性を算出する
ratings = data[:,1].astype(np.int)
posinega_arr = calcPosiNega(x,ratings)

polar_dict=np.array((vectizer.get_feature_names(),posinega_arr)).T
"""
print(polar_dict)
[['大阪' '0.0']
 ['天気' '0.0']
 ['晴れ' '0.09691001300805642']
 ['曇り' '0.0']
 ['本日' '0.0']
 ['雨' '-0.09691001300805642']]
"""