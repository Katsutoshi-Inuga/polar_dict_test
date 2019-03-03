#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:40:13 2019

@author: hoge
"""

import polarlike
import pandas as pd
import numpy as np

import util 


def removesp(text):
    text = text.replace(' ','')
    return text

df = pd.read_pickle('movie_rev.pkl')
df['title_main'] = df[2].str.cat(df[3])
df['title_main'] = df['title_main'].apply(removesp)
data=np.array(df.values.tolist())

x,vectizer = polarlike.cvectizer(data[:,4])


"""
出現回数(ベクトル)の確認
x.toarray()
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 1, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)
"""

"""
ベクトル算出時の単語
vectizer.get_feature_names()
 '衣装',
 '裏切り',
 '製作',
 '西',
 '要ら',
 '要素',
 '見',
 ...]
"""

#単語ごとの極性を算出する
ratings = data[:,1].astype(np.int)
rating_norm=util.normalzie(ratings)
posinega_arr = polarlike.calcPosiNega(x,rating_norm)

polar_dict=np.array((vectizer.get_feature_names(),posinega_arr)).T
aaa = pd.DataFrame(polar_dict)