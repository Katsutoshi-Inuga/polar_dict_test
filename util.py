#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 02:24:58 2019

@author: hoge
"""

import MeCab
import numpy as np

PART_OF_SEARCH=("名詞","形容詞","動詞","形容動詞")

def _split_to_word(text,to_stem=False):
    tagger = MeCab.Tagger('mecabrc')
    mecab_result = tagger.parse(text)
    words_info = mecab_result.split('\n')
    words = []
    for info in words_info:
        if info == 'EOS' or info =='':
            break
        info = info.replace('\t',',')
        info_elms = info.split(',')

        if info_elms[1] not in PART_OF_SEARCH:
            continue
        if to_stem and info_elms[7] != '*':
            words.append(info_elms[7])
        else:
            words.append(info_elms[0])
    return words

#基本形を返す    
def stems(text):
    stems = _split_to_word(text,to_stem=True)
    return stems

#表層形を返す
def terms(text):
    terms = _split_to_word(text,to_stem=False)
    return terms

def normalzie(x, amin=-1, amax=1):
    xmax = x.max()
    xmin = x.min()
    if xmin == xmax:
        return np.ones_like(x)
    return (amax - amin) * (x - xmin) / (xmax - xmin) + amin

if __name__ == '__main__':
    text = "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。"
    print(stems(text))
    print(terms(text))
    
    test_arr = np.array([10,20,30])
    print(normalzie(test_arr))