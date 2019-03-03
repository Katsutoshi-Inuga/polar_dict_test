#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 02:04:20 2019

@author: hoge
"""

'''URLを扱うモジュール'''
import urllib.request

import re
import time
import pandas as pd

from bs4 import BeautifulSoup

reexp_ptn=r'^/movie/.*/[0-9]{6}/$'
ptn = re.compile(reexp_ptn)

#https://movies.yahoo.co.jp/movie/の一ページ目のタイトルとURLを取得
def scrape(url):
    #[[title,url],...]
    movie_data=[]
    r = urllib.request.urlopen(url)
    html = r.read()
    parser = "html.parser"
    sp = BeautifulSoup(html, parser)
    for tag in sp.find_all("a"):
        url = tag.get("href")
        title = tag.get("title")
        if url is None or title is None:
            continue
        if ptn.match(url):
            movie_data.append([title,url])
    return movie_data

#作品ページのレビューの星の点数を取得
def selectStartRating(li):
    #i class="star-actived rate-100" -> 100
    star_text=li.find("i",class_="star-actived")
    _,rate_text=star_text.attrs.get('class')
    rate=int(rate_text.replace("rate-",""))
    return rate

#作品ページのレビューのタイトルを取得
def selectCommentTitle(li):
    #span class="color-sub text-small"
    comment_title_text=li.find("span",class_="color-sub text-small")
    return comment_title_text.text

#作品ページのレビューの本文を取得
def selectCommentMain(li):
    #<p class="text-xsmall text-overflow clear">
    comment_title_main=li.find("p",class_="text-xsmall text-overflow clear")
    rs=""
    #コメントが取得できな場合があった（ネタばれのフラグがついてるものは本文がない）
    if comment_title_main is not None:
      rs=comment_title_main.text
    return rs

#各ページ内のレビュー記事を取得する
def getReviewContents(title,url):
    #[Star_rating,CommentTitle,CommentMain]
    review_Data=[]
    r = urllib.request.urlopen(url)
    html = r.read()
    parser = "html.parser"
    sp = BeautifulSoup(html, parser)
    revwlst = sp.find(id="revwlst")
    li_list = revwlst.findAll("a" , class_="listview__element--right-icon" )
    for li in li_list:
        review_Data.append(
                [title.strip(),
                 selectStartRating(li),
                 selectCommentTitle(li).strip(),
                 selectCommentMain(li).strip(),
                 ]
                )
    return review_Data


if __name__ == '__main__':
    '''メイン処理'''
    url="https://movies.yahoo.co.jp/movie/"
    movie_page_url="https://movies.yahoo.co.jp"
    movie_datas = scrape(url)
    
    reviews=[]
    for movie_data in movie_datas:
        movie_by_title_url = movie_page_url + movie_data[1] + "review"
        time.sleep(3)
        reviews.extend(getReviewContents(movie_data[0],movie_by_title_url))
    
    print(reviews[0])
    df = pd.DataFrame(reviews)
    df.to_pickle("./movie_rev.pkl")
