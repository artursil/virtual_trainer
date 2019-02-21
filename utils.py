import os
import cv2
import pafy
import youtube_dl
import re
import json
import pandas as pd
import numpy as np
from moviepy.editor import *
from slugify import slugify

from itertools import chain
from langdetect import detect


def get_config_file(path):
    _path=path.replace("/txt_files","")
    with open(f'{_path}/config_file.json', encoding="utf8") as json_file:
        _data = json.load(json_file)
    return _data   

def get_stop_words(path,exercise):
    # TODO After testing delete stop_words
    # with open(f'{path}/stop_words.json', encoding="utf8") as json_file:
    #     _data = json.load(json_file)
    _data = get_config_file(path)
    _data = _data[exercise]
    exercises_d = _data['exercises']
    exercises = list(chain(*[v for k,v in exercises_d.items() if k!=exercise]))
    stop_words = _data['stop_words']
    return exercises, stop_words

def search_reg(title):
    reg1 =r"\d+[a-zA-Z]{0,3}\s{0,1}x\s{0,1}\d+[a-zA-Z]{0,3}"
    reg2 = r"\d+\s{0,1}rep"
    reg3 = r"\d+[a-zA-Z]{0,3}\s{0,1}for\s{0,1}\d+[a-zA-Z]{0,3}"
    m = re.search(f"{reg1}|{reg2}|{reg3}",title)
    return m

def get_reps(m,text):
    splitter = re.search(r"[a-zA-z]+\s{0,1}[a-zA-z]{0,3}",m.group(0)).group(0)
    m_ls = m.group(0).split(splitter)
    m_ls_int = [int(re.sub("[a-zA-Z]{0,3}","",x).replace(" ","")) for x in m_ls if re.sub("[a-zA-Z]{0,3}","",x).replace(" ","")!='']
    max_n = max(m_ls_int)
    weight=None
    mu = None
    # Cases where sets x reps
    if max_n<25:
        # Sets usually come first, reps second
        reps_n=m_ls_int[-1]
    else:
        # If one number greater than 25 probobaly is a weight not a set
        reps_n = min(m_ls_int)
        weight = max(m_ls_int)
        regx = r"(?<={})lbs".format(weight)
        if re.search(regx,text):
            mu = "lbs"
        else:
            mu= "kg"
    return reps_n, weight, mu

def get_reps_wrapper(title):
    m = search_reg(title)
    if m:
        reps_n = get_reps(m)
    else:
        reps_n=1
    return reps_n


def analyze_text(text,stop_words):
    _text = text.lower()
    txt_list = _text.split()
    if any(sw in txt_list for sw in stop_words):
        return False
    else:
        return True

def only_tags(text):
    ot = False
    if len(text)>0:
        w_list = text.split()
        tags_n = sum([x.find("#")>-1 for x in w_list])
        if tags_n==len(w_list):
            ot=True
    return ot

def detect_lang(text):
    try:
        return detect(text)
    except:
        if only_tags(text):
            return "en"
        else:
            return ""

def get_weight(text):
    weight = None
    mu = None
    reg1 = r"\d{2,3}(?=\s{0,1}lb)"
    reg2 = r"\d{2,3}(?=\s{0,1}kg)"
    m = re.search(f"{reg1}|{reg2}",text)
    if m:
        weight = int(m.group(0))
        regx = r"(?<={})lbs".format(weight)
        if re.search(regx,text):
            mu = "lbs"
        else:
            mu= "kg"
    return weight, mu

def get_lbs(kg):
    return kg*2.2046

def check_max_weight(weight,mu,max_kg):
    max_lbs = get_lbs(max_kg)
    weight_g = False
    if weight!=None:
        if mu=="kg":
            if weight<max_kg:
                weight_g = True
        else:
            if weight<max_lbs:
                weight_g=True
    return weight_g

def create_metadata_df(path,filename):
    if os.path.isfile(f"{path}/{filename}"):
        metadata_df = pd.read_csv(f"{path}/{filename}")
    else:
        metadata_df = pd.DataFrame()
    return metadata_df


def timer(func):
    def timer_wrapper(*args,**kwargs):
        st=time.time()
        func(*args,**kwargs)
        print(f'{func.__name__} time: {time.time()-st}')
        return func(*args,**kwargs)
    return timer_wrapper