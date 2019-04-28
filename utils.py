import os
import cv2
#import pafy
import youtube_dl
import re
import json
import pandas as pd
import numpy as np
import sqlite3
from drive_utils import *
from moviepy.editor import *
#from slugify import slugify

from itertools import chain
from langdetect import detect 
from langdetect.lang_detect_exception import LangDetectException


EXC_INTER_FEET = ['deadlift','squat','bodyweightsquats']

# def to_database(func,db_path):
#     def wrapper():
#         func()
#         #TODO pass to database
def save2db(db_path,df,table_name,append=False,schema=None):
    if db_path.find('instagram')>-1:
        db_path = db_path.replace('instagram','').replace('//','/')
    database = f'{db_path}/vtdb.db'
    download_db(database)
    conn = sqlite3.connect(database)
    if append==False:
        ifexists = 'replace'
    else:
        ifexists = 'append'
    df.to_sql(table_name,conn,schema=schema,if_exists=ifexists)
    upload_db(database)

def get_config_file(path):
    _path=path.replace("/txt_files","")
    with open(f'{_path}/config_file.json', encoding="utf8") as json_file:
        _data = json.load(json_file)
    return _data   

def get_stop_words(path,exercise):
    """get_stop_words

    Returns list of other exercises names and other words from config_file.json
    """
    # TODO After testing delete stop_words
    # with open(f'{path}/stop_words.json', encoding="utf8") as json_file:
    #     _data = json.load(json_file)
    _data = get_config_file(path)
    _data = _data[exercise]
    exercises_d = _data['exercises']
    exercises = list(chain(*[v for k,v in exercises_d.items() if k!=exercise]))
    stop_words = _data['stop_words']
    return exercises, stop_words

def get_text(container):
    text_container = container['edges'] 
    if text_container!=[]:
        text = text_container[0]['node']['text'].replace(":","")
    else:
        text=""
    return text

def search_reg(title):
    """search_reg

    Returns information about number of reps and weight.
    """
    reg1 =r"\d+[a-zA-Z]{0,3}\s{0,1}x\s{0,1}\d+[a-zA-Z]{0,3}"
    reg2 = r"\d+\s{0,1}rep"
    reg3 = r"\d+[a-zA-Z]{0,3}\s{0,1}for\s{0,1}\d+[a-zA-Z]{0,3}"
    reg4 = r"\d+[a-zA-Z]{0,3}\s{0,1}of\s{0,1}\d+[a-zA-Z]{0,3}"
    m = re.search(f"{reg1}|{reg2}|{reg3}|{reg4}",title)
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
    """analyze_text

    Returns flag if text contains stop words.
    """
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
    except LangDetectException:
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
