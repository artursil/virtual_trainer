import json
import re
import pandas as pd
from utils import *





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

def merge_metadata(path_json,exercise):
    files = os.listdir(path_json)
    exercise_files = [file for file in files if file.find(exercise)>-1 and file.find("json")>-1]
    json_list = []
    if len(exercise_files)==0:
        raise FileNotFoundError("Firstly download metadata file using create_metadata func")
    else:
        for f in exercise_files:
            print(f"Using: {f} file")
    for file in exercise_files:
        file_path = f"{path_json}/{file}"
        with open(file_path, encoding="utf8") as json_file:  
            _data = json.load(json_file)
        json_list += _data
    return json_list            
    
def filter_posts(path_json,exercise):
    # try:
    #     with open(f'{path_json}/{exercise}.json', encoding="utf8") as json_file:  
    #         data = json.load(json_file)
    # except FileNotFoundError:
    #     raise FileNotFoundError("Firstly download metadata file using create_metadata func")
    data = merge_metadata(path_json,exercise)
    path_config = path_json.replace("/videos","")
    max_kg = get_config_file(path_json.replace("/txt_files",""))[exercise]["max_kg"]
    other_exercises, stop_words = get_stop_words(f"{path_config}",exercise)
    maximum_tags = 15
    insta_df = pd.DataFrame()
    ix=0
    for idx,d in enumerate(data):
        if idx%1000==0:
            print(idx)
        is_video = d['is_video'] 
        if is_video==False:
            continue
        vid = d['id']
        text_container = d['edge_media_to_caption']['edges'] 
        if text_container!=[]:
            text = text_container[0]['node']['text'].replace(":","")
        else:
            text=""
        lang = detect_lang(text)
        tags = [tag.lower() for tag in d.get('tags',[])]
        text_good = analyze_text(text,stop_words+other_exercises)
        len_urls = len(d['urls'])
        len_tags_g = 0<len(tags)<maximum_tags
        tags_good = sum([tag in stop_words+other_exercises for tag in tags])==0
        crawl = lang=="en" and text!="" and len_urls==1 and text_good and tags_good and len_tags_g   
        if crawl:
            m = search_reg(text)
            if m:
                reps_n,weight,mu = get_reps(m,text)
                if weight==None:
                    weight, mu = get_weight(text)
                good_weight = check_max_weight(weight,mu,max_kg)
                filename = f"{ix}_{slugify(text)[:50]}"
                if good_weight:
                    insta_dict = {'id':vid,
                                'text':text,
                                'tags':str(tags),
                                'video_url':d['urls'][0],
                                'video_view_count':d['video_view_count'],
                                'likes':d['edge_liked_by']['count'],
                                'exercise':exercise,
                                'reps':reps_n,
                                'weight':f"{weight} {mu}",
                                'filename':filename
                                
                                }
                    ix+=1
                    insta_df = insta_df.append(pd.DataFrame(insta_dict,index=[0]),ignore_index=True)
                    

    return insta_df

def update_filtered(base_path,exercise):
    """[This function compares list of videos in downloaded_files.csv to a list of physical videos.
        If there are videos that were deleted, downloaded_files is updated and ids of those videos are passed to config file]
    
    Parameters
    ----------
    base_path : [str]
        [path to a instagram folder]
    exercise : [str]
        [name of an exercise]
    
    """
    # D:\Documents_D\data_science\win\virtual_trainer\instagram\txt_files
    down_df = pd.read_csv(f"{base_path}/txt_files/{exercise}_dl_files.csv")
    files = os.listdir(f"{base_path}/videos/{exercise}/")
    files = [file.split(".")[0] for file in files]
    json_file = f'{base_path}/config_file.json'
    with open(json_file, encoding="utf8") as json_file:  
        config_json = json.load(json_file)
    vids_to_exclude = config_json[exercise]["ids_to_exclude"]
    ind_to_drop = []
    #TODO should be vectorized
    for ix,row in down_df.iterrows():
        not_in_files = row['filename'] not in files
        too_long = row['duration']>=50
        t_per_rep = row['duration']/row['n_reps']
        reps_t_long = t_per_rep>=50
        t_many_reps = row['n_reps']>config_json[exercise]["max_reps"]
        cond1 = any([too_long,reps_t_long,t_many_reps]) and ~not_in_files
        cond2 = any([not_in_files,too_long,reps_t_long,t_many_reps])
        if cond1:
            os.remove(f"{row['filepath']}.mp4")
        if cond2:
            ind_to_drop.append(ix)
            vids_to_exclude.append(row['vid'])
            
    config_json[exercise]["ids_to_exclude"] = list(set(vids_to_exclude))
    down_df.drop(ind_to_drop,axis=0,inplace=True)
    down_df.to_csv(f"{base_path}/txt_files/{exercise}_dl_files.csv",index=False)
    config_json[exercise]["ids_to_exclude"] = list(set(vids_to_exclude))
    with open(json_file, "w") as write_file:
        json.dump(config_json, write_file)

