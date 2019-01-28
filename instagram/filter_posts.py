import json
import re
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

            

    
def filter_posts(path_json,exercise,max_kg):
    with open(f'{path_json}/{exercise}.json', encoding="utf8") as json_file:  
        data = json.load(json_file)
    max_lbs = get_lbs(max_kg)
    other_exercises, stop_words = get_stop_words("D:/A_Documents/virtual_trainer/instagram/",exercise)
    maximum_tags = 15
    insta_df = pd.DataFrame()

    for idx,d in tqdm(enumerate(data)):
        is_video = d['is_video'] 
        if is_video==False:
            continue
        vid = d['id']
        text_container = d['edge_media_to_caption']['edges'] 
        if text_container!=[]:
            text = text_container[0]['node']['text']
        else:
            text=""
        lang = detect_lang(text)
        tags = d.get('tags',[])
        text_good = analyze_text(text,stop_words+other_exercises)
        len_urls = len(d['urls'])
        len_tags_g = len(tags)<maximum_tags
        tags_good = tags not in stop_words+other_exercises
        crawl = lang=="en" and text!="" and len_urls==1 and text_good and tags_good and len_tags_g   
        if crawl:
            m = search_reg(text)
            if m:
                reps_n,weight,mu = get_reps(m,text)
                if weight==None:
                    weight, mu = get_weight(text)
                good_weight = check_max_weight(weight,mu,max_kg)
                if good_weight:
                    insta_dict = {'id':vid,
                                'text':text,
                                'tags':str(tags),
                                'video_url':d['urls'][0],
                                'video_view_count':d['video_view_count'],
                                'likes':d['edge_liked_by']['count'],
                                'exercise':exercise,
                                'reps':reps_n,
                                'weight':f"{weight} {mu}"
                                
                                }
                    insta_df = insta_df.append(pd.DataFrame(insta_dict,index=[0]),ignore_index=True)


    return insta_df
