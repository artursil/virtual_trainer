import cv2
import numpy as np
from openpose.mpii_config import *

def getKeypoints(probMap, threshold=0.1):
    
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    mapMask = mapMask.reshape(mapMask.shape[0],mapMask.shape[1],1)
#    mapMask2 = np.repeat(mapMask,3,2)
#    mapMask = mapMask.reshape(1,mapMask.shape[0],mapMask.shape[1])
#    mapMask2 = np.repeat(mapMask,3,0)
#    plt.imshow(mapMask2)
#    plt.show()
    #find the blobs
    _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask.reshape(blobMask.shape[0],blobMask.shape[1])
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


def getValidPairs(output,detected_keypoints,frameWidth, frameHeight):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(MAP_IDX)):
        # A->B constitute a limb
        pafA = output[0, MAP_IDX[k][0], :, :]
        pafB = output[0, MAP_IDX[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB 
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid
        
        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                            pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)
                    
                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair  
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:            
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs
    
def number_of_people(detected_keypoints):
    max_p=0
    num_points=0
    for l in range(len(detected_keypoints)):
        dk = detected_keypoints[l]
        if len(dk)>max_p:
            max_p=len(dk)
            num_points=1
            len_list =[l]
        elif len(dk)==max_p:
            num_points+=1
            len_list.append(l)
        else:
            pass
            
    if num_points<=5 and max_p>1:
        for i in len_list:
            del detected_keypoints[i][max_p-1]
    return detected_keypoints      
        
            

def getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list):
    # the last number in each row is the overall score 
    personwiseKeypoints = -1 * np.ones((0, 16))

    for k in range(len(MAP_IDX)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])): 
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score 
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


def more_keypoints(personwiseKeypoints,kp_threshold):
    n_plp = len(personwiseKeypoints)
    num_kp=np.empty(shape=(n_plp))
    sum_big = []
    for n in range(n_plp):
        kp_sum = sum(personwiseKeypoints[n]==-1)
        num_kp[n] = kp_sum
    inds = np.nonzero(num_kp>=kp_threshold)[0]
    if inds.shape[0]==0:
        return personwiseKeypoints[[np.argmax(num_kp)]]
    else:
        return personwiseKeypoints[list(inds)]


def bigger_person(personwiseKeypoints,keypoints_list,frameWidth, frameHeight):
    n_plp = len(personwiseKeypoints)
    norms= [0]*n_plp
    for i in range(14):
        skip=False
        n_skip=0
        for n in range(n_plp):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                skip=True
                n_skip+=1
        if skip==True or n_skip>8:
            continue        
        for n in range(n_plp):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            A = np.int32(keypoints_list[index.astype(int)][0][:2])
            B = np.int32(keypoints_list[index.astype(int)][1][:2])
            norms[n] = (norms[n]+np.linalg.norm(B-A))/(i+1)
    norms_sorted = list(reversed(np.argsort(norms)))
    to_delete = []
    for n in range(len(norms)-1):
        if norms[norms_sorted[0]]>norms[norms_sorted[n+1]]*1.8:
            to_delete.append(norms_sorted[n+1])
    for d in reversed(sorted(to_delete)):
        del norms_sorted[d]
        
    personwiseKeypoints = personwiseKeypoints[norms_sorted,:]
    return personwiseKeypoints.reshape(-1,16) 


def detect_spotters(personwiseKeypoints):
    n_plp = len(personwiseKeypoints)
    max_conf = 0 
    for n in range(n_plp):
        if personwiseKeypoints[n][-1]>max_conf:
            max_conf = personwiseKeypoints[n][-1]
    for n in range(n_plp):
        if personwiseKeypoints[n][-1]!=max_conf:
            if ((personwiseKeypoints[n][-1]-max_conf)/max_conf)<=0.2:
                return True
    return False        
    

            
def head_middle(personwiseKeypoints,keypoints_list,frameWidth, frameHeight):       
    n_plp = len(personwiseKeypoints)
    middle_arr = -1 * np.ones((n_plp,2),dtype=np.int32)
    for n in range(n_plp):
        index = int(personwiseKeypoints[n][np.array(POSE_PAIRS[0])][0])
        head = np.int32(keypoints_list[index][:2])
        ff = np.array([frameWidth,frameHeight]) - head
        distances = head*head+ff*ff
        middle_arr[n] = distances
    middle_arr = middle_arr.T.reshape(1,-1)
    widths = middle_arr[0][:n_plp]
    heights =middle_arr[0][n_plp:]
    widths = widths/min(widths)
    heights = heights/max(heights)
    person_index = np.argmin(np.abs((widths+heights)/2-1))
    personwiseKeypoints = personwiseKeypoints[person_index]

    return personwiseKeypoints.reshape(-1,16)


def save_picture(path,personwiseKeypoints,keypoints_list,frameClone):
    for i in range(14):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), COLORS[i], 3, cv2.LINE_AA)        
    cv2.imwrite(path,frameClone)


def draw_interpolated(PATH,ix,frame_c,outputs_df,frame,filename,desc="inter"):
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
    columns = ['0_0', '0_1', '1_0',
           '1_1', '2_0', '2_1', '3_0', '3_1', '4_0', '4_1', '5_0', '5_1', '6_0',
           '6_1', '7_0', '7_1', '8_0', '8_1', '9_0', '9_1', '10_0', '10_1', '11_0',
           '11_1', '12_0', '12_1', '13_0', '13_1', '14_0', '14_1','15_0', '15_1','16_0', '16_1']
    df = outputs_df.copy()
    df = df[df["vid_nr"]==ix]
    try:
        for point in range(16):
            pointsA = (int(df.iloc[frame_c].loc[columns[point*2]]),int(df.iloc[frame_c].loc[columns[point*2+1]]))
        
            if pointsA:
                cv2.circle(frame, pointsA, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    
    
        cv2.imwrite(f"{PATH.replace('clipped','processed')}/{filename}_{ix}_{frame_c}_{desc}.png",frame)
    except ValueError:
        pass


def interpolate(df):
    columns = ['0_0', '0_1', '1_0',
           '1_1', '2_0', '2_1', '3_0', '3_1', '4_0', '4_1', '5_0', '5_1', '6_0',
           '6_1', '7_0', '7_1', '8_0', '8_1', '9_0', '9_1', '10_0', '10_1', '11_0',
           '11_1', '12_0', '12_1', '13_0', '13_1', '14_0', '14_1']
    
    for c in range(len(columns)//2):
        col= columns[c*2]
        col2 = columns[c*2+1]
        avg = np.mean(df[col])
        std = np.std(df[col])
        df.loc[np.abs(df[col] -avg) > 2*std,[col,col2]] = np.nan
        
        # feet shouldnt be moving that's why they are treated differently 
        # -> interval with biggest amount of obs -> values that are in this interval 
        # -> mean of those values -> replace all the values with new one
        if c in [10,13]:
            x_cord = str(c) + "_0"
            y_cord = str(c) + "_1"
            df[x_cord+"_orig"] = df[x_cord]
            df[y_cord+"_orig"] = df[y_cord]
            try:
                new_x = int(np.mean(df.loc[[clip in df[x_cord].value_counts(bins=3).iloc[[0]].index[0]
                            for clip in df[x_cord]],x_cord]))

                new_y = int(np.mean(df.loc[[clip in df[y_cord].value_counts(bins=3).iloc[[0]].index[0]
                            for clip in df[y_cord]],y_cord]))
            except ValueError:
                df[x_cord] = np.nan
                df[y_cord] = np.nan               
            else:
                df[x_cord] = new_x
                df[y_cord] = new_y

        
    df.loc[df["num_keypoints"]<=10,columns]= np.nan
    df.interpolate(inplace=True)

    # additional keypoint between neck and head (15_0,15_1)
    df["15_0"] = np.floor(df[["0_0","1_0"]].min(axis=1) + np.abs(df["0_0"]-df["1_0"])/2)
    df["15_1"] = np.floor(df[["0_1","1_1"]].min(axis=1) + np.abs(df["0_1"]-df["1_1"])/2)
    # additional keypoint between left hip and right hip
    df["16_0"] = np.floor(df[["10_0","13_0"]].min(axis=1) + np.abs(df["10_0"]-df["13_0"])/2)
    df["16_1"] = np.floor(df[["10_1","13_1"]].min(axis=1) + np.abs(df["10_1"]-df["13_1"])/2)

    return df