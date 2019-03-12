from sklearn.model_selection import train_test_split
import numpy as np
from itertools import compress
import random
import os
import pickle
import torch

def int_f(x):
    return int(float(x))

class SiameseSimpleGenerator:

    def __init__(self, batch_len, actions, poses,ratings,filenames,model,
                 from_clip_flg, pad=0, causal_shift=0, test_split=0.2,
                 n_chunks=8, random_seed=1234, only_rated=True, 
                 oversample=False):
        assert len(actions) == len(poses)
        self.pad = pad
        self.causal_shift = causal_shift
#        self.poses = [np.pad(p,((pad+causal_shift,pad-causal_shift),(0,0),(0,0)),'edge') for p in poses]
        self.poses = poses
        self.actions = actions
        self.ratings = np.array(ratings)
        self.filenames = filenames
        self.model = model
        self.from_clip_flg = np.array(from_clip_flg)
        self.batch_len = batch_len
        self.test_split = test_split
        self.n_chunks = n_chunks +4 # n_chunks for training, 4 for validation
        self.random = np.random.RandomState(random_seed)
        self.receptive_field = 2*pad +1
        self.num_joints = self.poses[0].shape[-2]
        self.dims = self.poses[0].shape[-1]
        self.epoch=0
        self.seed = random_seed
        self.oversample = oversample
        if only_rated:
            self.__only_from_clips()
        self.next_epoch()
     
    def __only_from_clips(self):
        """
        Uses only clips that were manually rated
        """
        self.poses = list(compress(self.poses,list((self.from_clip_flg==True))))
        self.filenames = list(compress(self.filenames,list((self.from_clip_flg==True))))
        self.ratings = self.ratings[self.from_clip_flg==True]
        return self
    
    
    def __get_nines(self):
        """
        Selects clips with perfect form 
        """
        poses = list(compress(self.poses,list((self.ratings==9))))
        ratings = self.ratings[self.ratings==9]
        targets = list(compress(self.actions,list((self.ratings==9))))
        return poses, ratings, targets
        
    def __get_others(self):
        """
        Selects clips with not perfect form
        """
        poses = list(compress(self.poses,list((self.ratings!=9))))
        ratings = self.ratings[self.ratings!=9]
        ratings[ratings<6] = 6 #Change ratings of all clips rated below 6 to 6
        targets = list(compress(self.actions,list((self.ratings!=9))))
        return poses,ratings, targets
        
    def __create_all_combinations(self,range_a,range_b):
        """
        Creates cominations between videos with perfect form
        and all of the videos (including those with perfect form)
        """
        combinations =[]
        for a in range(range_a):
            combinations  += [(a,b) for b in range(range_b) if a!=b]
        # assert len(combinations)== range_a * (range_b-1)
        return combinations
    
    def __create_oversampled_combs(self,range_a,range_b):
        """
        Creates cominations between videos with perfect form
        and all of the videos (including those with perfect form)
        """
        combinations =[]
        for a in range(range_a):
            for b in range(range_b):
                rating = self.ratings_oth[b]
                class1 = self.targets[a]
                class2 = self.targets[b]
                if a!=b and class1==class2:
                    if rating==8:
                        for x in range(3):
                            combinations  += [(a,b)]
                    elif rating==7:
                        for x in range(2):
                            combinations  += [(a,b)]
                    else:
                        combinations  += [(a,b)]
        return combinations

    def __validation_chunks(self,val_size=4):
        """
        Returns list of chunk_ids that are used for validation_set
        """        
        start_val = (self.n_chunks-val_size)//2
        val_chunks = list(range(start_val,start_val+val_size))
        
        return val_chunks
    
    def __get_sets(self):
        """
        Prepare sets that are then used for creating chunks.
        Save all combinations to self.combinations variable.
        Should be run only once.
        """
        self.poses_nines,self.ratings_nines, self.targets_nines= self.__get_nines()
        poses_oth,self.ratings_oth, self.targets_oth = self.__get_others()
        self.poses = self.poses_nines + poses_oth
        self.ratings_oth = np.concatenate([self.ratings_nines,self.ratings_oth])
        self.targets = self.targets_nines + self.targets_oth
        if self.oversample==True:
            self.combinations = self.__create_oversampled_combs(len(self.ratings_nines),
                                                               len(self.ratings_oth))
        else:   
            self.combinations = self.__create_all_combinations(len(self.ratings_nines),
                                                          len(self.ratings_oth))
        return self
        
        
        
    def next_epoch(self):
        
        self.build_chunks()
        self.epoch+=1
        
    def build_all_possible_embds(self):
        """
        Run all videos through the model and save the results.
        """
        if self.epoch==0:
            self.__get_sets()
        if os.path.exists(f'/Data/emb_{len(self.poses)}_dict.pickle'):
            pickle_in = open("Data/dict.pickle","rb")
            self.all_embeddings = pickle.load(pickle_in)
        else:
            all_embeddings = {}
            model = self.model
            print(len(self.poses))
            for ix,pose in enumerate(self.poses):
                if ix%20==0:
                    print(ix)
                model.eval()
                if pose.shape[0]<= self.receptive_field:
                    continue
                pose = torch.from_numpy(pose).unsqueeze(0)
                pose = pose.to(dtype=torch.float)
                if torch.cuda.is_available():
                    eval_model = model.cuda()
                    pose = pose.cuda()  
                pred = eval_model(pose)
                pred = pred.detach().cpu().numpy()
                filename = self.filenames[ix]
                all_embeddings[filename] = pred
                
            pickle_out = open(f'Data/emb_{len(self.poses)}_dict.pickle',"wb")
            pickle.dump(all_embeddings, pickle_out)
            pickle_out.close()
            self.all_embeddings = all_embeddings

        return self

    def build_chunks(self):
        """
        Creates chunks both for training and validation.
        For validation chunks are created only once.
        Chunks from beginning and end of a clip are used for training,
        chunks from the middle are used for validation.
        """
        if self.epoch==0:
            self.__get_sets() # we want to create thoe datasets once       
            self.build_all_possible_embds()
        
        val_chunks = self.__validation_chunks()
        
        chunk_db = []
        chunk_val = []
        for id_a, id_b in self.combinations:
            # There is no padding for videos so there is need to filter them by length.
            if len(self.poses_nines[id_a])<=100 or len(self.poses[id_b])<=100:
                continue
            clips_len= [len(self.poses_nines[id_a])-self.receptive_field,\
                        len(self.poses[id_b])-self.receptive_field]
            big_clip = np.argmax(clips_len) # Choose longer clip
            sm_clip = np.argmin(clips_len)
            clips_diff = np.abs(clips_len[0]-clips_len[1])//2 #Determine how many frames should be added to to longer video
            # This way we are cropping longer video to the size of shorter one by not including first and last frames of length clips_diff

            self.n_frames = clips_len[sm_clip] // self.n_chunks # number of frames per chunk
            n_frames = self.n_frames
            random.seed(self.seed)
            rand_a = [random.randint(0,self.n_frames)\
                      for ch in range(self.n_chunks)] #Random shift of frames in each chunk
            rand_b = [random.randint(0,self.n_frames)\
                      for ch in range(self.n_chunks)]
            
            rating_a = self.ratings_nines[id_a]
            rating_b = self.ratings_oth[id_b]
            filename_a = self.filenames[id_a]
            filename_b = self.filenames[id_b]
            or_a = self.from_clip_flg[id_a]
            or_b = self.from_clip_flg[id_b]
            
            for ch in range(self.n_chunks):
                if ch in val_chunks:
                    if self.epoch==0: # created only once
                        if big_clip==0:
                            chunk = (filename_a, filename_b,
                                     ch*n_frames+(n_frames//2)+clips_diff,
                                     ch*n_frames+(n_frames//2),rating_a,rating_b,or_a,or_b)
                        else:
                            chunk = (filename_a, filename_b,ch*n_frames+(n_frames//2),
                                     ch*n_frames+(n_frames//2)+clips_diff,
                                     rating_a,rating_b,or_a,or_b)
                        chunk_val.append(chunk)
                        
                else:                    
                    if big_clip==0:
                        chunk = (filename_a, filename_b,ch*n_frames+rand_a[ch]+clips_diff,
                                 ch*n_frames+rand_b[ch],rating_a,rating_b,or_a,or_b)
                    else:
                        chunk = (filename_a, filename_b,
                                 min(ch*n_frames+rand_a[ch],clips_len[0]-1),
                                 min(ch*n_frames+rand_b[ch]+clips_diff,clips_len[1]-1),
                                 rating_a,rating_b,or_a,or_b)
                        
                    chunk_db.append(chunk)
        self.chunk_db = self.random.permutation(chunk_db)
        if self.epoch==0:
            self.chunk_val = chunk_val
        self.num_batches = int(len(chunk_db)//self.batch_len)+1                    
            
            

    def next_batch(self):
        this_batch=0
        b_ix=0
        emb_size = next(iter(self.all_embeddings.values())).shape[1]
        for b_ix in range(self.num_batches):

            this_batch = min(len(self.chunk_db) - b_ix * self.batch_len,
                             self.batch_len)

            self.batch_X1 = np.empty( (this_batch, emb_size) )
            self.batch_X2 = np.empty( (this_batch, emb_size) )
            self.batch_y = np.empty( (this_batch,1) )
            info= []
            for i in range(self.batch_len):
                shift = b_ix * self.batch_len + i
                if (shift >= len(self.chunk_db)):
                    break
                clip_id_a,clip_id_b , x1_ind,x2_ind , y1,y2, or_a, or_b= self.chunk_db[shift]
                or_a = str(or_a)
                or_b = str(or_b)
                x1_ind = int(x1_ind)
                x2_ind = int(x2_ind)
                self.batch_X1[i,:] = self.all_embeddings[clip_id_a][0,:,x1_ind]
                self.batch_X2[i,:] = self.all_embeddings[clip_id_b][0,:,x2_ind]
                self.batch_y[i,:] = int_f(y1)-int_f(y2)
                info.append((str(clip_id_a),str(clip_id_b),x1_ind,x2_ind,or_a,
                    or_b))
            yield self.batch_X1, self.batch_X2, self.batch_y, info


    def next_validation(self):
        this_batch=0
        b_ix=0
        num_val_batches = int(len(self.chunk_val)//self.batch_len)+1
        emb_size = next(iter(self.all_embeddings.values())).shape[1]
        for b_ix in range(num_val_batches):

            this_batch = min(len(self.chunk_val) - b_ix * self.batch_len,
                             self.batch_len)

            self.val_X1 = np.empty( (this_batch, emb_size) )
            self.val_X2 = np.empty( (this_batch, emb_size) )
            self.val_y = np.empty( (this_batch,1) )
            info = []
            for i in range(self.batch_len):
                shift = b_ix * self.batch_len + i
                if (shift >= len(self.chunk_val)):
                    break
                clip_id_a,clip_id_b , x1_ind,x2_ind , y1,y2, or_a, or_b = self.chunk_val[shift]
                or_a = str(or_a)
                or_b = str(or_b)
                x1_ind = int(x1_ind)
                x2_ind = int(x2_ind)

                self.val_X1[i,:] = self.all_embeddings[clip_id_a][0,:,x1_ind]
                self.val_X2[i,:] = self.all_embeddings[clip_id_b][0,:,x2_ind]
                self.val_y[i,:] = int_f(y1)-int_f(y2)
                info.append((str(clip_id_a),str(clip_id_b),x1_ind,x2_ind,or_a,
                    or_b))
            yield self.val_X1, self.val_X2, self.val_y, info
