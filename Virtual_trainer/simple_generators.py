"""
Simple generator because why needlessly complicate things
"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch

class SimpleSequenceGenerator:

    def __init__(self, batch_len, actions, poses, pad=0, causal_shift=0, test_split=0.2, random_seed=1234):
        assert len(actions) == len(poses)
        self.pad = pad
        self.causal_shift = causal_shift
        self.poses = [np.pad(p,((pad+causal_shift,pad-causal_shift),(0,0),(0,0)),'edge') for p in poses]
        self.actions = actions
        self.batch_len = batch_len
        self.test_split = test_split
        self.random = np.random.RandomState(random_seed)
        self.receptive_field = 2*pad +1
        self.num_joints = self.poses[0].shape[-2]
        self.dims = self.poses[0].shape[-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.poses, self.actions, stratify=self.actions, test_size=self.test_split)
        self.next_epoch()
    
    def next_epoch(self):
        
        self.build_chunks()

    def build_chunks(self):
        chunk_db = []
        for idx, seq in enumerate(self.X_train):
            clip_len = len(seq) - self.receptive_field
            clip_idx = np.full(clip_len, idx)
            x_ind = range(0, clip_len)
            y_val = np.full(clip_len, self.y_train[idx])
            chunk_db += zip(clip_idx,x_ind,y_val)
        self.chunk_db = self.random.permutation(chunk_db)
        self.num_batches = int(len(chunk_db)//self.batch_len)+1

    def next_batch(self):
        this_batch=0
        b_ix=0
        for b_ix in range(self.num_batches):

            this_batch = min(len(self.chunk_db) - b_ix * self.batch_len, self.batch_len)

            self.batch_X = np.empty( (this_batch, self.receptive_field,self.num_joints,self.dims) )
            self.batch_y = np.empty( (this_batch,1) )

            for i in range(self.batch_len):
                shift = b_ix * self.batch_len + i
                if (shift >= len(self.chunk_db)):
                    break
                clip_idx , x_ind , y_val = self.chunk_db[shift]
                idx_fin = x_ind + self.receptive_field

                self.batch_X[i,:,:,:] = self.X_train[clip_idx][x_ind:idx_fin,:,:]
                self.batch_y[i,:] = y_val
            yield self.batch_y, self.batch_X 

    def next_validation(self):
        for ix, y in enumerate(self.y_test):
            X = self.X_test[ix]
            yield y , np.expand_dims(X,axis=0)

class SimpleSiameseGenerator:

    def __init__(self, batch_len, actions, poses,ratings,model=None,filenames=None, pad=0, causal_shift=0, test_split=0.2, random_seed=1234):
        assert len(actions) == len(poses)
        self.pad = pad
        self.causal_shift = causal_shift
        self.poses_list = [np.pad(p,((pad+causal_shift,pad-causal_shift),(0,0),(0,0)),'edge') for p in poses]
        self.actions = actions
        self.model=model
        self.filenames=filenames
        self.ratings=ratings
        self.batch_len = batch_len
        self.test_split = test_split
        self.random = np.random.RandomState(random_seed)
        self.receptive_field = 2*pad +1
        self.num_joints = self.poses_list[0].shape[-2]
        self.dims = self.poses_list[0].shape[-1]
        self.poses = np.stack((np.array(self.poses_list),np.array(ratings)),axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.poses, self.actions, stratify=self.actions, test_size=self.test_split)
        self.r_train, self.r_test = self.X_train[:,1] , self.X_test[:,1]  
        self.X_train, self.X_test = self.X_train[:,0] , self.X_test[:,0]  
        self.next_epoch()
        self.build_chunks_val()
    
    def next_epoch(self):
        
        self.build_chunks()

    def build_chunks(self):
        chunk_db = []
        for idx, seq in enumerate(self.X_train):
            clip_len = len(seq) - self.receptive_field
            clip_idx = np.full(clip_len, idx)
            x_ind = range(0, clip_len)
            y_val = np.full(clip_len, self.y_train[idx])
            r_val = np.full(clip_len, self.r_train[idx])
            chunk_db += zip(clip_idx,x_ind,y_val,r_val)
        self.chunk_db = self.random.permutation(chunk_db)
        self.num_batches = int(len(chunk_db)//self.batch_len)+1

    def build_chunks_val(self):
        chunk_db = []
        for idx, seq in enumerate(self.X_test):
            clip_len = len(seq) - self.receptive_field
            clip_idx = np.full(clip_len, idx)
            x_ind = range(0, clip_len)
            y_val = np.full(clip_len, self.y_test[idx])
            r_val = np.full(clip_len, self.r_test[idx])
            chunk_db += zip(clip_idx,x_ind,y_val,r_val)
        self.chunk_db_val = self.random.permutation(chunk_db)
        self.num_batches_val = int(len(chunk_db)//self.batch_len)+1

    def next_batch(self):
        this_batch=0
        b_ix=0
        for b_ix in range(self.num_batches):

            this_batch = min(len(self.chunk_db) - b_ix * self.batch_len, self.batch_len)

            self.batch_X = np.empty( (this_batch, self.receptive_field,self.num_joints,self.dims) )
            self.batch_y = np.empty( (this_batch,1) )
            self.batch_r = np.empty( (this_batch,1) )

            for i in range(self.batch_len):
                shift = b_ix * self.batch_len + i
                if (shift >= len(self.chunk_db)):
                    break
                clip_idx , x_ind , y_val, r_val = self.chunk_db[shift]
                clip_idx = int(clip_idx)
                x_ind = int(x_ind)
                idx_fin = x_ind + self.receptive_field

                self.batch_X[i,:,:,:] = self.X_train[clip_idx][x_ind:idx_fin,:,:]
                self.batch_y[i,:] = y_val
                self.batch_r[i,:] = r_val
            yield  self.batch_X, self.batch_y, self.batch_r

    def next_validation(self):
        this_batch=0
        b_ix=0
        for b_ix in range(self.num_batches_val):

            this_batch = min(len(self.chunk_db_val) - b_ix * self.batch_len, self.batch_len)

            self.batch_X = np.empty( (this_batch, self.receptive_field,self.num_joints,self.dims) )
            self.batch_y = np.empty( (this_batch,1) )
            self.batch_r = np.empty( (this_batch,1) )

            for i in range(self.batch_len):
                shift = b_ix * self.batch_len + i
                if (shift >= len(self.chunk_db_val)):
                    break
                clip_idx , x_ind , y_val, r_val = self.chunk_db_val[shift]
                clip_idx = int(clip_idx)
                x_ind = int(x_ind)
                idx_fin = x_ind + self.receptive_field

                self.batch_X[i,:,:,:] = self.X_test[clip_idx][x_ind:idx_fin,:,:]
                self.batch_y[i,:] = y_val
                self.batch_r[i,:] = r_val
            yield  self.batch_X, self.batch_y, self.batch_r


    def build_embeds_dataframe(self,df_filename):
        """
        Run all videos through the model and save the results.
        """
        emb_df = pd.DataFrame(columns=['file','embeddings','rating','exc_class']) 
        if self.model is None:
            raise ValueError('You need to specify model to create embeddings')
        if self.filenames is None:
            raise ValueError('You need to specify filenames to create embeddings')
        if self.ratings is None:
            raise ValueError('You need to specify ratings to create embeddings')
            
        model = self.model
        print(len(self.poses_list))
        for ix,pose in enumerate(self.poses_list):
            if ix%20==0:
                print(ix)
            model.eval()
    #        if pose.shape[0]<= self.receptive_field:
    #            continue
            pose = torch.from_numpy(pose).unsqueeze(0)
            pose = pose.to(dtype=torch.float)
            if torch.cuda.is_available():
                eval_model = model.cuda()
                pose = pose.cuda()  
            pred = eval_model(pose).permute(0,2,1)
            pred = pred.detach().cpu().numpy()
            pred = pred.squeeze()
            filename = self.filenames[ix]
            rating = self.ratings[ix]
            target = self.actions[ix]
            for x in range(pred.shape[0]):
                df_row=dict(file=filename,embeddings = [pred[x]],rating=rating,exc_class=target)
                
                emb_df = emb_df.append(pd.DataFrame(df_row),ignore_index=False)

        emb_df.to_csv(f'Data/{df_filename}')

        return self