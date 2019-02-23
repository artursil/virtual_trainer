"""
Simple generator because why needlessly complicate things
"""
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleSequenceGenerator:

    def __init__(self, batch_len, actions, poses, pad=0, causal_shift=0, test_split=0.2, random_seed=1234):
        assert len(actions) == len(poses)
        self.pad = pad
        self.causal_shift = causal_shift
        self.poses = [np.pad(p,(pad+causal_shift,pad-causal_shift),(0,0),(0,0),'edge') for p in poses]
        self.actions = actions
        self.batch_len = batch_len
        self.test_split = test_split
        self.random = np.random.RandomState(random_seed)
        self.next_epoch()
    
    def next_epoch(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.poses, self.actions, stratify=self.actions, test_size=self.test_split)
        self.build_chunks()

    def build_chunks(self):
        chunk_db = []
        for idx, seq in enumerate(self.X_train):
            clip_len = len(seq)
            clip_idx = np.full(clip_len, idx)
            x_ind = range(0, clip_len)
            y_val = np.full(clip_len, self.y_train)
            chunk_db.append(zip(clip_idx,x_ind,y_val))
        self.chunk_db = self.random.permutation(chunk_db)
        self.num_batches = int(len(chunk_db)//self.batch_len)+1

    def next_batch(self):
        for b_ix in range(self.num_batches):
            batch_X, batch_y = [] , []
            for i in range(self.batch_len):
                shift = b_ix + i
                if (shift >= len(self.chunk_db)):
                    break
                clip_idx , x_ind , y_val = self.chunk_db[shift]
                x = self.X_train[clip_idx][x_ind,:,:]
                batch_X.append(x)
                batch_y.append(y_val)
            yield batch_X , batch_y

    def next_validation(self):
        for ix, y in enumerate(self.y_test):
            X = self.X_test[ix]
            yield X, np.full(len(X)-self.pad, y)
            
            
