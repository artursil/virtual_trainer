"""
@author: Ahmad Kurdi
Sequence generators for Virtual Trainer portfolio project.
DSR portfolio project with Artur Silicki
Simplified adaption of VideoPose3D generator classes
"""
import numpy as np
from itertools import zip_longest

class SequenceGenerator:
    """
    Non-batched sequence generator, used for testing.
     
    Arguments:
    actions --  list of actions
    poses -- list of input keypoints, one element for each video
    pad -- input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    """
    
    def __init__(self, actions, poses, pad=0, causal_shift=0):
        assert len(actions) == len(poses)
        self.pad = pad
        self.causal_shift = causal_shift
        self.poses = poses
        self.actions = actions
        
    def num_frames(self):
        count = 0
        for p in self.poses:
            count += p.shape[0]
        return count

    def next_epoch(self):
        for seq_act, seq_poses in zip_longest(self.actions, self.poses):
            batch_act = np.expand_dims(seq_act, axis=0)
            batch_poses = np.expand_dims(np.pad(seq_poses,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            
            yield batch_act, batch_poses


class BatchedGenerator(SequenceGenerator):
    """
    Batched version of sequence data generator

    Additional args:
    batch_size -- Training batch size
    shuffle -- Shuffle between epochs

    """
    def __init__(self, batch_size, actions, poses, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234, endless=False):

        super().__init__(actions, poses, pad, causal_shift)

    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame) tuples
        for i in range(len(poses)):
            n_chunks = poses[i].shape[0]
            bounds = np.arange(n_chunks+1)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:])
     
        self.batch_act = np.empty((batch_size, 1, actions[0].shape[-1]))
        self.batch_poses = np.empty((batch_size, 1+2*pad, poses[0].shape[-2], poses[0].shape[-1]))     
        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.endless = endless
        self.state = None
        
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
         
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_act, end_act) in enumerate(chunks):
                    start_poses = start_act - self.pad - self.causal_shift
                    end_poses = end_act + self.pad - self.causal_shift

                    #poses
                    seq_poses = self.poses[seq_i]
                    low_poses = max(start_poses, 0)
                    high_poses = min(end_poses, seq_poses.shape[0])
                    pad_left_poses = low_poses - start_poses
                    pad_right_poses = end_poses - high_poses
                    if pad_left_poses != 0 or pad_right_poses != 0:
                        self.batch_poses[i] = np.pad(seq_poses[low_poses:high_poses], ((pad_left_poses, pad_right_poses), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_poses[i] = seq_poses[low_poses:high_poses]
                    
                    # actions                   
                    seq_act = self.actions[seq_i]
                    low_act = max(start_act, 0)
                    high_act = min(end_act, seq_act.shape[0])
                    pad_left_act = low_act - start_act
                    pad_right_act = end_act - high_act
                    if pad_left_act != 0 or pad_right_act != 0:
                        self.batch_act[i] = np.pad(seq_act[low_act:high_act], ((pad_left_act, pad_right_act), (0, 0)), 'edge')
                    else:
                        self.batch_act[i] = seq_act[low_act:high_act]
  
                if self.endless:
                    self.state = (b_i + 1, pairs)

                yield np.squeeze(self.batch_act[:len(chunks)]), self.batch_poses[:len(chunks)]
            
            if self.endless:
                self.state = None
            else:
                enabled = False