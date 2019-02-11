
THRESHOLD = 0.1

NPOINTS = 15

#Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4,
#Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8,
#Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12,
#Left Ankle – 13, Chest – 14, Background – 15



POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
MAP_IDX = [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], 
            [12,13], [14,15], [16,17], [18,19], [20,21], [22,23], 
            [24,25], [26,27]]

COLORS = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
            [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
            [0,0,255], [255,0,0]]