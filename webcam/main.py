import os
import sys
sys.path.insert(0, "../../virtual_trainer")

import torch
from openpose.model import get_model
from flask import Flask, render_template, Response
from camera import VideoCamera
from webcam_model import ModelClass
import numpy as np

app = Flask(__name__)

CHECKPATH = 'Virtual_trainer/checkpoint'

# Data mountpoint
DATAPOINT = "Virtual_trainer/Data"

# --- Datasets ---
# H36M Ground truths
h36m_file = os.path.join(DATAPOINT,'Keypoints','data_2d_h36m_gt.npz')



# --- Parameters ---
batch_size = 2048
epochs = 20
embedding_len = 128
lr, lr_decay = 0.001 , 0.95 
split_ratio = 0.2

# --- H36M pretrained model settings ---
# checkpoint file
chk_filename = os.path.join(DATAPOINT,'BaseModels', 'epoch_45.bin')
# model architecture
filter_widths = [3,3,3]
channels = 1024
in_joints, in_dims, out_joints = 17, 2, 17
weight_name = '../../virtual_trainer/openpose/weights/openpose_mpii_best.pth.tar'
model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name)['state_dict'])
model = torch.nn.DataParallel(model)
model = model.cuda()
model.float()

camera_index = 2
predict_cl = ModelClass(model,camera_index)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/openpose')
def openpose():
    return render_template('openpose.html')

@app.route('/vp3d')
def vp3d():
    predict_cl.vp3d_recipe2()
    return render_template('vp3d.html',prediction = predict_cl.prediction, rating=predict_cl.rating)

@app.route('/vp3d2')
def vp3d2():
    array,n_frames = predict_cl.vp3d_model()
    array = array.tolist()
#     array = np.load('/media/artursil/backup/class_1.npy')
#     array *= np.array([1,-1,1])
#     print(array)
#     print(array.shape)
#     n_frames = array.shape[0]
#     array = array.flatten().tolist()
#     print(array)
    return render_template('rend_skel.html',prediction = predict_cl.prediction, array_3d=array,n_frames=n_frames)
#     return render_template('rend_skel.html',prediction = 0, array_3d=array,n_frames=n_frames)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_model(predict_cl):
    while True:
        frame = predict_cl.get_keypoints2()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
         
    return Response(gen(predict_cl),
#     return Response(gen(VideoCamera(camera_index)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/openpose_feed')
def openpose_feed():
#     predict_cl = OpenPose(model,camera_index)  
    return Response(gen_model(predict_cl),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='localhost', threaded=True)
