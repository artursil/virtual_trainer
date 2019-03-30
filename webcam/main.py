import os
import sys
sys.path.insert(0, "../../virtual_trainer")

import torch
from openpose.model import get_model
from flask import Flask, render_template, Response
from camera import VideoCamera
from webcam_model import ModelClass
from model_utils import load_all_models
from multiprocessing import Queue, Event
import json

app = Flask(__name__)
img_q = Queue()
openpose_model, class_model, model_embs, model_rank = load_all_models()
kill_get_kp = False
camera_index = 1
predict_cl = ModelClass(openpose_model, class_model, model_embs, model_rank,camera_index, img_q)

@app.route('/')
def index():
    return render_template('index.html',prediction = predict_cl.prediction, rating=predict_cl.rating)

@app.route('/openpose')
def openpose():
    return render_template('openpose.html')

@app.route('/vp3d')
def vp3d():
    predict_cl.vp3d_recipe2()
#     return render_template('vp3d.html',prediction = predict_cl.prediction, rating=predict_cl.rating)
#     return jsonify(predict_cl.prediction)
#     response = app.response_class(
#         response=json.dumps(predict_cl.prediction),
#         status=200,
#         mimetype='application/json'
#     )
#     return response
    return Response(json.dumps({'prediction':predict_cl.prediction,'rating':predict_cl.rating}),mimetype='application/json')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_model(predict_cl):
    while True:
        frame = predict_cl.get_keypoints()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
         
    return Response(gen(predict_cl),
#     return Response(gen(VideoCamera(camera_index)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# @app.route("/stream")
# def stream():
#     def eventStream():
#         while True:
#         #     prediction = img_q.get()
#             if predict_cl.new_pred:
#                 # yield "data: {}\n\n".format(prediction)
#                 yield "data: {}\n\n".format(predict_cl.prediction)
    
    return Response(eventStream(), mimetype="text/event-stream")
@app.route('/vp3d_feed')
def vp3d_feed():
    print('start')
    predict_cl.vp3d_recipe2()     
    print('stop')
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
