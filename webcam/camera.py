import cv2
import requests
import numpy as np

class VideoCamera(object):
    def __init__(self,index):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(index)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        print(type(image))
        print(image.shape)
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


        # # img_res = requests.get("http://192.168.2.107:8081/shot.jpg")
        # img_res = requests.get("https://www.nerdfitness.com/wp-content/uploads/2014/07/deadlift_faults2.jpg")

        # img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
        # img = cv2.imdecode(img_arr,-1)
        # ret, jpeg = cv2.imencode('.jpg', img)

        # return jpeg.tobytes()
