# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:55:05 2019

@author: AI & ML
"""
import numpy as np
import socketio     #creating a tannel btw python environment and simulator
import eventlet    #it is used for handling the event which is used by server and tunnel
from flask import Flask      #used to create a tunnel     
from keras.models import load_model     
import base64                  #used for reading a image from byte
from io import BytesIO         #used for reading a image from byte
from PIL import Image          #it is used for taking out image from python
import cv2 

sio = socketio.Server()

app = Flask(__name__) #'__main__'

speed_limit = 30

def img_preprocess(img):
  img = img[60:135,:,:]                 
  img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img, (3,3),0)
  img = cv2.resize(img,(200,66))
  img = img/255
  
  return img



@sio.on('telemetry')            #@ means class decorator,when connection will connect then it will call telementory function  

def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle,speed))
    send_control(steering_angle,throttle)         #after the prepossing telementory send data to send_control,then send_control sends data to simulator and run the car and send the data to the python environment

@sio.on('connect')#message, disconnect

def connect(sid, environ):
    print('connected')
    send_control(0,0)
    
def send_control(steering_angle,throttle):         #sending the throttle value to simulator
    sio.emit('steer',data={
            'steering_angle':steering_angle.__str__(),
            'throttle':throttle.__str__()
            })
    

if __name__=='__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio,app)          #app is the server,through the server the tunnel will be opened
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)       #blank space is IPV4 address(it may change but by default it take 0.0.0.0),4567 is the server port id
