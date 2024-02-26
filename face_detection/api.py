from __future__ import print_function
import os
import torch
import numpy as np
import cv2
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file


ROOT = os.path.dirname(os.path.abspath(__file__))

class FaceAlignment:
    def __init__(self, device='cuda', verbose=False):
        self.device = device
        self.verbose = verbose
        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True
    
    def get_detections_for_batch(self, images):

      face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml') 

      bboxlists = []

      for img in images:
      
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        bboxlist = [] 
        for (x,y,w,h) in faces:  
          box = (x, y, x+w, y+h)
          bboxlist.append(box)
        # print(bboxlist)
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]
        bboxlists.append(bboxlist)
      
      return bboxlists
