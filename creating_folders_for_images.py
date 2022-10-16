import os
import pandas as pd
import numpy as np
import random as rnd
import shutil
import sys
import glob
import cv2
import shutil
from functools import reduce

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
frame_rate = 10
def skeleton_tracker1(v, destination, participantName):
    # Open data file

    #fps = v.get(cv2.CAP_PROP_FPS)
    total = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    #fps = 25
    clip = rnd.randint(1500,3000)
    frame_seq = clip
    frame_no = (frame_seq/(total))
    v.set(1,frame_no)
    #v.set(cv2.CAP_PROP_POS_COUNT, 1)


    frameCounter = 0
    count = 0
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    cnt = clip
    while(1):
        #for j in range(10):
        ret ,frame = v.read() # read frames
        if ret == False:
            return
        cnt+=1
        if cnt%10 != 0: 
            continue
        if type(detect_one_face(frame)) == bool and not detect_one_face(frame):
            continue
        

        if cnt <= clip+1000:

            x,y,w,h = detect_one_face(frame)
            if w < 200 and h < 200 :
                continue
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_color = frame[y:y+h, x:x+w]
            croppedImg = roi_color
            if(croppedImg.shape[0]<=0 or croppedImg.shape[1]<=0):
                frameCounter = frameCounter + 1
                continue
            #cv2.imshow('img',frame)
            #cv2.imshow('img1',roi_color)
            data_name = destination+"/"+participantName+"_"+str(frameCounter)+".jpg"
            cv2.imwrite(data_name, frame[y:y+h,x:x+w])
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            frameCounter = frameCounter + 1
        else:
            break


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return False
    return faces[0]


dest_directory="output/frames"
base_directory="input"

for fold_folder in glob.glob(base_directory+"/*"):
    for subFolder in glob.glob(fold_folder+"/*"):
        for video_clips in glob.glob(subFolder+"/*"):
            destination_folder = os.path.join(dest_directory, subFolder.split('/')[-1] + "_" + video_clips.split("/")[-1].split(".")[0])
            os.mkdir(destination_folder)
            video = cv2.VideoCapture(video_clips)
            skeleton_tracker1(video, destination_folder, subFolder.split("/")[-1])


