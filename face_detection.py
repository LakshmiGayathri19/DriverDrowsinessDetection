import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from glob import glob

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#img = int(sys.argv[4])
def help_message():
   print("Usage: [Question_Number] [Input_Video] [data_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("[Input_Video]")
   print("Path to the input video")
   print("[data_Directory]")
   print("data directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return False
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

# Camshift
frame_rate = 10
def skeleton_tracker1(v, file_name, destination, participantName):
    # Open data file
    data_name = destination + file_name
    data = open(data_name,"w")

    #fps = v.get(cv2.CAP_PROP_FPS)
    total = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    #fps = 25
    clip = rnd.randint(1500,3000)
    frame_seq = clip
    frame_no = (frame_seq/(total))
    v.set(1,frame_no)
    #v.set(cv2.CAP_PROP_POS_COUNT, 1)


    frameCounter = 0
    # read first frame
    #for j in np.arange(10):
    # ret ,frame = v.read()
    # if ret == False:
    #     return
    
    # detect face in first frame 
    # if type(detect_one_face(frame)) == bool and not detect_one_face(frame):
    #     c,r,w,h = (0,0,0,0)
    # else:
    #     c,r,w,h = detect_one_face(frame)
    # pt = (str(0)+"_"+participantName,c+w/2,r+h/2)
    # # Write track point for first frame
    # data.write("%s,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    #frameCounter = frameCounter + 1

    # set the initial tracking window
    #track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    #roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
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
        #if cnt >= 750 and cnt <= 14250:

        #if cnt >= clip:
        if cnt <= clip+1000:
            #hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            #dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            #ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            x,y,w,h = detect_one_face(frame)
            if w < 200 and h < 200 :
                continue
            # write the result to the file data
            pt = (str(frameCounter)+"_"+participantName,x+w/2,y+h/2)
            data.write("%s,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_color = frame[y:y+h, x:x+w]
            croppedImg = roi_color
            if(croppedImg.shape[0]<=0 or croppedImg.shape[1]<=0):
                frameCounter = frameCounter + 1
                continue
            #cv2.imshow('img',frame)
            #cv2.imshow('img1',roi_color)
            data_name = destination+str(frameCounter)+"_"+participantName+".jpg"
            cv2.imwrite(data_name, frame[y:y+h,x:x+w])
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            frameCounter = frameCounter + 1
        else:
            break

    data.close()


if __name__ == '__main__':
    # question_number = -1

    # question_number = int(sys.argv[1])
    # #if (question_number > 100 or question_number < 1):
    #     #print("Input parameters out of bound ...")
    #     #sys.exit()

    # # read video file
    #print(sys.argv[2])
    input_directory = "input/*"
    output_directory = "output/"
    folders = glob(input_directory) #fold_part1,fold_part2..
    for folder in folders:
        if not os.path.isdir(output_directory + "/alert"):
                os.mkdir(output_directory + "/alert")
        if not os.path.isdir(output_directory + "/low_vigilant"):
                os.mkdir(output_directory + "/low_vigilant")
        if not os.path.isdir(output_directory + "/drowsy"):
                os.mkdir(output_directory + "/drowsy")
        video_folders = glob(folder + "/*")
        for video_folder in video_folders:#01,02,..
                videos = glob(video_folder + "/*")#01,05,10
                for videoClip in videos:
    	            videoClipName = (int)(videoClip.split("/")[-1].split(".")[0])
    	            destination_folder =""
    	            if videoClipName>=0 and videoClipName<=2:
      	               destination_folder = os.path.join(output_directory, "alert/") 
    	            if videoClipName>=3 and videoClipName<=5:
      	               destination_folder = os.path.join(output_directory, "low_vigilant/") 
    	            if videoClipName>=6:
      	               destination_folder = os.path.join(output_directory, "drowsy/")
    	            video = cv2.VideoCapture(videoClip)
    	            skeleton_tracker1(video, "data_camshift.txt", destination_folder, video_folder.split("/")[-1])
