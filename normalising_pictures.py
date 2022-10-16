from glob import glob
import os
import cv2 
import numpy as np
from matplotlib import pyplot

base_dir="data"
test_dir=base_dir+"/testing/"
train_dir=base_dir+"/training/"
valid_dir=base_dir+"/validation/"

#NORMALISING

for folder in [train_dir,valid_dir]:
    #print(folder)
    for label_type in glob(os.path.join(folder,"*")):
        #print(label_type)
        for img_file in glob(os.path.join(label_type,"*")):
            resized_img = cv2.resize(cv2.imread(img_file),(299,299))
            image = np.array(resized_img.astype(np.float))/np.max(resized_img)
            print(image)
            pyplot.imsave(img_file,image)
            #cv2.imshow(img_file, image)
            #img_file = img_file.split(".")[0] + ".png"
            #pyplot.imsave(img_file,image)
            #print(image)
            #cv2.waitKey(0)
            #closing all open windows
            #cv2.destroyAllWindows()