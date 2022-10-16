import os
import pandas as pd
import numpy as np
import random as rnd
import shutil
import sys
from glob import glob
import shutil
from functools import reduce

base_directory="output/"

alert = np.array([]) #contains fileNames of alert
drowsy = np.array([])
low_vigilant = np.array([])

for root, dirs, files in os.walk(os.path.join(os.getcwd(),base_directory,"drowsy")):
    for file_name in files:
        if not file_name.endswith('.txt'):
            drowsy = np.append(drowsy,file_name)
            

for root, dirs, files in os.walk(os.path.join(os.getcwd(),base_directory,"alert")):
    for file_name in files:
        if not file_name.endswith('.txt'):
            alert = np.append(alert,file_name)


for root, dirs, files in os.walk(os.path.join(os.getcwd(),base_directory,"low_vigilant")):
    for file_name in files:
        if not file_name.endswith('.txt'):
            low_vigilant = np.append(low_vigilant,file_name)

rnd.shuffle(alert)
rnd.shuffle(drowsy)
rnd.shuffle(low_vigilant)
print(alert)

print("###############\n")
print("Creating 'data' directory (Will delete if already exists!)...")
dir = os.path.join(os.getcwd(),"data")
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(dir)
print("Created 'data' directory successfully!")

print("\nCreating 'training' 'validation'and 'testing' directory inside 'data' directory...\n")
traindir=os.path.join(dir,"training")
os.mkdir(traindir)
validdir=os.path.join(dir,"validation")
os.mkdir(validdir)
testdir=os.path.join(dir,"testing")
os.mkdir(testdir)
seqdir=os.path.join(dir,"sequences")
os.mkdir(seqdir)
seqtrain=os.path.join(dir,"sequences", "training")
os.mkdir(seqtrain)
seqvalidation=os.path.join(dir,"sequences", "validation")
os.mkdir(seqvalidation)
seqtest=os.path.join(dir,"sequences", "testing")
os.mkdir(seqtest)
print("Created directories successfully!")

print("\nCreating 'Alert', low_vigilant and 'Drowsy' directory inside 'training', validation and 'testing' directories...\n")
os.mkdir(os.path.join(traindir,"alert"))
os.mkdir(os.path.join(validdir,"alert"))
os.mkdir(os.path.join(testdir,"alert"))
os.mkdir(os.path.join(traindir,"drowsy"))
os.mkdir(os.path.join(validdir,"drowsy"))
os.mkdir(os.path.join(testdir,"drowsy"))
os.mkdir(os.path.join(traindir,"low_vigilant"))
os.mkdir(os.path.join(validdir,"low_vigilant"))
os.mkdir(os.path.join(testdir,"low_vigilant"))
print("Created directories successfully!")

#alert_frames=3944 drowsy_frames=3976 lowVigliant=4268
#train_storage
train_drowsy=np.array([])
train_alert=np.array([])
train_lowVigliant=np.array([])
#validation_storage-[41,42,43]
valid_drowsy=np.array([])
valid_alert=np.array([])
valid_lowVigliant=np.array([])
#test_storage-[44,45,46,47,48]
test_drowsy=np.array([])
test_alert=np.array([])
test_lowVigliant=np.array([])

#Alert
total_alert = len(alert)
train_limit_alert = int(0.7*total_alert) 
val_limit_alert = int((total_alert-train_limit_alert)*0.5)
val_limit_alert = train_limit_alert+val_limit_alert

print("total_alert",total_alert)
print("train_limit",train_limit_alert)
print("validation_limit",val_limit_alert)
#Copying alert images into training/alert
for i in range(train_limit_alert):
    source = reduce(os.path.join,[os.getcwd(),"output","alert"])
    destination = reduce(os.path.join,[os.getcwd(),"data","training","alert"])
    print(shutil.copy(os.path.join(source,alert[i]),destination+"/"))

for i in range(train_limit_alert,val_limit_alert):
    source = reduce(os.path.join,[os.getcwd(),"output","alert"])
    destination = reduce(os.path.join,[os.getcwd(),"data","validation","alert"])
    shutil.copy(os.path.join(source,alert[i]),destination+"/")

for i in range(val_limit_alert,total_alert):
    source = reduce(os.path.join,[os.getcwd(),"output","alert"])
    destination = reduce(os.path.join,[os.getcwd(),"data","testing","alert"])
    shutil.copy(os.path.join(source,alert[i]),destination+"/")

#Drowsy
total_drowsy = len(drowsy)
train_limit_drowsy = int(0.7*total_drowsy) 
val_limit_drowsy = int((total_drowsy-train_limit_drowsy)*0.5)
val_limit_drowsy = train_limit_drowsy+val_limit_drowsy

for i in range(train_limit_drowsy):
    source = reduce(os.path.join,[os.getcwd(),"output","drowsy"])
    destination = reduce(os.path.join,[os.getcwd(),"data","training","drowsy"])
    print(shutil.copy(os.path.join(source,drowsy[i]),destination+"/"))

for i in range(train_limit_drowsy,val_limit_drowsy):
    source = reduce(os.path.join,[os.getcwd(),"output","drowsy"])
    destination = reduce(os.path.join,[os.getcwd(),"data","validation","drowsy"])
    shutil.copy(os.path.join(source,drowsy[i]),destination+"/")

for i in range(val_limit_drowsy,total_drowsy):
    source = reduce(os.path.join,[os.getcwd(),"output","drowsy"])
    destination = reduce(os.path.join,[os.getcwd(),"data","testing","drowsy"])
    shutil.copy(os.path.join(source,drowsy[i]),destination+"/")

#Drowsy
total_low_vigilant = len(low_vigilant)
train_limit_low_vigilant = int(0.7*total_low_vigilant) 
val_limit_low_vigilant = int((total_low_vigilant-train_limit_low_vigilant)*0.5)
val_limit_low_vigilant = train_limit_low_vigilant+val_limit_low_vigilant

for i in range(train_limit_low_vigilant):
    source = reduce(os.path.join,[os.getcwd(),"output","low_vigilant"])
    destination = reduce(os.path.join,[os.getcwd(),"data","training","low_vigilant"])
    print(shutil.copy(os.path.join(source,low_vigilant[i]),destination+"/"))

for i in range(train_limit_low_vigilant,val_limit_low_vigilant):
    source = reduce(os.path.join,[os.getcwd(),"output","low_vigilant"])
    destination = reduce(os.path.join,[os.getcwd(),"data","validation","low_vigilant"])
    shutil.copy(os.path.join(source,low_vigilant[i]),destination+"/")

for i in range(val_limit_low_vigilant,total_low_vigilant):
    source = reduce(os.path.join,[os.getcwd(),"output","low_vigilant"])
    destination = reduce(os.path.join,[os.getcwd(),"data","testing","low_vigilant"])
    shutil.copy(os.path.join(source,low_vigilant[i]),destination+"/")











'''
total_train=len(train_drowsy)+len(train_alert)+len(train_lowVigliant)
total_test=len(test_drowsy)+len(test_alert)+len(test_lowVigliant)
total_valid=len(valid_drowsy)+len(valid_alert)+len(valid_lowVigliant)

print("test_drowsy:",len(test_drowsy)," test_alert:",len(test_alert)," test_lowVigilant:",len(test_lowVigliant)," test_total:",total_test)
print("valid_drowsy:",len(valid_drowsy)," valid_alert:",len(valid_alert)," valid_lowVigilant:",len(valid_lowVigliant)," valid_total:",total_valid)
print("train_drowsy:",len(train_drowsy)," train_alert:",len(train_alert)," train_lowVigilant:",len(train_lowVigliant)," train_total:",total_train)

#train
rnd.shuffle(train_drowsy)
rnd.shuffle(train_alert)
rnd.shuffle(train_lowVigliant)

#test
rnd.shuffle(test_drowsy)
rnd.shuffle(test_alert)
rnd.shuffle(test_lowVigliant)

#valid
rnd.shuffle(valid_drowsy)
rnd.shuffle(valid_alert)
rnd.shuffle(valid_lowVigliant)

train_valid_test=np.array([])
for i in range(total_train):
    train_valid_test=np.append(train_valid_test,'training')
for i in range(total_valid):
    train_valid_test=np.append(train_valid_test,'validation')
for i in range(total_test):
    train_valid_test=np.append(train_valid_test,'testing')

print("\n\nsize of dataframe:",len(train_valid_test))

train_frames=np.append(train_alert,train_drowsy)
train_frames=np.append(train_frames,train_lowVigliant)

test_frames=np.append(test_alert,test_drowsy)
test_frames=np.append(test_frames,test_lowVigliant)

valid_frames=np.append(valid_alert,valid_drowsy)
valid_frames=np.append(valid_frames,valid_lowVigliant)

frames=np.append(train_frames,valid_frames)
frames=np.append(frames,test_frames)

print("\n\ntest_frames:",len(test_frames))
print("valid_frames",len(valid_frames))
print("train_frames:",len(train_frames))
print("toatl_frames:",len(frames))

train_labels=np.append(['Alert']*len(train_alert),['Drowsy']*len(train_drowsy))
train_labels=np.append(train_labels,['Low_Vigilant']*len(train_lowVigliant))

test_labels=np.append(['Alert']*len(test_alert),['Drowsy']*len(test_drowsy))
test_labels=np.append(test_labels,['Low_Vigilant']*len(test_lowVigliant))

valid_labels=np.append(['Alert']*len(valid_alert),['Drowsy']*len(valid_drowsy))
valid_labels=np.append(valid_labels,['Low_Vigilant']*len(valid_lowVigliant))

labels=np.append(train_labels,valid_labels)
labels=np.append(labels,test_labels)

print("\n\ntest_labels:",len(test_labels))
print("valid_labels",len(valid_labels))
print("train_labels:",len(train_labels))
print("toatl_labels:",len(labels))

print("\n ----Dataframe-----")
ret=pd.DataFrame(train_valid_test.reshape(-1))
ret['frames']=frames.reshape(-1)
ret['labels']=labels.reshape(-1)

print(ret)

print("###############\n")
print("Creating 'data' directory (Will delete if already exists!)...")
dir = os.path.join(os.getcwd(),"data")
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(dir)
print("Created 'data' directory successfully!")

ret.to_csv(os.path.join(dir,'data_file.csv'),index=False,header=False)

print("\nSaved data_file.csv to data directory")
print("\n###############")

print("\nCreating 'training' 'validation'and 'testing' directory inside 'data' directory...\n")
traindir=os.path.join(dir,"training")
os.mkdir(traindir)
validdir=os.path.join(dir,"validation")
os.mkdir(validdir)
testdir=os.path.join(dir,"testing")
os.mkdir(testdir)
seqdir=os.path.join(dir,"sequences")
os.mkdir(seqdir)
seqtrain=os.path.join(dir,"sequences", "training")
os.mkdir(seqtrain)
seqvalidation=os.path.join(dir,"sequences", "validation")
os.mkdir(seqvalidation)
seqtest=os.path.join(dir,"sequences", "testing")
os.mkdir(seqtest)
print("Created directories successfully!")

print("\nCreating 'Alert', low_vigilant and 'Drowsy' directory inside 'training', validation and 'testing' directories...\n")
os.mkdir(os.path.join(traindir,"alert"))
os.mkdir(os.path.join(validdir,"alert"))
os.mkdir(os.path.join(testdir,"alert"))
os.mkdir(os.path.join(traindir,"drowsy"))
os.mkdir(os.path.join(validdir,"drowsy"))
os.mkdir(os.path.join(testdir,"drowsy"))
os.mkdir(os.path.join(traindir,"low_vigilant"))
os.mkdir(os.path.join(validdir,"low_vigilant"))
os.mkdir(os.path.join(testdir,"low_vigilant"))
print("Created directories successfully!")

import pandas as pd
import os
import shutil
from functools import reduce


df = pd.read_csv("./data/data_file.csv", header = None)

df_train = df[df.iloc[:,0] == 'training']
df_validation = df[df.iloc[:,0] == 'validation']
df_test = df[df.iloc[:,0] == 'testing']

for i in range(df.shape[0]):
    folder=df.iloc[i,0] #training,testing,validation
    frame_type=df.iloc[i,2].lower() #alert,drowsy,low_vigliant
    frame_name = df.iloc[i,1] #*.jpg
    source = reduce(os.path.join,[os.getcwd(),"output",frame_type,frame_name])
    #print("source = ",source)
    destination = reduce(os.path.join,[os.getcwd(),"data",folder,frame_type,frame_name])
    #print("destination = ",destination)
    shutil.copy(source,destination)


def move(df,type1):
    for i in range(df.shape[0])[:2]:
        # print("IN MOVE FOR")
        # cat = df.iloc[i,2].strip().split('_')[0]
        # print("cat = ",cat)
        # num = df.iloc[i,2].strip().split('_')[1][-1]
        # print("num = ",num)
        folderName = df.iloc[i,1].lower()
        #shutil.move(reduce(os.path.join,[os.getcwd(),"output",folderName]),reduce(os.path.join,[os.getcwd(),"data",type1,folderName]))
        source = reduce(os.path.join,[os.getcwd(),"output",folderName])
        print("source = ",source)
        destination = reduce(os.path.join,[os.getcwd(),"data",type1,folderName])
        print("destination = ",destination)
        print("df.iloc[i,2]) = ",df.iloc[i,2])
        #shutil.copy(os.path.join(source,df.iloc[i,2]),destination+"/")


move(df_train,"training")
move(df_validation, "validation")
move(df_test,"testing")
'''