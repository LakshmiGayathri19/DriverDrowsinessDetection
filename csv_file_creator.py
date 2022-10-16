import os
import pandas as pd
import numpy as np
import random as rnd
import shutil
import sys
from glob import glob

base_directory="output/"
f_drowsy=np.array([])
f_alert=np.array([])
f_lowVigliant=np.array([])

for root, dirs, files in os.walk(os.path.join(os.getcwd(),base_directory,"drowsy")):
    for file in files:
        print(file)
        if not file.endswith('.txt'):
            f_drowsy = np.append(f_drowsy,file)

for root, dirs, files in os.walk(os.path.join(os.getcwd(),base_directory,"alert")):
    for file in files:
        if not file.endswith('.txt'):
            f_alert = np.append(f_alert,file)
    
    
for root, dirs, files in os.walk(os.path.join(os.getcwd(),base_directory,"low_vigilant")):
    for file in files:
        if not file.endswith('.txt'):
            f_lowVigliant = np.append(f_lowVigliant,file)

print("drowsy len = ", len(f_drowsy))
print("alert len = ", len(f_alert))
print("vigilent len = ", len(f_lowVigliant))
rnd.shuffle(f_drowsy)
rnd.shuffle(f_alert)
rnd.shuffle(f_lowVigliant)

total=len(f_drowsy)+len(f_alert)+len(f_lowVigliant)
print("total = ",total)
train_test_split=float(0.7)
train_test=np.array([])
for i in range(int(train_test_split*total)):
    train_test=np.append(train_test,'training')

validation_split = (total-int(train_test_split*total))
for i in range(int(validation_split*0.5)):
    train_test=np.append(train_test,'validation')

for i in range(int(validation_split*0.5)):
    train_test=np.append(train_test, 'testing')

names1=np.append(f_drowsy[:int(train_test_split*len(f_drowsy))],f_alert[:int(train_test_split*len(f_alert))])
names1=np.append(names1,f_lowVigliant[:int(train_test_split*len(f_lowVigliant))])
print("names1 len ",len(names1))

val_upper_limit_drowsy = int(train_test_split*len(f_drowsy)) + int(len(f_drowsy)-int(train_test_split*len(f_drowsy))*0.5)
val_upper_limit_alert = int(train_test_split*len(f_alert)) + int(len(f_alert)-int(train_test_split*len(f_alert))*0.5)
val_upper_limit_lowVigliant = int(train_test_split*len(f_lowVigliant)) + int(len(f_lowVigliant)-int(train_test_split*len(f_lowVigliant))*0.5)

names2=np.append(f_drowsy[int(train_test_split*len(f_drowsy)):val_upper_limit_drowsy],f_alert[int(train_test_split*len(f_alert)):val_upper_limit_alert])
names2=np.append(names2,f_lowVigliant[int(train_test_split*len(f_lowVigliant)):val_upper_limit_lowVigliant])
print("names1 len ",len(names2))

names3=np.append(f_drowsy[val_upper_limit_drowsy:],f_alert[val_upper_limit_alert:])
names3=np.append(names3,f_lowVigliant[val_upper_limit_lowVigliant:])
print("names1 len ",len(names3))

names=np.append(names1,names2)
names=np.append(names,names3)

types1=np.append(['Drowsy']*int(train_test_split*len(f_drowsy)),['Alert']*int(train_test_split*len(f_alert)))
types1=np.append(types1,['Low_Vigilant']*int(train_test_split*len(f_lowVigliant)))

types2=np.append(['Drowsy']*(val_upper_limit_drowsy-int(train_test_split*len(f_drowsy))),['Alert']*(val_upper_limit_alert-int(train_test_split*len(f_alert))))
types2=np.append(types2,['Low_Vigilant']*(val_upper_limit_lowVigliant-int(train_test_split*len(f_lowVigliant))))

types3=np.append(['Drowsy']*(len(f_drowsy)-val_upper_limit_drowsy),['Alert']*(len(f_alert)-val_upper_limit_alert))
types3=np.append(types3,['Low_Vigilant']*(len(f_lowVigliant)-val_upper_limit_lowVigliant))
types=np.append(types1,types2)
types=np.append(types,types3)

print("length of train_test = ",len(train_test))
print("length of types = ", len(types))
print("length of names = ",len(names))
ret = pd.DataFrame(train_test.reshape(-1))
ret['type']= types.reshape(-1)
ret['names']=names.reshape(-1)

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

print("\nCreating 'training' and 'testing' directory inside 'data' directory...\n")
traindir=os.path.join(dir,"training")
os.mkdir(traindir)
validation_dir=os.path.join(dir,"validation_split")
os.mkdir(validation_dir)
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
os.mkdir(os.path.join(testdir,"alert"))
os.mkdir(os.path.join(traindir,"drowsy"))
os.mkdir(os.path.join(testdir,"drowsy"))
os.mkdir(os.path.join(traindir,"low_vigilant"))
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

def move(df,type1):
    for i in range(df.shape[0]):
        # print("IN MOVE FOR")
        # cat = df.iloc[i,2].strip().split('_')[0]
        # print("cat = ",cat)
        # num = df.iloc[i,2].strip().split('_')[1][-1]
        # print("num = ",num)
        folderName = df.iloc[i,1].lower()
        #shutil.move(reduce(os.path.join,[os.getcwd(),"output",folderName]),reduce(os.path.join,[os.getcwd(),"data",type1,folderName]))
        source = reduce(os.path.join,[os.getcwd(),"output",folderName])
        #print("source = ",source)
        destination = reduce(os.path.join,[os.getcwd(),"data",type1,folderName])
        #print("destination = ",destination)
        shutil.copy(os.path.join(source,df.iloc[i,2]),destination+"/")


move(df_train,"training")
move(df_validation, "validation")
move(df_test,"testing")

#def move_temp(df,type1):
#    for i in range(df.shape[0]):
#s        cat = df.iloc[i,2].strip().split('_')[0]
#        num = df.iloc[i,2].strip().split('_')[1][-1]
#        strng = df.iloc[i,2]
#        shutil.copy(reduce(os.path.join,[os.getcwd(),"data",type1,strng]),reduce(os.path.join,[os.getcwd(),"data",type1]))
