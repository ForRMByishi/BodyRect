import numpy as np
import cv2
import imutils
from BodyRect import BodyRect
import os
from toolbox import *
from matplotlib import pyplot as plt
import pickle

backShape = {'feature_names':['ID', 'Quality','Q1','Q1-2','Q1-3','Q1-4'],\
             'data':[],\
             'DESCR':'Back Images Diagnostic Database',\
             'target':[],\
             'target_names': ['Nonscoliotic', 'Scoliotic', 'Postoperative']}

picNames = list(filter(lambda x: x.lower().endswith(('.png', '.jpg', '.jpeg')), os.listdir('pic-origs')))
i=0



for p in picNames:
    print(p) 
    dataList=[]
    dataFinish=False
    
    image= readImgResize('pic-origs/'+p)
    i2 = BodyRect(image)
    

   
    if i2.quality > 0.5:
        dataList.append(p)
        dataList.append(i2.quality)
        print('Rotation, Shift='+str(i2.optimizeRotShift()))
        
        im2=i2.retrieveSubR()
    
        cv2.imshow('Image',im2)
        
        q=i2.retrieveSubQuarts()
        
        for i in range(0,len(q)):
            x,y=i2.returnNormalizedXY(q[i])
            if len(x)==0:
                print('Image discarded')
                dataFinish=False
                break
            dataList.append((x,y))
            dataFinish=True
            print(np.mean(y))
        if dataFinish==False:
            continue
        t=input('0=Nonscoliotic, 1=scoliotic, 2=postoperative: ')
        if int(t)<3:
            backShape['data'].append(dataList)
            backShape['target'].append(t)
        else:
            print('Discarded')
        
        cv2.destroyAllWindows()
print(i)
pickle.dump( backShape, open( "backdata.p", "wb" ) )
