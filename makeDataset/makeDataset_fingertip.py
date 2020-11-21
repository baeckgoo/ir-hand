import cv2
import numpy as np
import csv
import sys
import time
import pickle
#from pyquaternion import Quaternion

import random
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange


from sklearn.decomposition import PCA
from scipy import stats, ndimage
#import pandas as pd

from tqdm import tqdm
from tqdm import trange

import torch
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
from datasetLoader_uvr import Utils 
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from datasetloader.datasetLoader_uvr import DatasetLoader as datasetloader_UVR

def detectMarker(irimg):
    irimg2=np.copy(irimg)
    irimg_norm=cv2.normalize(irimg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    irimg3c=np.uint8(cv2.cvtColor(irimg_norm, cv2.COLOR_GRAY2BGR))
        
        
    mask=irimg2<100 #use original ir
    irimg2[mask]=0
    irimg_norm = cv2.normalize(irimg2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    irimg_norm= np.uint8(irimg_norm)
    cv2.imshow("segmented_orig",irimg_norm)
            
    #--2D detection/labeling fingertips based on contour
    kernel=np.ones((3,3),np.uint8)
    irimg_norm=cv2.dilate(irimg_norm,kernel,iterations=2)
    ret, irimg_norm_seg = cv2.threshold(irimg_norm, 100, 255, cv2.THRESH_BINARY)
            
    contours, hierarchy = cv2.findContours(irimg_norm_seg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(irimg3c,contours,-1,(200,0,0),2)
    cv2.putText(irimg3c,'%d'%len(contours),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
            
    cnt2d=[]
    contour_size=[]
    for i,c in enumerate(contours):            
        M=cv2.moments(c)
        cnt2d.append(np.asarray([[M["m10"] / M["m00"]],[M["m01"] / M["m00"]]]))
        contour_size.append(len(c))
        
    idx=np.argmax(contour_size)
        
    #select a fingertip
    M=cv2.moments(contours[idx])
    x=int(M["m10"] / M["m00"])
    y=int(M["m01"] / M["m00"])
    
    cv2.circle(irimg3c,(x,y),5,(0,0,255),-1)
    cv2.imshow('circle',irimg3c)

    return [x,y]

if __name__=="__main__":
    ##--user setting
    dataset_name='v2'  #'test' , 'test_blur', 'v2'

    
    
    ##--common setting
    #datanum
    if dataset_name=='test' or dataset_name=='test_blur':
        data_num=1000
    elif dataset_name=='v2':
        data_num=8000
        
    #path
    if dataset_name=='test' or dataset_name=='test_blur':
        load_filepath_img='/home/yong/ssd/dataset/depth_ir/'+dataset_name+'/'
        save_filepath_marker='/home/yong/ssd/dataset/depth_ir/fingertip2d_gt_'+dataset_name+'.txt'
    elif dataset_name=='v2':
        load_filepath_img='/home/yong/ssd/dataset/blurHand20/'+dataset_name+'/images/'
        save_filepath_marker='/home/yong/ssd/dataset/blurHand20/'+dataset_name+'/fingertip2d_gt_'+dataset_name+'.txt'
           
    print('loaded image file path: ',load_filepath_img)
    print('save file path: ',save_filepath_marker)
    
    #enable
    marker_detection_enable=True
    save_enable=True
    
    ##--common setting
    d_minimum=0
    d_maximum=500
    
    camera_info={}
    camera_info['fx']=475.065948
    camera_info['fy']=475.065857
    camera_info['cx']=315.944855
    camera_info['cy']=245.287079
    camera_info['camerawidth']=640
    camera_info['cameraheight']=480
        
    camera_info['cube']=np.asarray([250,250,250])
    camera_info['trainImageSize']=128
    trainImageSize=camera_info['trainImageSize']
    cube=camera_info['cube']
    
    utils=Utils(camera_info)
    imgs_train=np.zeros((trainImageSize,2*trainImageSize),'float32')
    
    #--dataset of uvr
    datasetloader_uvr=datasetloader_UVR('..',data_num,camera_info,'test','generator')
    
    
    #--start
    ground2d={}
    for frame in range(data_num):
        depth=cv2.imread(load_filepath_img+'depth%d.png'%frame,2)
        ir=cv2.imread(load_filepath_img+'ir%d.png'%frame,2)

        depth_seg=depth.copy()
        depth_seg[depth_seg>d_maximum]=0
        
        # preprocess (depth)
        depth_train,depth_crop,com,window=datasetloader_uvr.preprocess_depth(depth_seg,trainImageSize,cube)
            
        # preprocess (ir)
        ir_train=datasetloader_uvr.preprocess_ir(ir,window)  
          
        # detect marker position (fingertip)
        ground2d[frame]=detectMarker(ir)
        cv2.waitKey(1)   
                   
    #save marker position (fingertip)
    if save_enable==True:            
        f = open(save_filepath_marker, 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        for fr in range(data_num):
            wr.writerow(ground2d[fr])
        f.close()   
    
            
 
    
    
 
    
    
        
    
    
    



















