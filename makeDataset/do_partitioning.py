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
from GrabCut import Grabcut

'''
def random_shuffle():
    dataset_name='recording1'
    
    data_num=10000
    data_num_train=8000
    data_num_test=2000
    
    train_save_filepath='/home/yong/ssd/dataset/depth_ir/train/'
    test_save_filepath='/home/yong/ssd/dataset/depth_ir/test/' 
    load_filepath='/home/yong/ssd/dataset/depth_ir/'+dataset_name+'/'  
    
    #random shuffling
    frames_shuffle=np.arange(data_num)
    np.random.shuffle(frames_shuffle)
    frame_idx_train=0
    frame_idx_test=0

    #--dataset of uvr
    datasetloader_uvr=datasetloader_UVR(load_filepath,"..",0,0)
    
    #--start
    progressbar=trange(data_num,leave=True)  
    for i in progressbar:
    #for i in range(data_num):
        frame=frames_shuffle[i]
        
        depth=datasetloader_uvr.loadDepthImage(frame)
        ir=datasetloader_uvr.loadIrImage(frame)
        
        
        #save
        if i<data_num_train:    
            cv2.imwrite(train_save_filepath+'depth%d.png'%frame_idx_train,depth)
            cv2.imwrite(train_save_filepath+'ir%d.png'%frame_idx_train,ir)
            frame_idx_train+=1
        else:
            cv2.imwrite(test_save_filepath+'depth%d.png'%frame_idx_test,depth)
            cv2.imwrite(test_save_filepath+'ir%d.png'%frame_idx_test,ir)
            frame_idx_test+=1
'''

    
if __name__=="__main__":
    #--user setting
    dataset_name='recording6' 
    
    #--common setting
    '''
    ordering:  train/ train_blur/ test/ test_blur
    '''
    #(1-1) training dataset (first good result, but should be deleted. We trained dataset with marker in fault. )
    '''
    bag={}
    bag['recording1']={}
    bag['recording1']['num']=[5000,0,0,0]
    bag['recording1']['start']=[0,0,0,0]
    bag['recording1']['data_num']=5000

    bag['recording2']={}
    bag['recording2']['num']=[7000,0,0,0]
    bag['recording2']['start']=[5000,0,0,0]
    bag['recording2']['data_num']=7000
    
    bag['recording4']={}
    bag['recording4']['num']=[2000,0,0,0]
    bag['recording4']['start']=[12000,0,0,0]
    bag['recording4']['data_num']=2000
    
    bag['recording5']={}
    bag['recording5']['num']=[5000,0,1000,1000]
    bag['recording5']['start']=[14000,0,0,0]
    bag['recording5']['data_num']=7000
    
    bag['recording6']={}
    bag['recording6']['num']=[10000,0,0,0]
    bag['recording6']['start']=[19000,0,1000,1000]
    bag['recording6']['data_num']=10000
    
    bag['recording7']={}
    bag['recording7']['num']=[0,1000,0,0]
    bag['recording7']['start']=[29000,0,1000,1000]
    bag['recording7']['data_num']=1000
    
    bag['recording8']={}
    bag['recording8']['num']=[5000,0,0,0]
    bag['recording8']['start']=[29000,1000,1000,1000]
    bag['recording8']['data_num']=5000
    '''
    
    #(1-2) training dataset (without motion blur)
    bag={}
    bag['recording1']={}
    bag['recording1']['num']=[5000,0,0,0]
    bag['recording1']['start']=[0,0,0,0]
    bag['recording1']['data_num']=5000
    
    bag['recording2']={}
    bag['recording2']['num']=[10000,0,0,0]
    bag['recording2']['start']=[5000,0,0,0]
    bag['recording2']['data_num']=10000
    
    
    bag['recording3']={}
    bag['recording3']['num']=[5000,0,0,0]
    bag['recording3']['start']=[15000,0,0,0]
    bag['recording3']['data_num']=5000    
    
    bag['recording4']={}
    bag['recording4']['num']=[10000,0,0,0]
    bag['recording4']['start']=[20000,0,0,0]
    bag['recording4']['data_num']=10000    
      
    bag['recording5']={}
    bag['recording5']['num']=[5000,0,0,0]
    bag['recording5']['start']=[30000,0,0,0]
    bag['recording5']['data_num']=5000    
    
    #(1-3) training dataset (With motion blur)
    bag['recording6']={}
    bag['recording6']['num']=[0,5000,0,0]
    bag['recording6']['start']=[35000,0,0,0]
    bag['recording6']['data_num']=5000    
    
    

    #(2) ablative dataset
    '''
    bag={}
    bag['recording9']={}
    bag['recording9']['num']=[0,0,1000,1000]
    bag['recording9']['start']=[0,0,0,0]
    bag['recording9']['data_num']=2000
    '''
    
    #(3) SOTA dataset (new challenging dataset)
    
    

    #--common setting
    data_num=bag[dataset_name]['data_num']
    
    data_num_train_noblur=bag[dataset_name]['num'][0]
    data_num_train_blur=np.sum(bag[dataset_name]['num'][0:2])
    data_num_test_noblur=np.sum(bag[dataset_name]['num'][0:3])
    data_num_test_blur=np.sum(bag[dataset_name]['num'][0:4])
    
    load_filepath='/home/yong/ssd/dataset/depth_ir/'+dataset_name+'/'  
    
    train_noblur_save_filepath='/home/yong/ssd/dataset/depth_ir/train/'
    train_blur_save_filepath='/home/yong/ssd/dataset/depth_ir/train_blur/'
    test_noblur_save_filepath='/home/yong/ssd/dataset/depth_ir/test/' 
    test_blur_save_filepath='/home/yong/ssd/dataset/depth_ir/test_blur/' 
    
    #random shuffling
    frames=np.arange(data_num)
    frame_idx_train_noblur=bag[dataset_name]['start'][0]
    frame_idx_train_blur=bag[dataset_name]['start'][1]
    frame_idx_test_noblur=bag[dataset_name]['start'][2]
    frame_idx_test_blur=bag[dataset_name]['start'][3]
    #--dataset of uvr
    #datasetloader_uvr=datasetloader_UVR(load_filepath,"..",0,0)
    
    
    
    #--start
    progressbar=trange(data_num,leave=True)  
    for i in progressbar:
    #for i in range(data_num):
        frame=frames[i]
        
        #depth=datasetloader_uvr.loadDepthImage(frame)
        #ir=datasetloader_uvr.loadIrImage(frame)
        depth=cv2.imread(load_filepath+'depth%d.png'%frame,2)
        ir=cv2.imread(load_filepath+'ir%d.png'%frame,2)
        
        
        #save
        if i<data_num_train_noblur:    
            cv2.imwrite(train_noblur_save_filepath+'depth%d.png'%frame_idx_train_noblur,depth)
            cv2.imwrite(train_noblur_save_filepath+'ir%d.png'%frame_idx_train_noblur,ir)
            frame_idx_train_noblur+=1
        elif i>=data_num_train_noblur and i<data_num_train_blur:
            cv2.imwrite(train_blur_save_filepath+'depth%d.png'%frame_idx_train_blur,depth)
            cv2.imwrite(train_blur_save_filepath+'ir%d.png'%frame_idx_train_blur,ir)
            frame_idx_train_blur+=1
        elif i>=data_num_train_blur and i<data_num_test_noblur:
            cv2.imwrite(test_noblur_save_filepath+'depth%d.png'%frame_idx_test_noblur,depth)
            cv2.imwrite(test_noblur_save_filepath+'ir%d.png'%frame_idx_test_noblur,ir)
            frame_idx_test_noblur+=1
        else:
            cv2.imwrite(test_blur_save_filepath+'depth%d.png'%frame_idx_test_blur,depth)
            cv2.imwrite(test_blur_save_filepath+'ir%d.png'%frame_idx_test_blur,ir)
            frame_idx_test_blur+=1
            
    
    
        


















