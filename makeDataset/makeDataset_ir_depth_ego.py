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

from multiprocessing import Process, Manager

from Segmentation.Segmentation import Segmentation

def fct_grabcut(Mobj,idx,ir,depth,com,window,d_max):
    ir_seg=Grabcut.segment_grabcut(ir,depth,com,window,d_max)#0.6
            
    Mobj[idx]=ir_seg
 
class Augmentation():
    def __init__(self,fx,fy,cx,cy,cube):
        self.fx=fx
        self.fy=fy
        self.cx=cx
        self.cy=cy
        self.calibMat=np.asarray([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]]) 
        self.cube=cube
        self.rng= np.random.RandomState(23455)        
        
    def unproject2Dto3D(self,p):
        '''[pixel,pixel,mm]->[mm,mm,mm]'''
        x=(p[2]*p[0]-p[2]*self.cx)/self.fx
        y=(p[2]*p[1]-p[2]*self.cy)/self.fy
        z=p[2]
        return np.asarray([x,y,z],'float')
    
    def project3D2imagePlane(self,pos):
        calibMat=self.calibMat
        p2d=np.matmul(calibMat,np.reshape(pos,3,1))
        p2d[0:2]=p2d[0:2]/p2d[2]
        return np.asarray(p2d,'int')#.astype(np.int32)
        
    def comToBounds(self,com,size):
        '''
        com: [pixel,pixel,mm] 
        '''
        fx=self.fx
        fy=self.fy
        
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2]*fx))
        xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2]*fx))
        ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2]*fy))
        yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2]*fy))
        
        return xstart, xend, ystart, yend, zstart, zend
    
    def crop(self,dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z=True):
        cropped = dpt[ystart:yend, xstart:xend].copy()
        msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = np.bitwise_and(cropped > zend, cropped != 0)
        cropped[msk1] = zstart
        cropped[msk2] = 0.
        return cropped
    
    def augmentRotation(self,img_seg,com,rot):
        M=cv2.getRotationMatrix2D((com[0],com[1]),-rot,1)
        img_seg_rot = cv2.warpAffine(img_seg, M, (img_seg.shape[1], img_seg.shape[0]), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img_seg_rot

    def augmentTranslation(self,img_seg,com,off):
        com3d=self.unproject2Dto3D(com) 
        com3d_new=com3d+off
            
        com2d_new=self.project3D2imagePlane(com3d_new)
        xstart, xend, ystart, yend, zstart, zend=self.comToBounds(com2d_new,self.cube)
        xstart=max(xstart,0)
        ystart=max(ystart,0)
        xend=min(xend,img_seg.shape[1])
        yend=min(yend,img_seg.shape[0])
            
        return com2d_new,com3d_new,[xstart,xend,ystart,yend,zstart,zend]
   
    def augmentScale(self,img_seg,cube,com2d,sc):
        new_cube = [s*sc for s in cube]
              
        xstart, xend, ystart, yend, zstart, zend=self.comToBounds(com2d,new_cube)
        xstart=max(xstart,0)
        ystart=max(ystart,0)
        xend=min(xend,img_seg.shape[1])
        yend=min(yend,img_seg.shape[0])
            
        return new_cube,[xstart,xend,ystart,yend,zstart,zend]
    
    def set_augmentation_parameters(self,sigma_com,rot_range,sigma_sc):
        #sigma_com = 5.
        #rot_range=180.
        #sigma_sc = 0.02
    
        self.off=self.rng.randn(3)*sigma_com # mean:0, var:sigma_com (normal distrib.)
    
        rot=self.rng.uniform(-rot_range,rot_range)
        self.rot=np.mod(rot,360)
               
        self.sc = np.fabs(self.rng.randn(1) * sigma_sc + 1.)
        
    def augment(self,img_seg,com):
        img_seg_2=self.augmentRotation(img_seg,com,self.rot)
        com2d,com3d,_=self.augmentTranslation(img_seg_2,com,self.off)
        cube,window=self.augmentScale(img_seg_2,self.cube,com2d,self.sc)
        
        #--apply augmentation to image 
        xstart,xend,ystart,yend,zstart,zend=window[0],window[1],window[2],window[3],window[4],window[5]
        img_crop=self.crop(img_seg_2,xstart,xend,ystart,yend,zstart,zend)
        #img_train=self.makeLearningImage(img_crop,com)
        
        return img_seg_2,img_crop,cube,window
    
    
    

if __name__=="__main__":
    ##--user setting
    
    dataset_name='train_ego'
    frame_start=0
    frame_end=5000
    save_enable=True
    
    '''    
    dataset_name='train_ego_blur'
    frame_start=0
    frame_end=5000
    save_enable=True
    '''
         
    ##-- thread num / augmentation num
    augment_enable=True
    augmentation_number=10   
    
    ##--common setting
    sigma_com = 5.
    rot_range=90.
    sigma_sc = 0.02


    load_filepath='/home/yong/ssd/dataset/depth_ir/'+dataset_name+'/'
    save_filepath_img='/home/yong/ssd/dataset/preprocessed_HIG/'+dataset_name+'/'
    
    save_filepath_img_aug='/home/yong/hdd/dataset_debug/segmented_HIG/'+dataset_name+'/'
    
    print('save file path(test): ',save_filepath_img)    
    print('loaded image file path: ',load_filepath)  
    
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
    datasetloader_uvr=datasetloader_UVR("..",0,camera_info,'..','..')
    
    #--saved dataset
    savepath_dict="/home/yong/ssd/dataset/preprocessed_HIG/dataset_dict_"+dataset_name+".pickle"
    dataset_dict={}
    dataset_dict['com']=[]
    dataset_dict['window']=[]
           
    
    #thread
    
    
    #augmentation
    fx,fy,cx,cy=camera_info['fx'],camera_info['fy'],camera_info['cx'],camera_info['cy']
    augmentation=Augmentation(fx,fy,cx,cy,cube)
    
    #segmentation
    lower_hsv=np.array([94,111,37])
    upper_hsv=np.array([120,255,255])
    ir_range=[40,120]
    segmentation=Segmentation(d_maximum,lower_hsv,upper_hsv,ir_range)
    
    #--start
    save_frame=0 #0
    
    progressbar=trange(frame_start,frame_end,leave=True)  
    for frame in progressbar:                 

        depth=cv2.imread(load_filepath+'depth%d.png'%frame,2)
        ir=cv2.imread(load_filepath+'ir%d.png'%frame,2)
        rgb=cv2.imread(load_filepath+'color%d.png'%frame)
                
                
        #(1-a) segment (depth) -without blueband
        '''
        depth_seg=depth.copy()
        depth_seg[depth_seg>d_maximum]=0
        '''
        #(1-b) segment (depth) -with blueband
        segmentation.setImages(rgb,depth,ir)
        segmentation.makeMask_band()
        depth_seg=segmentation.segmentDepth()
           
        #plt.imshow(depth_seg)
        
        #(1) preprocess (depth)
        depth_train,depth_crop,com,window=datasetloader_uvr.preprocess_depth(depth_seg,trainImageSize,cube)
                
        
        #(2-a) segment (ir) -with blueband
        ir_seg=segmentation.segmentIR(window)
        
        #plt.imshow(ir_seg)
        #jio
                
        #(2-b) segment (ir) -without blueband (grabcut)
        '''
        ir_seg=Grabcut.segment_grabcut(ir,depth,com,window,d_maximum)#0.6
        #h_mask=segmentation.h_mask
        #ir_seg,mask=Grabcut.segment_grabcut_blueband(ir,depth,com,window,d_maximum,h_mask)#0.6
        '''
                    
        # (2) preprocess (ir)
        ir_train=datasetloader_uvr.preprocess_ir(ir_seg,window)#0.0002

        #-save
        imgs_train[:,0:trainImageSize]=depth_train
        imgs_train[:,trainImageSize:2*trainImageSize]=ir_train
        cv2.imshow('train',imgs_train)
        cv2.waitKey(1)
                
        if save_enable==True:
            np.save(save_filepath_img+'%d.npy'%save_frame,imgs_train)
        save_frame+=1
                
        #augment
        if augment_enable==True:
            for j in range(augmentation_number):
                augmentation.set_augmentation_parameters(sigma_com,rot_range,sigma_sc)
                depth_seg_aug,depth_crop_aug,cube_aug,window_aug=augmentation.augment(depth_seg,com)
                depth_train_aug=datasetloader_uvr.utils.makeLearningImage(depth_crop_aug,com)
                        
                        
                ir_seg_aug=augmentation.augmentRotation(ir_seg,com,augmentation.rot)
                ir_train_aug=datasetloader_uvr.preprocess_ir(ir_seg_aug,window_aug)#0.0002
                        
                imgs_train[:,0:trainImageSize]=depth_train_aug
                imgs_train[:,trainImageSize:2*trainImageSize]=ir_train_aug
                cv2.imshow('train',imgs_train)
                cv2.waitKey(1)
                        
                if save_enable==True:
                    np.save(save_filepath_img+'%d.npy'%save_frame,imgs_train)                          
                save_frame+=1
                
        
        cv2.waitKey(1)
        #jio
        
        

    '''             
    if save_enable==True:
        with open(savepath_dict,'wb') as f:
            pickle.dump(dataset_dict,f,pickle.HIGHEST_PROTOCOL)
    '''
    
    
    
        
    
    
    



















