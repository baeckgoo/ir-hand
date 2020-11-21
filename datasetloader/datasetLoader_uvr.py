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

from multiprocessing import Process, Manager

def nothing(x):
    pass

def fct_load(Mobj,ttt,frame,data_path):
    while(1):
        Mobj[ttt]=np.load(data_path+'/%d.npy'%frame)
        
    '''
    Mobj[ttt]=np.load(data_path+'/%d.npy'%frame)
    '''
        
    
        
class Utils:
    def __init__(self,camera_info):

        self.fx=camera_info['fx']
        self.fy=camera_info['fy']
        self.cx=camera_info['cx']
        self.cy=camera_info['cy']
        self.calibMat=np.asarray([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]]) 
        self.cube=camera_info['cube']
        
    def project3D2imagePlane(self,pos):
        calibMat=self.calibMat
        p2d=np.matmul(calibMat,np.reshape(pos,3,1))
        p2d[0:2]=p2d[0:2]/p2d[2]
        return np.asarray(p2d,'int')#.astype(np.int32)

    def projectAlljoints2imagePlane(self,ground_3dpos):
        '''
        input:  [x,y,d]  unit: [mm,mm,mm]
        output: [u,v,d]  unit: [pixel,pixel,mm]
        '''
        calibMat=self.calibMat
        #pos2d=np.zeros((jointNum,2))
        pos2d=np.copy(ground_3dpos)#np.zeros_like(ground_3dpos)
        
        p2d=np.matmul(calibMat,np.transpose(ground_3dpos)) #(3*3)*(N*3)^T
        pos2d[:,0:2]=np.transpose(p2d[0:2,:]/p2d[2,:])
        
        
        #pos2d=np.transpose(pos2d[0:2,:])
        #return pos2d.astype(np.int32)
        return pos2d
    
    def circle3DJoints2Image(self,img,pos3d):
        img_=np.copy(img)
        jointnum=len(pos3d)//3
        
        if len(pos3d.shape)==1:
            ground_3dpos_=np.reshape(pos3d,(jointnum,3))
        
        ground_2dpos=self.projectAlljoints2imagePlane(ground_3dpos_)
        
        color=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255]]
        for j in range(jointnum):
            x,y=int(ground_2dpos[j,0]),int(ground_2dpos[j,1])

            #cid=j//4
            #cv2.circle(img_,(x,y),5,color[cid],-1)
            
            if j<6:
                cv2.circle(img_,(x,y),5,color[0],-1)
            else:
                cid=(j-6)//3
                cv2.circle(img_,(x,y),5,color[1+cid],-1)
            
        
        return img_
    
    def unproject2Dto3D(self,p):
        '''[pixel,pixel,mm]->[mm,mm,mm]'''
        x=(p[2]*p[0]-p[2]*self.cx)/self.fx
        y=(p[2]*p[1]-p[2]*self.cy)/self.fy
        z=p[2]
        return np.asarray([x,y,z],'float')
      
    def getRefinedCOM(self):
        return self.com_refined.astype(np.float)
    
    def refineCOMIterative(self,dimg,com,num_iter):
        dpt=dimg.copy()
        for k in range(num_iter):
            #size=np.asarray(size)*(1-0.1*k)
            #print(size)
            xstart, xend, ystart, yend, zstart, zend=self.comToBounds(com,self.cube)
            
            xstart=max(xstart,0)
            ystart=max(ystart,0)
            xend=min(xend,dpt.shape[1])
            yend=min(yend,dpt.shape[0])

            cropped=self.crop(dpt,xstart,xend,ystart,yend,zstart,zend)
            
            
            com=self.calculateCOM(cropped)
            
            if np.allclose(com,0.):
                com[2]=cropped[cropped.shape[0]//2,cropped.shape[1]//2]
            com[0]+=max(xstart,0)
            com[1]+=max(ystart,0)
    
        return com,cropped,[xstart,xend,ystart,yend,zstart,zend]

    def crop(self,dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z=True):
        cropped = dpt[ystart:yend, xstart:xend].copy()
        msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = np.bitwise_and(cropped > zend, cropped != 0)
        cropped[msk1] = zstart
        cropped[msk2] = 0.
        return cropped

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

    def calculateCOM(self,dimg,minDepth=10,maxDepth=500):
        
        dc=dimg.copy()
        
        dc[dc<minDepth]=0
        dc[dc>maxDepth]=0
        
        cc= ndimage.measurements.center_of_mass(dc>0) #0.001
        
        num=np.count_nonzero(dc) #0.0005
        
        com=np.array((cc[1]*num,cc[0]*num,dc.sum()),np.float) #0.0002
        
        if num==0:
            print('com can not be calculated (calculateCOM)')
            return np.zeros(3)
        else:
            return com/num
    
    def makeLearningImage(self,img_crop,com):
        s=128
        cnnimg=img_crop.copy()
        cnnimg[cnnimg==0]=com[2]+self.cube[2]/2.
        cnnimg=cnnimg-com[2]
        cnnimg=cnnimg/(self.cube[2]/2.)
        
        cnnimg=cv2.resize(cnnimg,(s,s))
        return np.copy(cnnimg)
    
    def makeLearningImage_pre(self,img_crop,trainImageSize):
        s=trainImageSize
        cnnimg=cv2.resize(img_crop,(s,s))
        cnnimg[cnnimg==0]=np.max(cnnimg)
        d_max=np.max(cnnimg)
        d_min=np.min(cnnimg)
        if d_max-d_min<1:
            print('dmax-dmin<1 min%f max:%f'%(d_min,d_max))
        cnnimg=-1+2*(cnnimg-d_min)/(d_max-d_min)        
        #return img_seg,cv2.resize(img_crop,(s,s))
            
        return np.copy(cnnimg)
        
class DatasetLoader:
    def __init__(self,dataset_path_preprocess,dataNum,camera_info,datasetName,loadOpt):
        #self.dataset_path=dataset_path
        self.dataset_path_preprocess=dataset_path_preprocess
        
        fx=camera_info['fx']#475.065948
        fy=camera_info['fy']#475.065857
        cx=camera_info['cx']#315.944855
        cy=camera_info['cy']#245.287079
        self.calibMat=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])   

        self.camerawidth=camera_info['camerawidth']#640
        self.cameraheight=camera_info['cameraheight']#480
        self.trainImageSize=camera_info['trainImageSize']#128
        self.cube=camera_info['cube']#np.asarray([250,250,250])
        
        self.dataNum=dataNum
        self.idx_data=np.arange(dataNum)
        self.datasetName=datasetName #'train' 'validate' 'train_blur' 'validate_blur
        
        self.utils=Utils(camera_info)
        
        #to get offline trainingset.end      
        self.rng= np.random.RandomState(23455)
        
        self.loadOpt=loadOpt
              
    def set_datasetLoader(self,batch_size):
        #batch data
        self.batch_size=batch_size
        #np.random.shuffle(self.idx_data)
        self.cnn_img_depth = np.zeros((batch_size,1,self.trainImageSize,self.trainImageSize), dtype = np.float32)
        self.cnn_img_ir = np.zeros((batch_size,1,self.trainImageSize,self.trainImageSize), dtype = np.float32)
        
        #set
        if self.loadOpt=='generator':
            self.getDataFromGenerator=self.generator_learningData()
        elif self.loadOpt=='ram':
            self.cnn_imgs=np.zeros((self.dataNum,self.trainImageSize,self.trainImageSize*2))
            for frame in range(self.dataNum):
                self.cnn_imgs[frame]=np.load(self.dataset_path_preprocess+self.datasetName+'/%d.npy'%frame)
        elif self.loadOpt=='thread':
            print('not implemented..')
            thread_num=16
            self.manager=Manager()
            self.Mobj=self.manager.list([0 for x in range(thread_num)])
            self.MpList=[]
            
            for ttt in range(thread_num):              
                self.MpList.append(Process(target=fct_load,args=(self.Mobj,ttt)))
    
        
        self.global_frame=0
        
    def shuffle(self):
        np.random.shuffle(self.idx_data)
        
        
    def load_data(self):
        if self.loadOpt=='generator':
            return next(self.getDataFromGenerator)
        elif self.loadOpt=='ram':
            return self.getDataFromRam()
        elif self.loadOpt=='thread':
            return self.getDataFromThread()
        
         
    #private
    def getDataFromThread(self):
        thread_num=16
        i=0
        
        for gg in range(self.batch_size//thread_num):        
            with Manager() as manager:
                Mobj=manager.list([0 for x in range(thread_num)])
                MpList=[]
            
                for ttt in range(thread_num):
                    if self.global_frame==self.dataNum:
                        self.setGlobalFrame(0)            
                    frame=self.global_frame
                
                    MpList.append(Process(target=fct_load,args=(Mobj,ttt,frame,self.dataset_path_preprocess+self.datasetName)))
                    
                    self.global_frame+=1
            
                #do work with threads
                for _ in MpList:
                    _.start()
                for _ in MpList:
                    _.join()
                
                #put depth/ir to batch
                for ttt in range(thread_num):
                    self.cnn_img_depth[i,0,:,:]=np.copy(Mobj[ttt][:,0:self.trainImageSize]) 
                    self.cnn_img_ir[i,0,:,:]=np.copy(Mobj[ttt][:,self.trainImageSize:2*self.trainImageSize])    
                    i+=1
        
        return self.cnn_img_depth,self.cnn_img_ir        
            
            
    
    def setGlobalFrame(self,data):
        self.global_frame=0
    
    def getDataFromRam(self):
        for i in range(self.batch_size):
            if self.global_frame==self.dataNum:
                self.setGlobalFrame(0)
            frame=self.global_frame
            
            self.cnn_img_depth[i,0,:,:]=np.copy(self.cnn_imgs[frame][:,0:self.trainImageSize]) 
            self.cnn_img_ir[i,0,:,:]=np.copy(self.cnn_imgs[frame][:,self.trainImageSize:2*self.trainImageSize])
            
            self.global_frame+=1
        
        return self.cnn_img_depth,self.cnn_img_ir
        
    #it directly load training image/label from file
    def generator_learningData(self):        
        j=0
        while True:            
            i=0
            while True:
                if j==self.dataNum:
                    j=0
                
                frame=self.idx_data[j]
                img_crop_norm=np.load(self.dataset_path_preprocess+self.datasetName+'/%d.npy'%frame)
                
                #put depth/ir to batch
                self.cnn_img_depth[i,0,:,:]=np.copy(img_crop_norm[:,0:self.trainImageSize]) 
                self.cnn_img_ir[i,0,:,:]=np.copy(img_crop_norm[:,self.trainImageSize:2*self.trainImageSize])
                
                #
                j+=1
                i+=1
                if i==self.batch_size:
                    break
                
            yield self.cnn_img_depth,self.cnn_img_ir
    
    def loadPreprocessedDepthImage(self,frame,dataset_type):
        trainImageSize=self.trainImageSize
        img_crop_norm=np.load(self.dataset_path_preprocess+dataset_type+'/%d.npy'%frame)      
        return np.copy(img_crop_norm[:,0:trainImageSize]) 

    def loadPreprocessedIrImage(self,frame,dataset_type):
        trainImageSize=self.trainImageSize
        img_crop_norm=np.load(self.dataset_path_preprocess+dataset_type+'/%d.npy'%frame)
        return np.copy(img_crop_norm[:,trainImageSize:2*trainImageSize])
        
    
    def preprocess_depth(self,depth_seg,trainImageSize,cube):
        com=self.utils.calculateCOM(depth_seg)
        #if com[0]==0 and com[1]==0 and com[2]==0:
        #    return None,None,None,None
        
        com,depth_crop,window=self.utils.refineCOMIterative(depth_seg,com,3)     
        depth_train=self.utils.makeLearningImage(depth_crop,com)
        
        return depth_train,depth_crop,com,window

    def preprocess_ir(self,ir,window):
        ir_crop=ir[window[2]:window[3],window[0]:window[1]]
        ir_crop=cv2.resize(ir_crop,(self.trainImageSize,self.trainImageSize))

        ir_max=np.max(ir_crop)
        ir_min=np.min(ir_crop)     
        ir_train=-1+2*(ir_crop-ir_min)/(ir_max-ir_min)     
               
        return ir_train.copy()

    

if __name__=="__main__":
    '''
    opt='train'
    data_num=10000
    save_enable=False
    '''
    
    opt='test'
    data_num=2000
    save_enable=False
    
    
    ##--setting
    load_filepath='/home/yong/hdd/dataset/depth_ir/'+opt+'/'
    save_filepath='../../../../preprocessed_HIG/'+opt+'/'
    print('loaded image file path: ',load_filepath)
    print('save file path: ',save_filepath)
    
    trainImageSize=128
    d_minimum=0
    d_maximum=500
    
    ##--fixed setting
    fx=475.065948
    fy=475.065857
    cx=315.944855
    cy=245.287079
    cube=np.asarray([250,250,250])
    
    utils=Utils(fx,fy,cx,cy,cube)
    
    imgs_train=np.zeros((trainImageSize,2*trainImageSize),'float32')
    #imgs_train=np.zeros((trainImageSize,2*trainImageSize))
    frame=0
   
    
    for frame in range(data_num):
        depth=cv2.imread(load_filepath+'depth%d.png'%frame,2)
        ir=cv2.imread(load_filepath+'ir%d.png'%frame,2)
        
        depth_seg=depth.copy()
        
        # preprocess (depth)
        depth_seg[depth_seg>d_maximum]=0
        com=utils.calculateCOM(depth_seg)
        com,depth_crop,window=utils.refineCOMIterative(depth_seg,com,3)     
        depth_train=utils.makeLearningImage(depth_crop,com)
        
        # preprocess (ir)
        ir_crop=ir[window[2]:window[3],window[0]:window[1]]
        ir_crop=cv2.resize(ir_crop,(trainImageSize,trainImageSize))
        
        ##debug
        
        ir_crop_vis=np.uint8(cv2.normalize((ir_crop), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))           
        cv2.imshow('ir_crop_vis',ir_crop_vis)
       
        hand_irval=ir[int(com[1]),int(com[0])]
        
        ir_crop[ir_crop<hand_irval-20]=0
        ir_crop[ir_crop>hand_irval+20]=0
        ir_crop_seg=np.uint8(cv2.normalize((ir_crop), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))           
                    
        cv2.imshow('ir_crop_seg',ir_crop_seg)
        
        ##debug
        
        
        ir_max=np.max(ir_crop)
        ir_min=np.min(ir_crop)     
        ir_train=-1+2*(ir_crop-ir_min)/(ir_max-ir_min) 
         
        
        #save image
        imgs_train[:,0:trainImageSize]=depth_train
        imgs_train[:,trainImageSize:2*trainImageSize]=ir_train
            
        cv2.imshow("imgs_train",imgs_train)
        cv2.waitKey(1)    
            
        cv2.imwrite("imgs_train.png",imgs_train)
        
        if save_enable==True:
            np.save(save_filepath+'%d.npy'%frame,imgs_train)
            
        
        
            
 
    
    
 
    
    
        
    
    
    



















