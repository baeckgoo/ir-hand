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

def nothing(x):
    pass

class Utils:
    def __init__(self,jointNum,fx,fy,cx,cy,cube):
        self.jointNum=jointNum
        self.fx=fx
        self.fy=fy
        self.cx=cx
        self.cy=cy
        self.calibMat=np.asarray([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]]) 
        self.cube=cube
        self.rng= np.random.RandomState(23455)
     
    def preprocess_depth(self,depth_orig):
        depth_orig[depth_orig>500]=0
        com=self.calculateCOM(depth_orig)
        com,depth_crop,window=self.refineCOMIterative(depth_orig,com,3)
        depth_train=self.makeLearningImage(depth_crop,com)
        
        self.window=window
        
        return depth_train,depth_crop,com
    
    #it should be run after 'preprocess_depth' due to self.window.
    def preprocess_ir(self,ir_orig,trainImageSize):
        window=self.window
        
        ir_crop=ir_orig[window[2]:window[3],window[0]:window[1]]
        ir_crop=cv2.resize(ir_crop,(trainImageSize,trainImageSize))
        
        ir_max=np.max(ir_crop)
        ir_min=np.min(ir_crop)     
        ir_train=-1+2*(ir_crop-ir_min)/(ir_max-ir_min)
    
        return ir_train
    
    #input (img_crop) should be cropped depth image.  
    def makeLearningImage(self,img_crop,com):
        s=128
        cnnimg=img_crop.copy()
        cnnimg[cnnimg==0]=com[2]+self.cube[2]/2.
        cnnimg=cnnimg-com[2]
        cnnimg=cnnimg/(self.cube[2]/2.)
        
        cnnimg=cv2.resize(cnnimg,(s,s))
        return np.copy(cnnimg)
        
    #will be deleted
    def makeLearningImage_pre(self,img_crop,frame):
        s=128
        cnnimg=cv2.resize(img_crop,(s,s))
        cnnimg[cnnimg==0]=np.max(cnnimg)
        d_max=np.max(cnnimg)
        d_min=np.min(cnnimg)
        if d_max-d_min<1:
            print('[frame:%d]dmax-dmin<1 min%f max:%f'%(frame,d_min,d_max))
        cnnimg=-1+2*(cnnimg-d_min)/(d_max-d_min)        

        return np.copy(cnnimg)
    
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
        jointNum=self.jointNum
        calibMat=self.calibMat
        #pos2d=np.zeros((jointNum,2))
        pos2d=np.copy(ground_3dpos)#np.zeros_like(ground_3dpos)
        
        p2d=np.matmul(calibMat,np.transpose(ground_3dpos)) #(3*3)*(N*3)^T
        pos2d[:,0:2]=np.transpose(p2d[0:2,:]/p2d[2,:])
        
        #print('p2d shape',p2d.shape)
        #print('pos2d shape',pos2d.shape)
        #print('p2d',p2d)
        #print('pos2d',pos2d)
        #jio
        
        #pos2d=np.transpose(pos2d[0:2,:])
        #return pos2d.astype(np.int32)
        return pos2d
    
    def circle2DJoints2Image(self,img,pos2d):
        color=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255]]
        
        for j in range(len(pos2d)):
            x,y=pos2d[j,0],pos2d[j,1]
            x=int(x)
            y=int(y)
            #cid=j//4
            #cv2.circle(img_,(x,y),5,color[cid],-1)
            if j<6:
                cv2.circle(img,(x,y),6,color[0],-1)
            else:
                cid=(j-6)//3
                cv2.circle(img,(x,y),6,color[1+cid],-1)
        
        return img
        
    #shape of pos3d should be 63 
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
                cv2.circle(img_,(x,y),6,color[0],-1)
            else:
                cid=(j-6)//3
                cv2.circle(img_,(x,y),6,color[1+cid],-1)
            
        
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

    def calculateCOM(self,dimg,minDepth=10,maxDepth=1000):
        
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
        
    def detectHand(self,dimg):
        maxDepth=500
        minDepth=10
        steps=10
        dz=(maxDepth-minDepth)/float(steps)
    
        maxContourSize=100
        for j in range(steps):
            part=dimg.copy()
            part[part<j*dz+minDepth]=0
            part[part>(j+1)*dz+minDepth]=0
            part[part!=0]=10
                
            ret,thresh=cv2.threshold(part,1,255,cv2.THRESH_BINARY)
            thresh=thresh.astype(dtype=np.uint8)
            contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(dimg3c, contours, -1, (0,255,0), 3)
            
            for c in range(len(contours)):
                if cv2.contourArea(contours[c])>maxContourSize:
                    maxContourSize=cv2.contourArea(contours[c])
                    
                    #centroid
                    M=cv2.moments(contours[c])
                    cx=int(np.rint(M['m10']/M['m00']))
                    cy=int(np.rint(M['m01']/M['m00']))
                        
                    #crop
                    xstart=int(max(cx-100,0))
                    xend=int(min(cx+100,part.shape[1]-1))
                    ystart=int(max(cy-100,0))
                    yend=int(min(cy+100,part.shape[0]-1))
                        
                    cropped=dimg[ystart:yend,xstart:xend].copy()
                    cropped[cropped<j*dz+minDepth]=0
                    cropped[cropped>(j+1)*dz+minDepth]=0
                    
                    part_selected=part.copy()
            
        if maxContourSize==100:
            return part,[0,0,0,0]
        else:      
            return cropped,[xstart,xend,ystart,yend]
        
    def _relative_joints(self,width,height,ground_2dpos,window3d):
        ground_2dpos_crop= ground_2dpos-[window3d[0,0],window3d[1,0]]
        x=ground_2dpos_crop[:,0]*width/(window3d[0,1]-window3d[0,0])
        y=ground_2dpos_crop[:,1]*height/(window3d[1,1]-window3d[1,0])
        return np.transpose(np.asarray([x,y],'int32')) 
    
    def _generate_hm(self, height, width ,joints, maxlenght):
        num_joints = joints.shape[0]
        hm = np.zeros((num_joints,height, width), dtype = np.float32)
        for i in range(num_joints):
            if not(np.array_equal(joints[i], [-1,-1])):
                s = int(np.sqrt(maxlenght) * maxlenght * 10 / (width*height)) + 2
                hm[i,:,:] = self._makeGaussian(height, width, sigma= s, center= (joints[i,0], joints[i,1]))
            else:
                hm[i,:,:] = np.zeros((height,width))
        return hm      				
		       
    def _makeGaussian(self, height, width, sigma = 3, center=None):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 =  width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0= center[1]
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2) 
    
#augmentation   
    def set_augmentation_parameters(self,sigma_com,rot_range,sigma_sc):
        #sigma_com = 5.
        #rot_range=180.
        #sigma_sc = 0.02
    
        self.off=self.rng.randn(3)*sigma_com # mean:0, var:sigma_com (normal distrib.)
    
        rot=self.rng.uniform(-rot_range,rot_range)
        self.rot=np.mod(rot,360)
               
        self.sc = np.fabs(self.rng.randn(1) * sigma_sc + 1.)
    
    def augment(self,img_seg,joints3d,com):
        img_seg_2,joints3d_2=self.augmentRotation(img_seg,joints3d,com,self.rot)
        com2d,com3d,_=self.augmentTranslation(img_seg_2,joints3d_2,com,self.off)
        cube,window=self.augmentScale(img_seg_2,self.cube,com2d,self.sc)
        
        #--apply augmentation to image 
        xstart,xend,ystart,yend,zstart,zend=window[0],window[1],window[2],window[3],window[4],window[5]
        img_crop=self.crop(img_seg_2,xstart,xend,ystart,yend,zstart,zend)
        img_train=self.makeLearningImage(img_crop,com)
    
        #--apply augmentation to joints
        joints3d_norm=np.concatenate(joints3d_2)-np.tile(com3d,(1,21))
        joints3d_norm2=joints3d_norm/(cube[2]/2.)
        joints3d_norm3=np.clip(joints3d_norm2,-1,1)
    
    
        return  img_seg_2,img_train,com2d,com3d,window,joints3d_2,joints3d_norm3
    #def applyAugmentation2Image(self):
        
    #def applyAugmentation2Joints(self):
        
        
    def augmentTranslation(self,img_seg,joints3d,com,off):
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
                
    def augmentRotation(self,img_seg,joints3d,com,rot):
        if np.allclose(rot,0.0):
            return img_seg,joints3d
        
        M=cv2.getRotationMatrix2D((com[0],com[1]),-rot,1)
        
        img_seg_rot = cv2.warpAffine(img_seg, M, (img_seg.shape[1], img_seg.shape[0]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        #img_seg_rot=imutils.rotate(img_seg,rot)
        #TIME=time.time()
        joints3d_=np.reshape(joints3d,(21,3))
        joints2d=self.projectAlljoints2imagePlane(joints3d_)
        
        data2d=np.zeros_like(joints2d)
        for k in range(data2d.shape[0]):
            data2d[k]= self.rotatePoint2D(joints2d[k],com,rot)
               
        data3d=np.zeros_like(data2d)
        for k in range(data2d.shape[0]):
            data3d[k]=self.unproject2Dto3D(data2d[k])
        #print(time.time()-TIME) 
        return img_seg_rot,data3d
        
    
    def rotatePoint2D(self,p1,center,angle):
        """
        Rotate a point in 2D around center
        :param p1: point in 2D (u,v,d)
        :param center: 2D center of rotation
        :param angle: angle in deg
        :return: rotated point
        """
        alpha = angle * np.pi / 180.
        pp = p1.copy()
        pp[0:2] -= center[0:2]
        pr = np.zeros_like(pp)
        pr[0] = pp[0]*np.cos(alpha) - pp[1]*np.sin(alpha)
        pr[1] = pp[0]*np.sin(alpha) + pp[1]*np.cos(alpha)
        pr[2] = pp[2]
        ps = pr
        ps[0:2] += center[0:2]
        return ps
        
    
        
    
class DatasetLoader:
    def __init__(self,num_class):
        fx=475.065948
        fy=475.065857
        cx=315.944855
        cy=245.287079
        self.calibMat=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])   
        self.jointNum=21
        self.num_class=num_class#pcadim
        self.camerawidth=640
        self.cameraheight=480
        self.trainImageSize=128
        self.cube=np.asarray([250,250,250])
        
        self.utils=Utils(self.jointNum,fx,fy,cx,cy,self.cube)  
        self.rng= np.random.RandomState(23455)
        
        
        #self.fulldatanum_train=957032#119629#957032
        #self.fulldatanum_test=295510
        #---to make offline trainingset.begin
        '''
        self.traindataNum=957032#957032#239258
        self.validateNum=295510
        self.idx_train=None #np.arange(1)
        self.idx_validate=None #np.arange(1)
        '''
        #---to make offline trainingset.end
        
        
    def load(self,filepath):
        self.filepath=filepath
        
        #load
        if filepath['pca']!=None:
            filepath_pca=filepath['pca']  
            with open(filepath_pca,'rb') as f:
                self.pca=pickle.load(f)
        if filepath['gt_train']!=None:
            filepath_gt_train=filepath['gt_train']
        if filepath['gt_test']!=None:
            filepath_gt_test=filepath['gt_test']    
        if filepath['gt_full_train']!=None:
            filepath_gt_full_train=filepath['gt_full_train']   
        if filepath['gt_full_test']!=None:
            filepath_gt_full_test=filepath['gt_full_test']
        if filepath['image_train']!=None:
            self.image_train=np.load(filepath['image_train'])
        if filepath['image_test']!=None:
            self.image_test=np.load(filepath['image_test'])   
           
        #
        if filepath['gt_train']!=None:
            with open(filepath_gt_train,'rb') as f:
                self.dataset_dict_train=pickle.load(f)
            self.traindataNum=len(self.dataset_dict_train['frames_valid'])#119629
            self.idx_train=self.dataset_dict_train['frames_valid']
        if filepath['gt_test']!=None:        
            with open(filepath_gt_test,'rb') as f:
                self.dataset_dict_test=pickle.load(f)
            self.validateNum=len(self.dataset_dict_test['frames_valid'])  #295510
            self.idx_validate=self.dataset_dict_test['frames_valid']
        
 
        if filepath['gt_full_train']!=None:
            gt=self.loadFullLabels(filepath_gt_full_train)
            self.joint3d_gt_train=np.asarray(gt,'float')
        if filepath['gt_full_test']!=None:
            gt=self.loadFullLabels(filepath_gt_full_test)
            self.joint3d_gt_test=np.asarray(gt,'float')            
            
        print('dataset loader ..done')
        
    def generator_imageData(self,batch_size,opt):
        if opt=='train':
            np.random.shuffle(self.idx_train)
        elif opt=='validate':
            np.random.shuffle(self.idx_validate)
        
        trainImageSize=self.trainImageSize
        cnn_img = np.zeros((batch_size,1,trainImageSize,trainImageSize), dtype = np.float32)
        
        j=0
        while True:
            i=0
            while True:
                if opt=='train':
                    if j==self.traindataNum:
                        j=0
                elif opt=='validate':
                    if j==self.validateNum:
                        j=0
                    
                if opt=='train':
                    frame=self.idx_train[j]
                elif opt=='validate':
                    frame=self.idx_validate[j]
                
                self.frame=frame
                
                if opt=='train':
                    if self.filepath['image_train']==None:
                        img_crop_norm=np.load(self.filepath['image_train_each']%frame)
                    else:
                        img_crop_norm=self.image_train[frame,:,:]               
                                                     
                elif opt=='validate':
                    if self.filepath['image_test']==None:
                        img_crop_norm=np.load(self.filepath['image_test_each']%frame)
                    else:
                        img_crop_norm=self.image_test[frame,:,:]               
                                                        
                cnn_img[i,0,:,:]=np.copy(img_crop_norm)
                              
                j+=1
                i+=1
                if i==batch_size:
                    break
                
            yield cnn_img
        
    
    #it directly load training image/label from file
    def generator_learningData(self,batch_size,opt,augment_enable,aug_ratio):
        if opt=='train':
            np.random.shuffle(self.idx_train)
        elif opt=='validate':
            np.random.shuffle(self.idx_validate)
        elif opt=='test_error':
            np.random.shuffle(self.idx_validate)
            
        #print(self.idx_train[0])
        
        trainImageSize=self.trainImageSize
        jointNum=self.jointNum
        cnn_img = np.zeros((batch_size,1,trainImageSize,trainImageSize), dtype = np.float32)
        cnn_gt=np.zeros((batch_size,self.num_class),np.float32)
        cnn_gt_joint3d=np.zeros((batch_size,3*self.jointNum),np.float32)
        com3d=np.zeros((batch_size,3))
        
        
        j=0
        while True:
            i=0
            while True:
                if opt=='train':
                    if j==self.traindataNum:
                        j=0
                elif opt=='validate':
                    if j==self.validateNum:
                        j=0
                elif opt=='test_error':
                    if j==self.validateNum:
                        j=0
                    
                if opt=='train':
                    frame=self.idx_train[j]
                elif opt=='validate':
                    frame=self.idx_validate[j]
                elif opt=='test_error':
                    frame=self.idx_validate[j]
                
                self.frame=frame
                
                if opt=='train':
                    if self.filepath['image_train']==None:
                        img_crop_norm=np.load(self.filepath['image_train_each']%frame)
                    else:
                        img_crop_norm=self.image_train[frame,:,:]               
                  
                    cnn_gt[i]=np.copy(self.dataset_dict_train['label_embed'][frame])    
                    cnn_gt_joint3d[i]=None#np.copy(self.joint3d_gt_train[frame]) 
                    com3d[i]=np.copy(self.dataset_dict_train['com3d'][frame])   
                
                elif opt=='validate':
                    if self.filepath['image_test']==None:
                        img_crop_norm=np.load(self.filepath['image_test_each']%frame)
                    else:
                        img_crop_norm=self.image_test[frame,:,:]               
                    cnn_gt[i]=np.copy(self.dataset_dict_test['label_embed'][frame]) 
                    cnn_gt_joint3d[i]=np.copy(self.joint3d_gt_test[frame]) 
                    com3d[i]=np.copy(self.dataset_dict_test['com3d'][frame])   
                
                elif opt=='test_error':
                    if self.filepath['image_test']==None:
                        img_crop_norm=np.load(self.filepath['image_test_each']%frame)
                    else:
                        img_crop_norm=self.image_test[frame,:,:]      
                    cnn_gt[i]=np.copy(self.dataset_dict_test['label_embed'][frame])  
                    cnn_gt_joint3d[i]=np.copy(self.joint3d_gt_test[frame]) 
                    com3d[i]=np.copy(self.dataset_dict_test['com3d'][frame])   
                
                    
                    
                cnn_img[i,0,:,:]=np.copy(img_crop_norm)
                
                
                j+=1
                i+=1
                if i==batch_size:
                    break
               
            yield cnn_img, cnn_gt, cnn_gt_joint3d, com3d
            
    #it make training image/label in online.
    def generator_learningData_aug(self,batch_size,opt,augment_enable,aug_ratio):
        if opt=='train':
            np.random.shuffle(self.idx_train)
        elif opt=='validate':
            np.random.shuffle(self.idx_validate)
        elif opt=='test_error':
            np.random.shuffle(self.idx_validate)
            
        #print(self.idx_train)
        trainImageSize=self.trainImageSize
        #htmapSize=self.htmapSize
        jointNum=self.jointNum
        cnn_img = np.zeros((batch_size,1,trainImageSize,trainImageSize), dtype = np.float32)
        cnn_gt=np.zeros((batch_size,self.num_class),np.float32)
        cnn_gt_joint3d=np.zeros((batch_size,3*self.jointNum),np.float32)
        com3d=np.zeros((batch_size,3))
        
        j=0
        while True:
            i=0
            while True:
                if opt=='train':
                    if j==self.traindataNum:
                        j=0
                elif opt=='validate':
                    if j==self.validateNum:
                        j=0
                    
                if opt=='train':
                    frame=self.idx_train[j]
                elif opt=='validate':
                    frame=self.idx_validate[j]
                
                self.frame=frame
                
                if opt=='train':
                    if self.filepath['image_train']==None:
                        img_crop_norm=np.load(self.filepath['image_train_each']%frame)
                    else:
                        img_crop_norm=self.image_train[frame,:,:]               
                  
                    cnn_gt[i]=np.copy(self.dataset_dict_train['label_embed'][frame])    
                    cnn_gt_joint3d[i]=np.copy(self.joint3d_gt_train[frame]) 
                    com3d[i]=np.copy(self.dataset_dict_train['com3d'][frame])   
                
                elif opt=='validate':
                    if self.filepath['image_test']==None:
                        img_crop_norm=np.load(self.filepath['image_test_each']%frame)
                    else:
                        img_crop_norm=self.image_test[frame,:,:]               
                    cnn_gt[i]=np.copy(self.dataset_dict_test['label_embed'][frame]) 
                    cnn_gt_joint3d[i]=np.copy(self.joint3d_gt_test[frame]) 
                    com3d[i]=np.copy(self.dataset_dict_test['com3d'][frame]) 
                    
                #augment
                
                
                if augment_enable=='aug':
                    print('..not implemented..')
                
                ##########################################
                self.frame=frame
                img=self.loadImage("images_train/",frame)
                joint3d=self.joint3d_gt_train[frame,:]
                img_seg,isvalid_bool=self.segmentHand(img,joint3d)
                
                if isvalid_bool==False:
                    j+=1
                    continue
                
                joints3d_gt=self.joint3d_gt[frame,:]
                
                #put preprocessed data to training array
                
                #++0.003sec
                com=self.utils.calculateCOM(img_seg)
                com,img_crop,window=self.utils.refineCOMIterative(img_seg,com,3)
                cube=self.cube
                #++
                if augment_enable=='aug':
                    if i%aug_ratio==0:
                        img_crop_norm=self.utils.makeLearningImage(img_crop,frame)  #0.003 sec
                        
                        com3d=self.utils.unproject2Dto3D(com)   
                        joints3d_norm=joints3d_gt-np.tile(com3d,(1,jointNum))
                        joints3d_norm2=joints3d_norm/(cube[2]/2.)
                        joints3d_norm3=np.clip(joints3d_norm2,-1,1)  
                        trainlabel_embed=self.pca.transform(joints3d_norm3)
                        
                    else:
                        #augmentation  
                        sigma_com = 5.
                        off=self.rng.randn(3)*sigma_com # mean:0, var:sigma_com (normal distrib.)
                        
                        rot_range=180.
                        rot=self.rng.uniform(-rot_range,rot_range)
                        rot=np.mod(rot,360)
        
                        sigma_sc = 0.02
                        sc = np.fabs(self.rng.randn(1) * sigma_sc + 1.)
                               
                        #rot (0.001sec)                       
                        img_seg,joints3d=self.utils.augmentRotation(img_seg,joints3d_gt,com,rot)
                        
                        #translation (0.0001sec)                
                        com2d,com3d,window=self.utils.augmentTranslation(img_seg,joints3d,com,off)
                          
                        #scale (0sec)                       
                        cube=self.cube
                        cube,window=self.utils.augmentScale(img_seg,cube,com2d,sc)
                         
                        #apply augmentation                       
                        xstart,xend,ystart,yend,zstart,zend=window[0],window[1],window[2],window[3],window[4],window[5]
                        img_crop=self.utils.crop(img_seg,xstart,xend,ystart,yend,zstart,zend)
                        img_crop_norm=self.utils.makeLearningImage(img_crop,frame)  #0.003 sec  
                          
                        joints3d_norm=joints3d_gt-np.tile(com3d,(1,jointNum))
                        joints3d_norm2=joints3d_norm/(cube[2]/2.)
                        joints3d_norm3=np.clip(joints3d_norm2,-1,1)
                        trainlabel_embed=self.pca.transform(joints3d_norm3) 

                        j+=1
                else:
                    img_crop_norm=self.utils.makeLearningImage(img_crop,frame)  #0.003 sec
                        
                    com3d=self.utils.unproject2Dto3D(com)   
                    joints3d_norm=joints3d_gt-np.tile(com3d,(1,jointNum))
                    joints3d_norm2=joints3d_norm/(cube[2]/2.)
                    joints3d_norm3=np.clip(joints3d_norm2,-1,1)  
                    trainlabel_embed=self.pca.transform(joints3d_norm3)    
                    j+=1
                
                #put trained data to batch
                train_img[i,0,:,:]=np.copy(img_crop_norm)
                train_gt[i]=np.copy(trainlabel_embed)                    
                i+=1
                if i==batch_size:
                    break
                
            yield train_img, train_gt
               
        
    def getBoundBox(self):
        return np.copy(self.window3d)
      
    def segmentHand(self,img,joint3d):        
        #self.window3d : pixel,pixel,mm
        ground_3dpos_=np.copy(joint3d)
        ground_3dpos=np.reshape(ground_3dpos_,(self.jointNum,3))

        calibMat=self.calibMat
        ground_2dpos=np.matmul(calibMat,np.transpose(ground_3dpos))/ground_3dpos[:,2]
        ground_2dpos=np.transpose(ground_2dpos)
        
        rest=[30,30,50]
        window3d=np.asarray([[ground_2dpos[:,0].min()-rest[0],ground_2dpos[:,0].max()+rest[0]], #x
                             [ground_2dpos[:,1].min()-rest[1],ground_2dpos[:,1].max()+rest[1]], #y
                             [ground_3dpos[:,2].min()-rest[2],ground_3dpos[:,2].max()+rest[2]]]) #z
        
        w=self.camerawidth
        h=self.cameraheight
        
        x=[int(window3d[0,0]),int(window3d[0,1])]
        y=[int(window3d[1,0]),int(window3d[1,1])]
        z=[int(window3d[2,0]),int(window3d[2,1])]
        #print(x,y,z)
        
        if x[0]<=-rest[0] or y[0]<=-rest[1] or x[1]>(w-1+rest[0]) or y[1]>(h-1+rest[1]):
            return None,False
        
        #bound
        x[0]=min(max(x[0],0),w-1) 
        y[0]=min(max(y[0],0),w-1)
        
        x[1]=min(max(x[1],0),h-1) 
        y[1]=min(max(y[1],0),h-1) 
        #print(x,y,z)
        
        
        '''TIME consuming'''
        img_seg=img.copy()
        img_seg[:,0:x[0]]=0
        img_seg[:,x[1]:w]=0
        img_seg[0:y[0],:]=0
        img_seg[y[1]:h,:]=0
        img_seg[img_seg<z[0]]=0
        img_seg[img_seg>z[1]]=0
        '''TIME consuming'''
        
        return img_seg,True
    
    def showBox(self,img,window):    
        p0=(window[0],window[2])
        p1=(window[1],window[2])
        p2=(window[0],window[3])
        p3=(window[1],window[3])
        cv2.line(img,p0,p1,(255,0,0),5)
        cv2.line(img,p0,p2,(255,0,0),5)
        cv2.line(img,p1,p3,(255,0,0),5)
        cv2.line(img,p2,p3,(255,0,0),5)
        #cv2.imshow('box',np.uint8(img))
        #cv2.waitKey(1)
        #xstart,xend,ystart,yend,zstart,zend
        return img

    def makeHeatMap(self,frame):
        ground_3dpos_=np.copy(self.joint3d_gt[frame,:])
        ground_3dpos=np.reshape(ground_3dpos_,(self.jointNum,3))
        ground_2dpos=self.utils.projectAlljoints2imagePlane(ground_3dpos)
        
        #window2d=datasetloader.window2d
        window3d=self.getBoundBox()
        s=self.htmapSize
        new_j=self.utils._relative_joints(s,s,ground_2dpos,window3d)
        self.new_j=new_j
        hm = self.utils._generate_hm(s, s, new_j, s)
        return hm
    
    def splitDataset(self,kfold,k_idx):
        ratio=self.traindataNum//kfold
        
        #full dataset
        mylist=np.arange(self.traindataNum)
        #validation dataset
        self.idx_valid = mylist[k_idx*ratio:(k_idx+1)*ratio]
        #train dataset
        if k_idx==0:
            self.idx_train = mylist[(k_idx+1)*ratio:]
        else:
            idx_train_0=mylist[0:k_idx*ratio]
            idx_train_1=mylist[(k_idx+1)*ratio:]
            self.idx_train=np.concatenate((idx_train_0,idx_train_1))
                    
    def loadImage(self,folder,frame):
        image_fname=folder+"image_D%08d.png"%(frame+1)
        img=cv2.imread(image_fname,2)

        return img
    
    
    def loadFullLabels(self,label_fname):
        csv_file=open(label_fname,'r')
        csv_reader=csv.reader(csv_file,delimiter='\t')
    
        frame=0
        out=[]
        for line in csv_reader:
            #self.joint3d_gt[frame,:]=line[1:-1]
            out.append(line[1:-1])
            frame+=1
            
        csv_file.close()
        
        return out
    
    '''
    def loadFullLabels(self,num,label_fname):
        csv_file=open(label_fname,'r')
        csv_reader=csv.reader(csv_file,delimiter='\t')
    
        frame=0
        out=[]
        for line in csv_reader:
            #self.joint3d_gt[frame,:]=line[1:-1]
            out.append(line[1:-1])
            frame+=1
            if frame==num:
                break
            
        csv_file.close()
        
        return out
    '''         

    def showLabel(self,frame,img):
        jointNum=self.jointNum
        calibMat=self.calibMat
        
        ground_3dpos=np.copy(self.joint3d_gt[frame,:])
        ground_3dpos=np.reshape(ground_3dpos,(jointNum,3))
        
        img2=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        color=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255]]
        for i in range(jointNum):
            pos3d=np.asarray([ground_3dpos[i,0],ground_3dpos[i,1],ground_3dpos[i,2]])
            pos=np.matmul(calibMat,pos3d)/pos3d[2]

            if i<6:
                cv2.circle(img2,(int(pos[0]),int(pos[1])),5,(255,0,255),-1)
            else:
                colorid=int((i-6)/3)
                cv2.circle(img2,(int(pos[0]),int(pos[1])),5,color[colorid],-1)
    
        cv2.imshow("label",np.uint8(img2))
        cv2.waitKey(1)
    
    
def calculateRotation(a,b):
    #calculate rotation from reference frame.
    avg_a=np.mean(a,0)
    avg_b=np.mean(b,0)
    
    A=np.mat(a-avg_a)
    B=np.mat(b-avg_b)
    
    H=np.transpose(A)*B #  ref-> target  or target->ref
    #H=np.transpose(B)*A #  ref-> target  or target->ref
    
    U,S,Vt=np.linalg.svd(H)
    R=Vt.T*U.T
    print('R',R)
    # special reflection case
    if np.linalg.det(R) < 0:
        print( "Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T * U.T
    
    return Quaternion(matrix=R)  

def show3Dposition(img,position3d,jointNum,calibMat):
        
    img2=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    color=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,255,255],[0,255,255]]
    for i in range(jointNum):
        pos3d=np.asarray([position3d[i,0],position3d[i,1],position3d[i,2]])
        pos=np.matmul(calibMat,pos3d)/pos3d[2]
        cv2.circle(img2,(int(pos[0]),int(pos[1])),5,color[i],-1)
        
    cv2.imshow("rot",np.uint8(img2))
    cv2.waitKey(1)
 
    
    
#if __name__=="__main__":
def test_internalcode():
    frame=152  #12435,  91040, 62505, 42275, 87439  , 20545

    #make quaternion data
    #setting
    jointNum=21
    width=int(640)
    height=int(480)
    trainImageSize=256
    htmapSize=256
    batchsize=1
    #test to make training dataset
    
    datasetloader=DatasetLoader("../../../dataset/HANDS17/training/")
    #shuffler=datasetloader.shuffleGenerator()
    
    batch_size=16
    stacks=1
    joints_num=21
    
    
    while True:
        #load
        #frame=next(shuffler)
        img=datasetloader.loadImage("images_train/",frame)
        ground_3dpos_=np.copy(datasetloader.joint3d_gt[frame,:])
        ground_3dpos=np.reshape(ground_3dpos_,(datasetloader.jointNum,3))
        ground_2dpos=datasetloader.utils.projectAlljoints2imagePlane(ground_3dpos)
        #print(ground_2dpos)
        #plt.imshow(img)
        
        #segment hand
        joint3d=datasetloader.joint3d_gt_train[frame,:]
        img,isvalid_bool=datasetloader.segmentHand(img,joint3d)
        
        if isvalid_bool==False:
            print("not valid data frame")
            break
        plt.imshow(img)
        
        #com
        com=datasetloader.utils.calculateCOM(img)
        com,img_crop,window=datasetloader.utils.refineCOMIterative(img,com,3)
        
        #normalize for cnn image
        s=datasetloader.trainImageSize
        img_crop[img_crop==0]=np.max(img_crop)
        d_max=np.max(img_crop)
        d_min=np.min(img_crop)
        if d_max-d_min<1:
            print('dmax-dmin<! frame%d min%f max:%f'%(frame,d_min,d_max))
        img_crop=-1+2*(img_crop-d_min)/(d_max-d_min)        
        
       # plt.imshow(img_crop)
        
        cv2.line(img,(window[0],window[2]),(window[1],window[2]),(255,0,0),5)
        cv2.line(img,(window[0],window[2]),(window[0],window[3]),(255,0,0),5)
        cv2.line(img,(window[1],window[2]),(window[1],window[3]),(255,0,0),5)
        cv2.line(img,(window[0],window[3]),(window[1],window[3]),(255,0,0),5)
        cv2.circle(img,(int(com[0]),int(com[1])),5,(255,0,0),5,-1)
        
        
        #plt.imshow(img)
        
        com3d=datasetloader.utils.unproject2Dto3D(com)
                

        break
    
#if __name__=="__main__":
def test_generator_learningData():
    frame=0##7

    #make quaternion data
    #setting
    jointNum=21
    width=int(640)
    height=int(480)
    trainImageSize=256
    htmapSize=64
    batchsize=32
    #test to make training dataset
    
    datasetloader=DatasetLoader("../../../dataset/HANDS17/training/")
    #shuffler=datasetloader.shuffleGenerator()
    
    batch_size=64
    joints_num=21
    
    for epoch in range(1):  
        print('epoch..',epoch)
        generator=datasetloader.generator_learningData(batch_size,'train','noaug',0)
                   
        iter_num_train=len(datasetloader.idx_train)//batch_size
        for i in range(iter_num_train):
        #progressbar=trange(iter_num_train)
        #for i in progressbar:
            #start_time=time.time()
            img_train,gt_train=next(generator)   
            #print('time:',time.time()-start_time)
            jio
            """
            #cv2.imshow('img_train',img_train[0,0,:,:])
            gt_train_debug=np.zeros((htmapSize,htmapSize))
            for j in range(joints_num):
                gt_train_=gt_train[0,j,:,:].copy()
                maxval=np.max(gt_train_)
                gt_train_[gt_train_<maxval*0.9]=0
                gt_train_debug+=gt_train_
            """
        #cv2.imshow("gt_train",gt_train_debug)
          
      
        
    cv2.waitKey(1)
        
        
    

#if __name__=="__main__":        
def save_com():    
    datasetloader=DatasetLoader("../../../dataset/HANDS17/training/")
    
    savefile="../../../dataset/HANDS17/training/com.txt"
    
    #write
    f = open(savefile, mode='w', encoding='utf-8',newline='')
    wr=csv.writer(f)
    
    progressbar=trange(957032,leave=True)
    frames_notvalid=[]
    for frame in progressbar:
        img=datasetloader.loadImage("images_train/",frame)
        joint3d=datasetloader.joint3d_gt_train[frame,:]
        img_seg,isvalid_bool=datasetloader.segmentHand(img,joint3d)
        if isvalid_bool==False:
            wr.writerow([0,0,0])
            frames_notvalid.append(frame)
            continue
        
        com=datasetloader.utils.calculateCOM(img_seg)
        com,img_crop,window=datasetloader.utils.refineCOMIterative(img_seg,com,3)    
        com3d=datasetloader.utils.unproject2Dto3D(com)    
        wr.writerow(com3d)    
    f.close()
    
    with open('../../../dataset/HANDS17/training/notvalidframes.pickle','wb') as f:
        pickle.dump(frames_notvalid,f,pickle.HIGHEST_PROTOCOL)
     
    #read
    """
    for frame in range(10):
        csv_file=open(savefile,'r')
        csv_reader=csv.reader(csv_file)

        label_csv=[]

        for row in csv_reader:
            label_csv.append(row)
        csv_file.close()

    label=np.asarray(label_csv,'float')
    """
      
#if __name__=="__main__":        
def test_time_make_dataset():
    datasetloader=DatasetLoader("../../../dataset/HANDS17/training/")
    
    frame=0
    img=datasetloader.loadImage("images_train/",frame)    
    joints3d_gt=datasetloader.joint3d_gt[frame,:]
    cube=datasetloader.cube
    jointNum=datasetloader.jointNum
    
    #test
    joint3d_gt=datasetloader.joint3d_gt_train[frame,:]
    img_seg,isvalid_bool=datasetloader.segmentHand(img,joint3d_gt) #0.0006sec->9.5min
    TIME=time.time()
    com=datasetloader.utils.calculateCOM(img_seg) #0.004sec ->  63 min
    print(time.time()-TIME)
    com,img_crop,window=datasetloader.utils.refineCOMIterative(img_seg,com,3) #0.002sec->31 min
    
    datasetloader.utils.makeLearningImage(img_crop,frame)#0.0006sec->9.5min
    
    #test (augmentation)
    '''
    
    sigma_com = 5.
    off=datasetloader.rng.randn(3)*sigma_com # mean:0, var:sigma_com (normal distrib.)
                         
    rot_range=180.
    rot=datasetloader.rng.uniform(-rot_range,rot_range)
    rot=np.mod(rot,360)
        
    sigma_sc = 0.02
    sc = np.fabs(datasetloader.rng.randn(1) * sigma_sc + 1.)
    
                   
    #rot (0.002sec) -> 31 min x10= 310min 
    TIME=time.time()                     
    img_seg_rot,joints3d=datasetloader.utils.augmentRotation(img_seg,joints3d_gt,com,rot)
    print(time.time()-TIME)   
    plt.imshow(img_seg_rot)       
    
    #torch-based augmentation
    data_batch=np.zeros((2,1,480,640))
    data_batch[0,0,:,:]=img_seg
    data_batch[1,0,:,:]=img_seg
    
    input=torch.FloatTensor(data_batch)
    torchvision.transforms.functional.rotate(,30)
    '''
  
    '''
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10)
    ])
    
    TIME=time.time()
    img_seg2=content_transform(np.asarray(img_seg,'float'))
    print(time.time()-TIME)
    #plt.imshow(img_seg2)
    '''

    
if __name__=="__main__":        
#def make_offline_trainingset():
    
    
    
    #select
    '''
    option="train"
    ratio="eighth"
    save_individually=False
    '''
    
    option="test"
    ratio="full"
    save_individually=True
    
    
    if option=="train" and ratio=="eighth":
        datanumber=119629 #295510#957032
        
    if option=="test" and ratio=="full":
        datanumber=295510
    
  
    if not 'datasetloader' in locals():
        datasetloader=DatasetLoader()   
        filepath={}
        filepath['pca']="/home/yong/hdd/dataset/HANDS17/training/preprocessed/pca_45.pickle"
        filepath['gt_train']=None
        filepath['gt_test']=None
        filepath['gt_full_train']="/home/yong/hdd/dataset/HANDS17/training/original/Training_Annotation.txt"
        filepath['gt_full_test']="/home/yong/hdd/dataset/HANDS17/training/original/Test_Annotation.txt"
        
        filepath['image_train']=None
        filepath['image_train_each']=None
        filepath['image_test']=None
        filepath['image_test_each']=None
        
        datasetloader.load(filepath)
        
        
    if save_individually==True:
        savepath="../../preprocessed/"
        savepath_img="../../preprocessed/images_"+option+"/"
        savepath_dict="../../preprocessed/dataset_dict_"+option+".pickle"
    else:
        savepath="/home/yong/hdd/dataset/HANDS17/training/preprocessed/"
        savepath_img="/home/yong/hdd/dataset/HANDS17/training/preprocessed/images_"+option+"_"+ratio+".npy"
        savepath_dict="/home/yong/hdd/dataset/HANDS17/training/preprocessed/dataset_dict_"+option+"_"+ratio+".pickle"
        
    print('savepath_dict:',savepath_dict)    
    
    #saved dataset
    dataset_dict={}
    dataset_dict['frames_valid']=[]
    dataset_dict['com']=[]
    dataset_dict['com3d']=[]
    dataset_dict['label_embed']=[]
    dataset_dict['window']=[]
    if save_individually==False:
        img_crop_norm_bag=np.zeros((datanumber,128,128))
    
    
    jointNum=datasetloader.jointNum
    #119629
    
    progressbar=trange(datanumber,leave=True)
    for frame in progressbar:
    #for frame in range(10):
        if option=="train":  
            img=datasetloader.loadImage("/home/yong/hdd/dataset/HANDS17/training/original/images_train/",frame)
        elif option=="test":
            img=datasetloader.loadImage("/home/yong/hdd/dataset/HANDS17/training/original/images_test/",frame) 
        
        if option=="train":
            joints3d_gt=datasetloader.joint3d_gt_train[frame,:]
        elif option=="test":
            joints3d_gt=datasetloader.joint3d_gt_test[frame,:]
            
        img_seg,isvalid_bool=datasetloader.segmentHand(img,joints3d_gt)
        if isvalid_bool==False:
            dataset_dict['com'].append(None)
            dataset_dict['com3d'].append(None)
            dataset_dict['window'].append(None)
            dataset_dict['label_embed'].append(None)
            continue
        
        dataset_dict['frames_valid'].append(frame)
        
            
        cube=datasetloader.cube
       
        #make com
        com=datasetloader.utils.calculateCOM(img_seg)
        com,img_crop,window=datasetloader.utils.refineCOMIterative(img_seg,com,3)
        dataset_dict['com'].append(com)
        dataset_dict['window'].append(window)
        
        #make cnn image
        img_crop_norm=datasetloader.utils.makeLearningImage(img_crop,frame)
        if save_individually==True:
            np.save(savepath_img+"%d.npy"%frame,img_crop_norm)
        else:
            img_crop_norm_bag[frame,:,:]=img_crop_norm
        
        
        #make train label        
        
        com3d=datasetloader.utils.unproject2Dto3D(com)   
        joints3d_norm=joints3d_gt-np.tile(com3d,(1,jointNum))
        joints3d_norm2=joints3d_norm/(cube[2]/2.)
        joints3d_norm3=np.clip(joints3d_norm2,-1,1)  
        trainlabel_embed=datasetloader.pca.transform(joints3d_norm3) 
        dataset_dict['label_embed'].append(trainlabel_embed)
        dataset_dict['com3d'].append(com3d)
        
    
    #save preprocessed image 
    if save_individually==False:
        np.save(savepath_img,img_crop_norm_bag)
        
    
    #save label
    with open(savepath_dict,'wb') as f:
        pickle.dump(dataset_dict,f,pickle.HIGHEST_PROTOCOL)
                 
    #load label
    with open(savepath_dict,'rb') as f:
        dataset_load=pickle.load(f)
     
        
     
    
