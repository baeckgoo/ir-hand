import pyrealsense2 as rs
import os,sys,inspect
currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
basedir=currentdir+'\\..\\'
sys.path.append(basedir)
#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')

from HPE.model.deepPrior import dpnet
from HIG.model.network import Pix2pix
from Model.fusionNet import FusionNet
from Segmentation.Segmentation import Segmentation

from datasetloader.datasetLoader_uvr import DatasetLoader as datasetloader_UVR
from GrabCut import Grabcut

#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/Utility/')
from Utility.visualize import Visualize_combined_outputs

#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
from datasetloader.datasetLoader_icvl import DatasetLoader as datasetloader_ICVL

import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import progress
import argparse
from datetime import datetime
from tqdm import trange
import csv
import time
from pyquaternion import Quaternion

from multiprocessing import Process, Queue
from Udp import UDP

#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from Sensor.Realsense import Realsense

    
def segment_irhand_contour(ir_orig):
    ir_norm=ir_orig #its type is already uint8.
            
    #contours     
    ret,thresh=cv2.threshold(ir_norm,40,255,0)  #40
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
    #size of maximum contour
    mask=np.ones(ir_orig.shape[:2],dtype='uint8')*255
    areas=[cv2.contourArea(c) for c in contours]
    max_area=np.max(areas)
            
    #erase smaller contour than maximum contour
    for c in contours:
        area=cv2.contourArea(c)
        if area<max_area:
            cv2.drawContours(mask,[c],-1,0,-1)
            
    irhand_seg=cv2.bitwise_and(thresh,ir_norm,mask=mask)
            
    return irhand_seg   
    

#if __name__ == '__main__':
class Gymodule():
    def __init__(self):    
        parser = argparse.ArgumentParser(description='PyTorch hand pose Training')
        args=parser.parse_args()
    
        '''user setting'''
        args.gpu_ids='0'
        args.use_net=['hpe1_orig','hpe2'] #hpe1_orig, hig, hpe1, hpe2
        forward_list=['hpe1_orig','hpe2'] #hpe1_orig, hig_hpe1, hpe2

        

        trained_modelFile_hpe1_orig=basedir+'..\\HPE\\output\\2020_2_26_11_29\\trainedModel_epoch99.pth.tar'
        trained_modelFile_hpe2=basedir+'..\\HPE2\\output\\2020_4_10_18_46\\trainedModel_epoch99.pth.tar'

        '''common setting''' 
        out_list={}
        args.train_net=[]
        args.cgan=True
        
        args.is_train=False
        args.gan_mode='vanilla'
        args.trainImageSize=128
        trainImageSize=128
        args.skeleton_pca_dim=52
        args.skeleton_orig_dim=63
        args.discriminator_reconstruction=False 
        args.test_batch=1    
        hpe_numBlocks=5
        
        #device
        #device = torch.device(torch.device("cuda:%s"%args.gpu_ids if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.device=device
        cudnn.benchmark = True  
        
        #--HPE_orig net setting 
        if 'hpe1_orig' in args.use_net:
            if not 'hpe1_orig' in locals():
                hpe1_orig=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
                checkpoint=torch.load(trained_modelFile_hpe1_orig,map_location=lambda storage, loc: storage)
                hpe1_orig.load_state_dict(checkpoint['state_dict'])
                    
        #--HPE2 net setting
        if 'hpe2' in args.use_net:
            if not 'hpe2' in locals():
                hpe2=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
                checkpoint=torch.load(trained_modelFile_hpe2,map_location=lambda storage, loc: storage)
                hpe2.load_state_dict(checkpoint['state_dict_hpe2'])
                       
        with open(basedir+'..\\pca_52_by_957032_augmented_X2.pickle','rb') as f:
            self.pca=pickle.load(f)
    
        #--fusion net setting
        #fusionnet=FusionNet(hpe1_orig,pix2pix,hpe1,hpe2,None,args)    
        fl=[]
        if 'hpe1_orig' in locals(): fl.append(hpe1_orig) 
        else: fl.append(None)
        if 'pix2pix' in locals(): fl.append(pix2pix) 
        else: fl.append(None)
        if 'hpe1' in locals(): fl.append(hpe1) 
        else: fl.append(None)
        if 'hpe2' in locals(): fl.append(hpe2) 
        else: fl.append(None)
        fusionnet=FusionNet(fl[0],fl[1],fl[2],fl[3],None,args)    
        

        self.fusionnet=fusionnet.to(device) 
    
   
    
        #--dataset of uvr (for utils)
        camera_info={}
        camera_info['fx']=475.065948
        camera_info['fy']=475.065857
        camera_info['cx']=315.944855
        camera_info['cy']=245.287079 
        camera_info['camerawidth']=640
        camera_info['cameraheight']=480
        camera_info['cube']=np.asarray([250,250,250])
        camera_info['trainImageSize']=128
        
        self.trainImageSize=camera_info['trainImageSize']
        self.cube=camera_info['cube']
        camerawidth=camera_info['camerawidth']
        cameraheight=camera_info['cameraheight']
        
        jointnum=21
        d_maximum=500
        
        self.datasetloader_uvr=datasetloader_UVR('..',0,camera_info,'..','..')
        self.utils=self.datasetloader_uvr.utils
             
        #--start
        self.fusionnet.set_mode('eval')
        self.ir_batch = np.zeros((1,1,args.trainImageSize,args.trainImageSize), dtype = np.float32)
        self.depth_batch = np.zeros((1,1,args.trainImageSize,args.trainImageSize), dtype = np.float32)
  
    def run(self,depth_seg,ir_orig):                   
            #preprocess depth image (0.01 SEC)
            depth_train,depth_crop,com,window=self.datasetloader_uvr.preprocess_depth(depth_seg,self.trainImageSize,self.cube)
            
            #segment ir
            '''
            if segment_method=='blueband':
                ir_seg=segmentation.segmentIR(window)
            else:
                ir_seg=segment_irhand_contour(np.uint8(ir_orig))
            
            ir_seg=segment_irhand_contour(np.uint8(ir_orig))
            '''
            
 
            outimg3c=np.uint8(depth_seg)
            outimg3c=cv2.cvtColor(outimg3c,cv2.COLOR_GRAY2BGR)            

            
            self.depth_batch[0,0,:,:]=depth_train
            self.fusionnet.set_input(self.ir_batch,self.depth_batch)
            
            com3d=self.utils.unproject2Dto3D(com) 
            
            self.fusionnet.forward_('hpe1_orig')  # 0.01 sec 
            
            out=self.fusionnet.reconstruct_joints(self.pca,com3d,self.cube,'hpe1_orig','tocpu')
            
            #cv2.cvtColor(rgb_orig,cv2.COLOR_GRAY2BGR)
            outimg3c=self.utils.circle3DJoints2Image(outimg3c,out[0,:]) 
            
            #cv2.imshow('outimg3c',outimg3c)
            #cv2.waitKey(1)
            
            return out,outimg3c
            
            
            

        
        
            