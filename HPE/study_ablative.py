import torch
import sys
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np

#from datasetloader.datasetLoader_icvl import DatasetLoader
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
from datasetLoader_icvl import DatasetLoader
#import loss as losses
from utils.evaluation import evaluateEmbeddingError

import torch.backends.cudnn as cudnn

from tqdm import tqdm
from tqdm import trange
import time
import pickle
import numpy as np

# select proper device to run
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33
#criterion= losses.JointsMSELoss().to(device)

#frane 24 is strange? compressed error?
if __name__=="__main__":
    option="test"
    batch_size=128
    
    ###---select history---###   
    if not 'position_estimated' in locals():
        loadfolder='/home/yong/hdd/HPE/output/2019_12_29_21_11/'
        f=open(loadfolder+'position_estimated.txt','r')
        lines=[]
        for line in f:
            lines.append(line.split(','))
        position_estimated=np.asarray(lines,'float')
        
    frame_num=len(position_estimated)
            
    ###--dataset--###
    if not 'datasetloader' in locals():
        datasetloader=DatasetLoader(1)
        filepath={}
        filepath['pca']=None
        filepath['gt_train']=None#"/home/yong/hdd/dataset/HANDS17/training/preprocessed/dataset_dict_train_eighth_45.pickle"
        filepath['gt_test']=None
        filepath['gt_full_train']=None#"/home/yong/hdd/dataset/HANDS17/training/original/Training_Annotation.txt"
        filepath['gt_full_test']="/home/yong/ssd/dataset/HANDS17/training/original/Test_Annotation.txt"
                
        filepath['image_train']=None#"/home/yong/hdd/dataset/HANDS17/training/preprocessed/images_train_eighth.npy"
        filepath['image_train_each']=None#"../preprocessed/images_train_eighth/%d.npy"
        filepath['image_test']=None
        filepath['image_test_each']="/home/yong/ssd/dataset/preprocessed_HPE/images_test/%d.npy"
        
        datasetloader.load(filepath)    
    position_gt=datasetloader.joint3d_gt_test
    
    ###--- show results (images)---###
    frame=0
    
    for frame in range(100):
        img=datasetloader.loadImage("/home/yong/ssd/dataset/HANDS17/training/original/images_%s/"%option,frame)
        img=np.uint8(cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))           
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        
        img_save=np.zeros((480,640*2,3))
        img_save[:,0:img.shape[1],:]=datasetloader.utils.circle3DJoints2Image(img,datasetloader.joint3d_gt_test[frame,:])
        img_save[:,img.shape[1]:2*img.shape[1],:]=datasetloader.utils.circle3DJoints2Image(img,position_estimated[frame,:])
               
        
        #cv2.putText(img_save,'error:%f'%error.item(),(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.imwrite(loadfolder+'test_result/%d.png'%frame,img_save)
                
          
    
    
    
    #compare 
    position_error=(position_gt-position_estimated)**2 #(295510,63)
    position_error=np.reshape(position_error,(len(position_error),21,3)) #(295510,21,3)
    position_error=np.sqrt(np.sum(position_error,2))  #(295510,21)
 
    
    #percentage of frames with worst error < e
    
    pof_joints=[]
    position_error_max=np.max(position_error,1)
    for th in range(80):
        pof_joints.append(100*np.count_nonzero(position_error_max<th)/frame_num)
    
    plt.grid()
    plt.xlabel('error threshold' +'  \u03B5' + ' (mm)')
    plt.ylabel('proportion of frames with all joints error < '+'\u03B5')
    plt.plot(pof_joints)
                
    fig=plt.gcf()
    fig.savefig(loadfolder+'test result.png')  
    
    
    #percentage of joints within e
    '''
    pof_joints=[]
    for th in range(80):
        pof_joints.append(100*np.count_nonzero(position_error<th)/(21*frame_num))
    
    plt.grid()
    plt.xlabel('error threshold' +'  \u03B5' + ' (mm)')
    plt.ylabel('proportion of joints with error < '+'\u03B5')
    plt.plot(pof_joints)
                
    fig=plt.gcf()
    fig.savefig(loadfolder+'test result.png')  
    '''
    
        
    
       
    
            
            
            

        
        
        
        
        
        
        
        
        
        

                    