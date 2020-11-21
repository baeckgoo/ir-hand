import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
import cv2
import numpy as np
import pickle

import sys
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
from datasetLoader_uvr import Utils 
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from datasetloader.datasetLoader_uvr import DatasetLoader as datasetloader_UVR

from GrabCut import Grabcut

from multiprocessing import Process, Manager

def fct_grabcut(Mobj,idx,ir,depth_seg,com,window,d_maximum):
    ir_seg=Grabcut.segment_grabcut(ir,depth_seg,com,window,d_maximum)#0.6      

    Mobj[idx]=ir_seg
    
def segment_irhand(ir_orig):
    ir_norm=ir_orig #its type is already uint8.
    
    #ir_norm=np.uint8(cv2.normalize((ir_orig), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)) 
    
    #ir_norm3c=cv2.cvtColor(ir_norm,cv2.COLOR_GRAY2BGR)
    #ir_norm3c_rgb=cv2.applyColorMap(ir_norm3c,cv2.COLORMAP_JET)
    #cv2.imshow('ir_norm',ir_norm)
            
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
            
    return ir_norm,irhand_seg   

if __name__=="__main__":
    ##--user setting
    frame_start=0
    data_num=8000
    thread_num=10

    
    #--common setting
    dataset_version='v2' #'v1', 'v1_shuffle',  'v2'
    load_filepath_img='/home/yong/ssd/dataset/blurHand20/'+dataset_version+'/images/'
    save_filepath_img='/home/yong/ssd/dataset/preprocessed_blurHand20/'+dataset_version+'/' 
    savepath_dict="/home/yong/ssd/dataset/preprocessed_blurHand20/dataset_dict_"+dataset_version+".pickle"
    save_enable=True
    
    ##--common setting
    d_minimum=0
    d_maximum=500
    
    camera_info={}
    if dataset_version=='v1' or dataset_version=='v1_shuffle':
        camera_info['fx']=475.065948/2.0
        camera_info['fy']=475.065857/2.0
        camera_info['cx']=315.944855/2.0
        camera_info['cy']=245.287079/2.0
        camera_info['camerawidth']=320
        camera_info['cameraheight']=240    
    elif dataset_version=='v2':
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
    datasetloader_uvr=datasetloader_UVR('..',0,camera_info,'..','..')
    
    dataset_dict={}
    dataset_dict['com']={}
    dataset_dict['window']={}
    
    #--start
    progressbar=trange(0,data_num//thread_num,leave=True)  
    for g in progressbar:
        with Manager() as manager:
            Mobj=manager.list([0 for x in range(thread_num)])
            MpList=[]
            depth_train_bag=[]
            window_bag=[]
            com_bag=[]
            
            mask_bag=[]
            depth_seg_bag=[]
            
            
            for i in range(thread_num):
                frame=thread_num*g+i
                #input
                if dataset_version=='v1' or dataset_version=='v1_shuffle':
                    depth=cv2.imread(load_filepath_img+'depth-%07d.png'%frame,2)
                    ir=cv2.imread(load_filepath_img+'ir-%07d.png'%frame,2)
                elif dataset_version=='v2':
                    depth=cv2.imread(load_filepath_img+'depth%d.png'%frame,2)
                    ir=cv2.imread(load_filepath_img+'ir%d.png'%frame,2)
                
                #mask from depth
                mask=np.ones(depth.shape[:2])
                mask[depth>d_maximum]=0
                
                
                #segment depth 
                depth_seg=depth*mask 
                
                # preprocess (depth)
                depth_train,depth_crop,com,window=datasetloader_uvr.preprocess_depth(depth_seg,trainImageSize,cube)
                
            
                depth_train_bag.append(depth_train)
                window_bag.append(window)
                com_bag.append(com)
                mask_bag.append(mask)
                depth_seg_bag.append(depth_seg)
            
                MpList.append(Process(target=fct_grabcut,args=(Mobj,i,ir,depth,com,window,d_maximum)))
                
            #do work with threads (grabcut)
            for _ in MpList:
                _.start()
            for _ in MpList:
                _.join()
                
            for i in range(thread_num):
                frame=g*thread_num+i
                
                depth_train=depth_train_bag[i]
                window=window_bag[i]
                com=com_bag[i]
                
                ir_seg=Mobj[i]
                
                ir_train=datasetloader_uvr.preprocess_ir(ir_seg,window)#0.0002
            
                # save image
                imgs_train[:,0:trainImageSize]=depth_train
                imgs_train[:,trainImageSize:2*trainImageSize]=ir_train
                
                cv2.imshow("imgs_train",imgs_train)
                cv2.waitKey(1)    
                
                if save_enable==True:
                    np.save(save_filepath_img+'%d.npy'%frame,imgs_train)
                    dataset_dict['com'][frame]=com
                    dataset_dict['window'][frame]=window
            

    dataset_save={}
    dataset_save['com']=[]
    dataset_save['window']=[]
    for i in range(len(dataset_dict['com'])):
        dataset_save['com'].append(dataset_dict['com'][i])
        dataset_save['window'].append(dataset_dict['window'][i])
           
    with open(savepath_dict,'wb') as f:
        pickle.dump(dataset_save,f,pickle.HIGHEST_PROTOCOL)
    
    
        