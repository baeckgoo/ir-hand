import pyrealsense2 as rs

import os,sys,inspect
currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
basedir=currentdir+'\\..\\'
sys.path.append(basedir)

#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from datasetloader.datasetLoader_uvr import DatasetLoader as datasetloader_UVR

#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
from datasetloader.datasetLoader_icvl import Utils 


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
from tqdm import trange
import csv
import time

def init_device():
    # Configure depth streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
    print('config')
    # Start streaming
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print('Depth Scale is:' , depth_scale)
    return pipeline, depth_scale
   
def stop_device(pipeline):
    pipeline.stop()
    
def read_frame_from_device(pipeline, depth_scale):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    ir_frame =  frames.get_infrared_frame()
    #if not depth_frame:
    #    return None
    # Convert images to numpy arrays
    depth_image = np.asarray(depth_frame.get_data(), dtype=np.float32)
    depth = depth_image * depth_scale * 1000
        
    ir= np.asarray(ir_frame.get_data(), dtype=np.uint8)
        
    return depth,ir

#
def segment_with_blurband():
    a=0

#for vr20
def segment_grabcut_VR20_dataset(ir_orig,depth_seg,com,window):
    ir_orig3c=cv2.cvtColor(np.uint8(ir_orig),cv2.COLOR_GRAY2BGR)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    #(1)--first grabcut
    
    mask=np.zeros(ir_orig.shape[:2],np.uint8)
    rect=(window[0],window[2],window[1]-window[0],window[3]-window[2])
    cv2.grabCut(ir_orig3c,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    ir_seg2 = ir_orig*mask2[:,:]

    #(2)--second grabcut
    ir_seg2=cv2.cvtColor(np.uint8(ir_seg2),cv2.COLOR_GRAY2BGR)
    
    #option
    #newmask=np.where(depth_seg>0,3,2).astype('uint8') #our dataset
    newmask=np.ones(ir_orig.shape[:2],np.uint8)
    
    newmask[0:window[2],:]=0
    newmask[:,0:window[0]]=0
    newmask[window[3]:240,:]=0
    newmask[:,window[1]:320]=0
            
    #newmask[int(com[1])-10:int(com[1])+10,int(com[0])-10:int(com[0])+10]=1
    
    
    mask[newmask==0]=0
    #mask[newmask==1]=1
    
    #run
    newmask2,_,_=cv2.grabCut(ir_seg2,mask,None,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_MASK)
            
    #out
    mask_out2 = np.where((newmask2==2)|(newmask2==0),0,1).astype('uint8')
    #mask_out2 = np.where((newmask2==2),0,1).astype('uint8')
    
    ir_seg2 = ir_orig*mask_out2[:,:]
    
    
    return ir_seg2     

def segment_grabcut_blueband(ir_orig,depth,com,window,d_max,band_mask):
    ir_orig3c=cv2.cvtColor(np.uint8(ir_orig),cv2.COLOR_GRAY2BGR)
    
    #option
    mask=np.where(depth>0,3,2).astype('uint8')
    
    #mask[band_mask==255]=2
    
    
    mask[0:window[2],:]=0
    mask[:,0:window[0]]=0
    mask[window[3]:480,:]=0
    mask[:,window[1]:640]=0
    
    mask[depth>d_max]=0
    
    
    
    #mask[int(com[1])-10:int(com[1])+10,int(com[0])-10:int(com[0])+10]=1
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    #run
    mask_out1,_,_=cv2.grabCut(ir_orig3c,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
            
    #out
    mask_out2 = np.where((mask_out1==2)|(mask_out1==0),0,1).astype('uint8')
    ir_seg2 = ir_orig*mask_out2[:,:]
    
    return ir_seg2,mask  

#0: background , 1: foreground, 2:probable background, 3:probable foreground    
def segment_grabcut(ir_orig,depth,com,window,d_max):
    ir_orig3c=cv2.cvtColor(np.uint8(ir_orig),cv2.COLOR_GRAY2BGR)
    
    #option
    mask=np.where(depth>0,3,2).astype('uint8')
    
    
    mask[0:window[2],:]=0
    mask[:,0:window[0]]=0
    mask[window[3]:480,:]=0
    mask[:,window[1]:640]=0
    
    mask[depth>d_max]=0
    
    #mask[int(com[1])-10:int(com[1])+10,int(com[0])-10:int(com[0])+10]=1
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    #run
    mask_out1,_,_=cv2.grabCut(ir_orig3c,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
            
    #out
    mask_out2 = np.where((mask_out1==2)|(mask_out1==0),0,1).astype('uint8')
    ir_seg2 = ir_orig*mask_out2[:,:]
    
    return ir_seg2  

    '''
    ir_orig3c=cv2.cvtColor(np.uint8(ir_orig),cv2.COLOR_GRAY2BGR)
    
    #option
    mask=np.where(depth_seg>0,3,2).astype('uint8')
    
    mask[0:window[2],:]=0
    mask[:,0:window[0]]=0
    mask[window[3]:480,:]=0
    mask[:,window[1]:640]=0
            
    mask[int(com[1])-10:int(com[1])+10,int(com[0])-10:int(com[0])+10]=1
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    #run
    mask_out1,_,_=cv2.grabCut(ir_orig3c,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
            
    #out
    mask_out2 = np.where((mask_out1==2)|(mask_out1==0),0,1).astype('uint8')
    ir_seg2 = ir_orig*mask_out2[:,:]
    
    return ir_seg2     
    '''

def segment_irhand_contour(ir_orig):
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
    

if __name__ == '__main__':
    
    #--to test
    dataset_name='test_blur'   #test, test_blur
    
    #--user setting
    input_type='dataset_uvr' # 'sr300' 'dataset_uvr'
    imgfile_path_orig="/home/yong/ssd/dataset/depth_ir/"+dataset_name+"/"  
    

    #--common setting   
    ointNum=21
    fx=475.065948
    fy=475.065857
    cx=315.944855
    cy=245.287079   
    cube=np.asarray([250,250,250])
    
    utils=Utils(21,fx,fy,cx,cy,cube) 

    
    #--dataset of uvr
    datasetloader_uvr=datasetloader_UVR(imgfile_path_orig,0,0)
    #imgfile_path_orig="/home/yong/Downloads/saveData/uvrDataset - ablation/images/"
    
    
    ##--init realsense--##
    if input_type=='sr300':
        pipeline, depth_scale = init_device()
        
    
    #--start        
    try:
        for frame in range(690,1000):
        #while(1):
            #capture
            if input_type=='sr300':
                depth_orig,ir_orig = read_frame_from_device(pipeline, depth_scale)
            else:
                depth_orig=datasetloader_uvr.loadDepthImage(frame)
                ir_orig=datasetloader_uvr.loadIrImage(frame)
                
            depth_seg=depth_orig.copy()
            depth_seg[depth_seg>500]=0
            if depth_seg.max()<200:  
                continue
            cv2.imshow('depth_seg',np.uint8(depth_seg))
                        
            # preprocess (depth)
            depth_train,depth_crop,com,window=datasetloader_uvr.preprocess_depth(depth_seg,128,cube)
            print('window',window)
            # preprocess (ir)
            ir_train=datasetloader_uvr.preprocess_ir(ir_orig,window)
            
            
            ####### segment ir with grabcut  
            #ir_seg2=segment_grabcut(ir_orig,depth_seg,com,window)
            
            ir_orig3c=cv2.cvtColor(np.uint8(ir_orig),cv2.COLOR_GRAY2BGR)
            #option
            mask=np.where(depth_seg>0,3,2).astype('uint8')
            mask[0:window[2],:]=0
            mask[:,0:window[0]]=0
            mask[window[3]:480,:]=0
            mask[:,window[1]:640]=0
            
            mask[int(com[1])-10:int(com[1])+10,int(com[0])-10:int(com[0])+10]=3
    
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
    
            #run
            mask_out1,_,_=cv2.grabCut(ir_orig3c,mask,None,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_MASK)
            
            #out
            mask_out2 = np.where((mask_out1==2)|(mask_out1==0),0,1).astype('uint8')
            ir_seg2 = ir_orig*mask_out2[:,:]
    

            ####### show and save
            #plt.imsave('ir_seg_2.png',ir_seg2)
            plt.imsave('depth_seg.png',depth_seg)
            plt.imsave('ir.png',ir_orig)
            plt.imsave('ir_seg.png',ir_seg2)
    
            cv2.imshow('seg',np.uint8(ir_seg2))
            plt.imshow(ir_seg2)
            
            cv2.waitKey(1)
            break
            
            
            
            #break
            
    finally:
        if input_type=='sr300':
            print('stop device')
            stop_device(pipeline)
    
   
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    