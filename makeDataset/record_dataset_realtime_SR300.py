import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import pyrealsense2 as rs
import sys

from scipy import stats, ndimage
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from Sensor.Realsense import Realsense


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
    opt='recording999'
    frame_start=-10
    frame_end=5000
    save_enabled=True
    marker_enabled=False
    
    
    ##--common setting
    save_filepath='/home/yong/ssd/dataset/depth_ir/'+opt+'/'
    trainImageSize=128
    d_minimum=200
    d_maximum=500
    
    fx=475.065948
    fy=475.065857
    cx=315.944855
    cy=245.287079
    cube=np.asarray([250,250,250])
    
    savefolder="/home/yong/ssd/dataset/depth_ir/"+opt
    
    ##--init realsense--##
    realsense=Realsense()
    
    imgs_train=np.zeros((trainImageSize,2*trainImageSize))
    
    try:
        #while True:
        for frame in range(frame_start,frame_end):
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            realsense.run()
            ir=realsense.getIrImage()
            depth=realsense.getDepthImage()            
            color=realsense.getColorImage()
            
            depth=depth.astype(np.uint16)
            
            #cv2.imshow('ir',ir)
            #cv2.imshow('depth',np.uint8(depth))
            
            #save original image
            if save_enabled==True and frame>-1:
                print(frame)
                cv2.imwrite(save_filepath+'depth%d.png'%frame,depth)
                cv2.imwrite(save_filepath+'ir%d.png'%frame,ir)
                cv2.imwrite(save_filepath+'color%d.png'%frame,color)
            
            depth_seg=np.copy(depth)
            depth_seg[depth_seg>d_maximum]=0
            cv2.imshow('depth_seg',np.uint8(depth_seg))
            cv2.imshow('color',color)
            
            #detect marker positiopn
            if marker_enabled==True:
                detectMarker(ir)
            
            cv2.waitKey(10)
            if frame==frame_end:
                break
            
            if frame%100==0:
                print('frame..',frame)
            
    finally:
        print('stop device')
        realsense.release()    
        
    print(frame)
        
    
    