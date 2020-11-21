import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import pyrealsense2 as rs
import sys

from scipy import stats, ndimage

class Utils:
    def __init__(self,fx,fy,cx,cy,cube):

        self.fx=fx
        self.fy=fy
        self.cx=cx
        self.cy=cy
        self.calibMat=np.asarray([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]]) 
        self.cube=cube
        
    def project3D2imagePlane(self,pos):
        calibMat=self.calibMat
        p2d=np.matmul(calibMat,np.reshape(pos,3,1))
        p2d[0:2]=p2d[0:2]/p2d[2]
        return np.asarray(p2d,'int')#.astype(np.int32)
    
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
        
    
    
def makeLearningImage(img_crop,trainImageSize):
    s=trainImageSize
    img_crop[img_crop==0]=np.max(img_crop)
    d_max=np.max(img_crop)
    d_min=np.min(img_crop)
    if d_max-d_min<1:
        print('dmax-dmin<1 min%f max:%f'%(d_min,d_max))
    img_crop=-1+2*(img_crop-d_min)/(d_max-d_min)        
    #return img_seg,cv2.resize(img_crop,(s,s))
        
    cnnimg=cv2.resize(img_crop,(s,s))

    return np.copy(cnnimg)
    
def init_device():
    # Configure depth streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
    config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 90)
    
    config.enable_stream(rs.stream.accel,rs.format.motion_xyz32f,250)
    config.enable_stream(rs.stream.gyro,rs.format.motion_xyz32f,200)
    
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


if __name__=="__main__":
    
    opt='train'
    frame=8000
    data_num=10000
    
    '''
    opt='test'
    frame=1000
    data_num=2000
    '''
    
    ##--setting
    save_filepath='/home/yong/hdd/dataset/depth_ir/'+opt+'/'
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
    
    ##--init realsense--##
    pipeline, depth_scale = init_device()
    #imgs_train=np.zeros((trainImageSize,2*trainImageSize))
    imgs_train=np.zeros((trainImageSize,2*trainImageSize))
    
    try:
        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            depth,ir = read_frame_from_device(pipeline, depth_scale)
            depth=depth.astype(np.uint16)
            cv2.imshow('ir',ir)
            cv2.imshow('detth',np.uint8(depth))
            
            #save original image
            '''
            cv2.imwrite(save_filepath+'depth%d.png'%frame,depth)
            cv2.imwrite(save_filepath+'ir%d.png'%frame,ir)
            '''
            
            cv2.waitKey(1)
            frame+=1
            if frame==data_num:
                break
            
            if frame%100==0:
                print('frame..',frame)
            
    finally:
        print('stop device')
        stop_device(pipeline)
    
    