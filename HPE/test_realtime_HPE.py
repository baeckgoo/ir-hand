import os,sys,inspect
currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
basedir=currentdir+'\\..\\'
sys.path.append(basedir)

weightdir=basedir+'..\\HPE\\output\\'

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

import pyrealsense2 as rs
import torch.backends.cudnn as cudnn
import torch

import pickle


#from datasetloader.datasetLoader_icvl import Utils
#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
from datasetloader.datasetLoader_icvl import Utils
from model.deepPrior import dpnet

# select proper device to run
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33


def init_device():
    # Configure depth streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
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
    #if not depth_frame:
    #    return None
    # Convert images to numpy arrays
    depth_image = np.asarray(depth_frame.get_data(), dtype=np.float32)
    depth = depth_image * depth_scale * 1000
    return depth


if __name__=="__main__":
    ##--setting--##
    #if not 'datasetloader' in locals():
    #    datasetloader=DatasetLoader("../preprocessed/")
    jointNum=21
    fx=475.065948
    fy=475.065857
    cx=315.944855
    cy=245.287079   
    cube=np.asarray([250,250,250])
    trainImageSize=128#datasetloader.trainImageSize
    outDim=52
    numBlocks=5
   
    utils=Utils(jointNum,fx,fy,cx,cy,cube) 
    
    with open(basedir+'..\\pca_52_by_957032_augmented_X2.pickle','rb') as f:
        pca=pickle.load(f)

                        
    ##--select model--##
    trained_modelFile=basedir+'..\\HPE\\output\\2020_2_26_11_29\\trainedModel_epoch99.pth.tar'

        
    if not 'model' in locals():
        model=dpnet(outDim,numBlocks,device)
        model=model.to(device)        
        checkpoint=torch.load(trained_modelFile)
        model.load_state_dict(checkpoint['state_dict'])
        #model.load_state_dict(torch.load(trained_modelFile))
        model.eval()
    

    ##--init realsense--##
    pipeline, depth_scale = init_device()

    ##--run--##
    img_test = np.zeros((1,1,trainImageSize,trainImageSize), dtype = np.float32)
    joint_test=np.zeros((1,jointNum))
    print('start..')
    try:
        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            depth = read_frame_from_device(pipeline, depth_scale)
            depth_seg=depth.copy()
            
            # preprocess
            depth_seg[depth_seg>500]=0
            if depth_seg.max()<200:  
                continue
            #print(depth_seg.max())
            com=utils.calculateCOM(depth_seg)
            com,img_crop,window=utils.refineCOMIterative(depth_seg,com,3)
            
            
            cv2.imshow('cnnimg',np.uint8(img_crop))
            img_crop=utils.makeLearningImage(img_crop,com)
            img_test[0,0,:,:]=img_crop.copy()
            
            
            # predict
            input=torch.FloatTensor(img_test)
            input=input.to(device)
            #TIME=time.time()
            output_embed=model(input)
            #print('time..:',time.time()-TIME)
            
            #reconstruct
            pca_w=torch.FloatTensor(pca.components_)
            pca_w=pca_w.to(device)
            pca_b=torch.FloatTensor(pca.mean_)
            pca_b=pca_b.to(device)
            output_recon=torch.mm(output_embed,pca_w)+pca_b
            output_recon=output_recon.detach().cpu().numpy()
            
            #denormalize
            com3d=utils.unproject2Dto3D(com)   
            cube=utils.cube
            output_recon=output_recon*(cube[2]/2.)+np.tile(com3d,(1,21))
                 
            # show
            img_show=np.zeros((480,640,3))
            img_show=np.uint8(cv2.normalize(depth_seg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))           
            img_show=cv2.cvtColor(img_show,cv2.COLOR_GRAY2BGR)
            
            img_show=utils.circle3DJoints2Image(img_show,output_recon[0,:])
            cv2.imshow("result",img_show)
            cv2.waitKey(1)
            
        
    finally:
        print('stop device')
        stop_device(pipeline)
    
    