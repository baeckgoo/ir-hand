import os,sys,inspect
currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
basedir=currentdir+'\\..\\'
sys.path.append(basedir)

from HPE.model.deepPrior import dpnet
from HIG.model.network import Pix2pix
from Model.fusionNet import FusionNet
from Segmentation.Segmentation import Segmentation
from datasetloader.datasetLoader_uvr import DatasetLoader as datasetloader_UVR
from Utility.visualize import Visualize_combined_outputs
from datasetloader.datasetLoader_icvl import DatasetLoader as datasetloader_ICVL

import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import pickle
import os
import argparse
import csv
from pyquaternion import Quaternion
from Udp import UDP
from Sensor.Realsense import Realsense

def calculateRotation(a,b):
    avg_a=np.mean(a,0)
    avg_b=np.mean(b,0)
    
    A=np.mat(a-avg_a)
    B=np.mat(b-avg_b)
    
    H=np.transpose(A)*B
    
    U,S,Vt=np.linalg.svd(H)
    R=Vt.T*U.T
    #print('R',R)
    
    if np.linalg.det(R) <0:
        print('reflection detected')
        Vt[2,:]*=-1
        R=Vt.T*U.T
        
    return Quaternion(matrix=R)
    
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
    
def saveFingertip(filename,position):
    f = open(filename, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for fr in range(len(position)):
        wr.writerow(position[fr])
    f.close() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch hand pose estimation')
    args=parser.parse_args()
    
    #-user setting
    args.gpu_ids='0'
    args.use_net=['hpe1_orig','hig','hpe1','hpe2'] #hpe1_orig, hig, hpe1, hpe2
    forward_list=['hpe1_orig','hig_hpe1','hpe2'] #hpe1_orig, hig_hpe1, hpe2

    input_type= 'sr300'  #'sr300' , 'dataset'
    frame=88# 100 #100

    segment_method='barehand'  #'blueband' 'barehand'
    run_method='ir_hpe'  #'depth_hpe' 'ir_hpe'
    
    #-to test
    trained_modelFile_hpe1_orig=basedir+'..\\HPE\\output\\2020_2_26_11_29\\trainedModel_epoch99.pth.tar'
    trained_modelFile_hpe1=basedir+'..\\HIG\\output\\2020_3_31_22_4\\trainedModel_epoch98.pth.tar'
    trained_modelFile_hig=basedir+'..\\HIG\\output\\2020_3_31_22_4\\trainedModel_epoch98.pth.tar'
    trained_modelFile_hpe2=basedir+'..\\HPE2\\output\\2020_4_10_18_46\\trainedModel_epoch99.pth.tar'
    
    
    
    #--common setting
    #dataset_name='v1' #'v1', 'v2' 
    out_list={}
    args.train_net=[]
    args.cgan=True
 
    #--load file path
    #pcafile_name='pca_52_by_957032_augmented_X2.pickle'
    load_filepath_imgs='/home/yong/ssd/dataset/depth_ir/recording999/'
 
    #--save file path
    '''
    if dataset_name!=None:
        savefolder='/home/yong/hdd/realtime_demo_result'
        if not os.path.isdir(savefolder):
            os.mkdir(savefolder)
    '''
        
    #--common setting
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
    device = torch.device("cuda:%s"%args.gpu_ids if torch.cuda.is_available() else "cpu")
    args.device=device
    cudnn.benchmark = True  
    
    
    #--HIG net setting
    if 'hig' in args.use_net:
        if not 'pix2pix' in locals():
            pix2pix=Pix2pix(args) 
            checkpoint=torch.load(trained_modelFile_hig,map_location=lambda storage, loc: storage)

            if 'hig' in args.use_net:
                pix2pix.netG.load_state_dict(checkpoint['state_dict_hig'])

            if 'hid' in args.use_net:
                pix2pix.netD.load_state_dict(checkpoint['state_dict_hid'])      
         
    #--HPE net setting 
    if 'hpe1' in args.use_net:
        if not 'hpe1' in locals():
            hpe1=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
            checkpoint=torch.load(trained_modelFile_hpe1,map_location=lambda storage, loc: storage)
            hpe1.load_state_dict(checkpoint['state_dict_hpe1'])
    
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
        pca=pickle.load(f)
    
    
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
    
     
    if 'hpd' in args.use_net:       
        w=np.transpose(pca.components_)
        b=pca.mean_
        fusionnet.set_reconstruction_net(w,b)
    fusionnet=fusionnet.to(device) 
    
    ##--init realsense--##
    if input_type=='sr300':
        realsense=Realsense()
        #pipeline, depth_scale = init_device()
        
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
    
    trainImageSize=camera_info['trainImageSize']
    cube=camera_info['cube']
    camerawidth=camera_info['camerawidth']
    cameraheight=camera_info['cameraheight']
    
    jointnum=21
    d_maximum=500
    
    datasetloader_uvr=datasetloader_UVR('..',0,camera_info,'..','..')
    utils=datasetloader_uvr.utils
     
    '''
    if input_type=='dataset':
        loadpath_dict="/home/yong/ssd/dataset/preprocessed_blurHand20/dataset_dict_"+dataset_name+".pickle"
        with open(loadpath_dict,'rb') as f:
            dataset_load=pickle.load(f)   
    '''
    
    #--dataset of bighand
    if not 'datasetloader_icvl' in locals():
        datasetloader_icvl=datasetloader_ICVL(args.skeleton_pca_dim)
        gt=datasetloader_icvl.loadFullLabels("D:\\research\\handdataset\\HANDS17\\training\\Training_Annotation.txt")
        joint3d_bighand_train=np.asarray(gt,'float')
    
    #segmentation
    lower_hsv=np.array([94,111,37])
    upper_hsv=np.array([120,255,255])
    ir_range=[40,120]
    segmentation=Segmentation(d_maximum,lower_hsv,upper_hsv,ir_range)      
      
    #udp
    portnum=2325
    if not 'udp' in locals():
        udp=UDP(portnum)
        udp_out=np.zeros((1,63+4),'float32')
        
            
    #--start
    fusionnet.set_mode('eval')
    ir_batch = np.zeros((1,1,args.trainImageSize,args.trainImageSize), dtype = np.float32)
    depth_batch = np.zeros((1,1,args.trainImageSize,args.trainImageSize), dtype = np.float32)
    
    vis=Visualize_combined_outputs(utils,len(forward_list),1,camerawidth,cameraheight)

    
    #--thread
    '''
    dp_s={}
    dp_s['ir'],dp_s['depth_seg'],dp_s['com'],dp_s['window']=Queue(),Queue(),Queue(),Queue()
    irSeg_s=Queue()    
    em_s=Queue()
    
    EM_thread = Process(target=EM_fct, args=(dp_s,irSeg_s,em_s))    
    EM_thread.start()
    EM_update_frame=0
    '''
    initial_frame=True
    
    blur_key=False
    
    try:
        while(1):    
        #for fr in range(1):
            ########## input (0.0007 SEC) ##############
            if input_type=='sr300':
                #depth_orig,ir_orig = read_frame_from_device(pipeline, depth_scale)
                realsense.run()
                depth_orig=realsense.getDepthImage()
                ir_orig=realsense.getIrImage()
                rgb_orig=realsense.getColorImage()   
            elif input_type=='dataset':
                depth_orig=cv2.imread(load_filepath_imgs+'depth%d.png'%frame,2)
                ir_orig=cv2.imread(load_filepath_imgs+'ir%d.png'%frame,2)
                rgb_orig=cv2.imread(load_filepath_imgs+'color%d.png'%frame)
            
            cv2.imshow("color",rgb_orig)
            #ir_orig=np.uint8(ir_orig)
            
            
            #segment depth
            if segment_method=='blueband':
                segmentation.setImages(rgb_orig,depth_orig,ir_orig)
                if segmentation.makeMask_band()==None:
                    continue
                depth_seg=segmentation.segmentDepth()
            else:
                depth_seg=depth_orig.copy()
                depth_seg[depth_seg>d_maximum]=0
                    
            if not np.any(depth_seg):
                continue
             
            #preprocess depth image (0.01 SEC)
            depth_train,depth_crop,com,window=datasetloader_uvr.preprocess_depth(depth_seg,trainImageSize,cube)
            #if not np.any(com):
            #    continue
            
            #segment ir
            if segment_method=='blueband':
                ir_seg=segmentation.segmentIR(window)
            else:
                ir_seg=segment_irhand_contour(np.uint8(ir_orig))
        
        
            #preprocess IR image (0.003 SEC) 
            ir_train=datasetloader_uvr.preprocess_ir(ir_seg,window) 

            #check if preprocessed image is appropriate        
            if np.isnan(ir_train[0][0])==True or np.isnan(depth_train[0][0])==True:
                print('preprocessed image is not appropriate')
                continue
            
                 
            ########## set input (0.0002 SEC) ##############
            ir_batch[0,0,:,:]=ir_train.copy()
            depth_batch[0,0,:,:]=depth_train.copy()       
            fusionnet.set_input(ir_batch,depth_batch)  
            com3d=utils.unproject2Dto3D(com)  

            ########## Forward through HPE2 (0.01 sec) ##############
            if 'hpe2' in forward_list:
                fusionnet.forward_('hpe2')  # 0.01 sec 
                out_list['hpe2']=fusionnet.reconstruct_joints(pca,com3d,cube,'hpe2','tocpu')   #0.0003 sec

            ########## Forward through HIG-HPE1 (0.01 sec)  #################
            if 'hig_hpe1' in forward_list:
                fusionnet.forward_('hig_hpe1')
                out_list['hig_hpe1']=fusionnet.reconstruct_joints(pca,com3d,cube,'hig_hpe1','tocpu')
            
            ########## Forward through HPE1  #################
            if 'hpe1_orig' in forward_list:
                fusionnet.forward_('hpe1_orig')  # 0.01 sec 
                out_list['hpe1_orig']=fusionnet.reconstruct_joints(pca,com3d,cube,'hpe1_orig','tocpu')   #0.0003 sec
            
            #average
            #out_average=(out1+out2)/2
            #print(time.time()-TIME)
            for i,n in enumerate(forward_list):
                vis.setWindow(n,[0,i])
                if n=='hpe1_orig':
                    vis.set_inputimg(n,depth_train)
                    vis.set_outputimg(n,depth_orig)
                else:
                    vis.set_inputimg(n,ir_train)
                    #ir_seg[ir_seg==0]=255
                    vis.set_outputimg(n,ir_orig)
                    
                vis.convertTo3cInputImages(n)
                
                
                vis.set_finalimg(n,out_list[n])
                           
            img_save=vis.get_img_save()
            cv2.imshow('result',cv2.resize(img_save,(camerawidth*len(forward_list),cameraheight)))
            cv2.waitKey(1)
            
            
            ########## udp  #################
            #calculate rotation (quaternion)
            points_palm_ref=np.reshape(joint3d_bighand_train[0],(21,3))[[0,1,2,3,4,5],:]
            
            if run_method=='depth_hpe':
                points_palm_now=np.reshape(out_list['hpe1_orig'],(21,3))[[0,1,2,3,4,5],:]
            else:
                points_palm_now=np.reshape(out_list['hpe2'],(21,3))[[0,1,2,3,4,5],:]
            
            quaternion_delta=calculateRotation(points_palm_ref,points_palm_now)
            #print('quaternion:',quaternion_delta)
            
            if run_method=='depth_hpe':
                udp_out[0,0:63]=out_list['hpe1_orig']
            else:
                udp_out[0,0:63]=out_list['hpe2']
            
            udp_out[0,63]=quaternion_delta[0]
            udp_out[0,64]=quaternion_delta[1]
            udp_out[0,65]=quaternion_delta[2]
            udp_out[0,66]=quaternion_delta[3]
            
            udp.send(udp_out)
            
            frame+=1
                             
    finally:
        if input_type=='sr300':
            print('stop device')
            realsense.release()
            #stop_device(pipeline)
            
            