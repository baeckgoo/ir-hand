import pyrealsense2 as rs
import os,sys,inspect
currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
basedir=currentdir+'\\..\\'
sys.path.append(basedir)

#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from HPE.model.deepPrior import dpnet
from HIG.model.network import Pix2pix
from datasetloader.datasetLoader_uvr import DatasetLoader as datasetloader_UVR
from GrabCut import Grabcut
from Sensor.Realsense import Realsense

#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
from datasetloader.datasetLoader_icvl import Utils 

#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/Utility/')
from Utility.visualize import Visualize

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


def get_state_dict(origin_dict): 
    old_keys = origin_dict.keys() 
    new_dict = {} 
    for ii in old_keys:
        temp_key = str(ii) 
        if temp_key[0:7] == "module.": 
            new_key = temp_key[7:] 
        else: 
            new_key = temp_key 
            
        new_dict[new_key] = origin_dict[temp_key] 
        
    return new_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch hand pose Training')
    args=parser.parse_args()
    
    #--user setting
    input_type= 'sr300'  #'dataset', 'sr300'
    dataset_name='test' #'train' 'test'  'test_blur' 
    show_enabled=True
  
    args.gpu_ids='1'
    data_num=1000
    
    args.use_net=['hig','hpe1']
    args.train_net=['']

    args.cgan=True

    #--load file path
    trained_modelFile_hpe=basedir+'../HIG/output/2020_3_31_22_4/trainedModel_epoch98.pth.tar'
    trained_modelFile_hig=basedir+'../HIG/output/2020_3_31_22_4/trainedModel_epoch98.pth.tar'
    load_filepath='/home/yong/ssd/dataset/depth_ir/'+dataset_name+'/'
    load_filepath_preprocess='/home/yong/ssd/dataset/preprocessed_HIG/'
    
    #--save file path
    savefolder_fingertip='/home/yong/hdd/fingertip_test/ablative/'+dataset_name+'/'
    save_fingertip=True
    #savefolder='/home/yong/hdd/HIG/output/2020_3_31_22_4/'+dataset_name+'/'
    savefolder='/home/yong/hdd/HIG/output/2020_3_31_22_4/synthesized/'
    
    #if not os.path.isdir(savefolder):
    #    os.mkdir(savefolder)

        
    #--camera setting
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
    
    
    #--common setting
    args.is_train=False
    args.gan_mode='vanilla'
    args.lambda_L1=50
    args.trainImageSize=128
    trainImageSize=128
    args.skeleton_pca_dim=52
    args.skeleton_orig_dim=63
    args.discriminator_reconstruction=True
    args.test_batch=1
     
    
    hpe_numBlocks=5
    args.hpe_enabled=True
    trainImageSize=128
    
    #device
    device = torch.device("cuda:%s"%args.gpu_ids if torch.cuda.is_available() else "cpu")
    args.device=device
    
    ##--select model (generator)--##
    #if not os.path.isdir(savefolder):
    #    os.mkdir(savefolder)
      
    if not 'pix2pix' in locals():
        pix2pix=Pix2pix(args) 
        checkpoint=torch.load(trained_modelFile_hig)
        if args.gpu_ids=='all':
            pix2pix.netG.load_state_dict(checkpoint['state_dict_hig'])
        else:
            checkpoint_dict=get_state_dict(checkpoint['state_dict_hig'])
            pix2pix.netG.load_state_dict(checkpoint_dict)
        pix2pix.netG.eval()
        
    
    ##--select model (HPE)--##
    if args.hpe_enabled==True:
        if not 'hpe' in locals():
            hpe=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
            device = torch.device("cuda:%s"%args.gpu_ids if torch.cuda.is_available() else "cpu")
            hpe=hpe.to(device)
            
            cudnn.benchmark = True     
            checkpoint=torch.load(trained_modelFile_hpe)
            hpe.load_state_dict(checkpoint['state_dict_hpe1'])
            hpe.eval()
            img_hpe = np.zeros((1,1,trainImageSize,trainImageSize), dtype = np.float32)
        
        with open(basedir+'..\\pca_52_by_957032_augmented_X2.pickle','rb') as f:
            pca=pickle.load(f)
    
    ##--init realsense--##
    if input_type=='sr300':
        realsense=Realsense()
    
    ##--dataset--##
    jointnum=21
    
    datasetloader_uvr={}
    datasetloader_uvr['train']=datasetloader_UVR(load_filepath_preprocess,data_num,camera_info,'train','generator')
    datasetloader_uvr['train'].set_datasetLoader(args.test_batch)
    datasetloader_uvr['train_blur']=datasetloader_UVR(load_filepath_preprocess,data_num,camera_info,'train_blur','generator')
    datasetloader_uvr['train_blur'].set_datasetLoader(args.test_batch)
    datasetloader_uvr['test']=datasetloader_UVR(load_filepath_preprocess,data_num,camera_info,'test','generator')
    datasetloader_uvr['test'].set_datasetLoader(args.test_batch)
    datasetloader_uvr['test_blur']=datasetloader_UVR(load_filepath_preprocess,data_num,camera_info,'test_blur','generator')
    datasetloader_uvr['test_blur'].set_datasetLoader(args.test_batch)
    
    utils=datasetloader_uvr[dataset_name].utils
    
    d_maximum=500
    
    if input_type=='dataset':
        loadpath_dict="/home/yong/ssd/dataset/preprocessed_HIG/dataset_dict_"+dataset_name+".pickle"
        with open(loadpath_dict,'rb') as f:
            dataset_load=pickle.load(f)
                

    #start
    frame=0
    depth_error_bag=[]
    joint_error_bag=[]    
    vis=Visualize(utils)
    fingertip={}
    fingertip['hpe1']=[]
    fingertip['hig_hpe1']=[]
    
    try:
        progressbar=trange(data_num,leave=True)    
        for i in progressbar:
        #for i in range(1000):
            frame=i
            ##--input
            if input_type=='sr300':
                realsense.run()
                depth_orig=realsense.getDepthImage()
                ir_orig=realsense.getIrImage()
                rgb_orig=realsense.getColorImage() 
            elif input_type=='dataset':
                depth_orig=cv2.imread(load_filepath+'depth%d.png'%frame,2)
                ir_orig=cv2.imread(load_filepath+'ir%d.png'%frame,2)
         
            #--preprocess depth/ir
            if input_type=='sr300':
                depth_seg=depth_orig.copy()
                depth_seg[depth_seg>d_maximum]=0
                
                if not np.any(depth_seg):
                    continue
            
                depth_train,depth_crop,com,window=datasetloader_uvr['test'].preprocess_depth(depth_seg,trainImageSize,cube)
            
                ir_seg=segment_irhand_contour(np.uint8(ir_orig))
                ir_train=datasetloader_uvr['test'].preprocess_ir(ir_seg,window) 
                 
            elif input_type=='dataset':
                depth_train=datasetloader_uvr[dataset_name].loadPreprocessedDepthImage(frame,dataset_name)
                ir_train=datasetloader_uvr[dataset_name].loadPreprocessedIrImage(frame,dataset_name)
        
                com=dataset_load['com'][frame]
                window=dataset_load['window'][frame]
                    
                depth_crop=datasetloader_uvr[dataset_name].utils.crop(depth_orig,window[0],window[1],window[2],window[3],window[4],window[5])
                 
                
                
            ##--network
            depth_pred=pix2pix.forward_test_phrase(ir_train,trainImageSize)
        
            ##--store joints
            output_recon_gt_pred=[]
            img_hpe[0,0,:,:]=depth_train.copy()
            com3d=utils.unproject2Dto3D(com)  
            output_recon_hpe1=hpe.forward_with_reconstruction(img_hpe,pca,com3d,cube)
            output_recon_gt_pred.append(output_recon_hpe1)
            
            img_hpe[0,0,:,:]=depth_pred.copy()
            com3d=utils.unproject2Dto3D(com)  
            output_recon_hig_hpe1=hpe.forward_with_reconstruction(img_hpe,pca,com3d,cube)
            output_recon_gt_pred.append(output_recon_hig_hpe1)
            
            fingertip['hpe1'].append(output_recon_hpe1[0,17*3:18*3])
            fingertip['hig_hpe1'].append(output_recon_hig_hpe1[0,17*3:18*3])
        
            if show_enabled==True:
                vis.set_depth_train(depth_train)
                vis.set_depth_pred(depth_pred)
                
                vis.convertTo3cImage(ir_train,trainImageSize)
                #depth_error=vis.calculateDepthError(depth_crop,cube,com)
                
                #depth error
                d_recon_gt=depth_train*(cube[2]/2)+com[2]
                d_recon_pred=depth_pred*(cube[2]/2)+com[2]
                d_recon_error=np.sqrt((d_recon_gt-d_recon_pred)*(d_recon_gt-d_recon_pred))
                
                #d_error=np.sum(d_recon_error)/(128*128)
                depth_error_bag.append(d_recon_error)
                #print(np.mean(d_recon_error),np.median(d_recon_error))
                #
                
            
            
                vis.set_img_save_leftwindow(depth_orig,output_recon_hpe1)
                vis.set_img_save_rightwindow(depth_orig,output_recon_hig_hpe1)
                
                err_joints=abs(output_recon_gt_pred[0]-output_recon_gt_pred[1])
                #vis.putText(depth_error,err_joints)

                img_save=vis.get_img_save()
                cv2.imshow('result',img_save)
                cv2.waitKey(1)
            
                #write
                depth_save=np.zeros((128*4,128,3))
                #original ir
                depth_save[0:128,:,0]=255*(ir_train-np.min(ir_train))/(np.max(ir_train)-np.min(ir_train))
                depth_save[0:128,:,1]=depth_save[0:128,:,0]
                depth_save[0:128,:,2]=depth_save[0:128,:,0]
                #original depth
                depth_save[128:256,:,0]=255*(depth_train-np.min(depth_train))/(np.max(depth_train)-np.min(depth_train))
                depth_save[128:256,:,1]=depth_save[128:256,:,0]
                depth_save[128:256,:,2]=depth_save[128:256,:,0]
                #predicted depth
                depth_save[256:384,:,0]=255*(depth_pred-np.min(depth_pred))/(np.max(depth_pred)-np.min(depth_pred))
                depth_save[256:384,:,1]=depth_save[256:384,:,0]
                depth_save[256:384,:,2]=depth_save[256:384,:,0]
                #difference of depth
                depth_dif=np.abs(depth_save[128:256,:]-depth_save[256:384,:])
                depth_dif=np.asarray(depth_dif,np.uint8)
                #depth_save[128*2:128*3,:,:]=cv2.applyColorMap(cv2.convertScaleAbs(depth_dif, alpha=0.03), cv2.COLORMAP_JET)
                depth_save[384:512,:,:]=cv2.applyColorMap(depth_dif, cv2.COLORMAP_JET)
                
                #depth_save[128*2:128*3,:]=depth_dif
                
                cv2.imwrite(savefolder+'%d.png'%frame,depth_save)  
               
    finally:
        if input_type=='sr300':
            print('stop device')
            realsense.release()
    
    
    #save fingertip position
    if save_fingertip==True:     
        saveFingertip(savefolder_fingertip+'hpe1.txt',fingertip['hpe1'])
        saveFingertip(savefolder_fingertip+'hig_hpe1.txt',fingertip['hig_hpe1'])
    
    
   
      
    
    #error
    '''
    print('avg depth_error',np.mean(depth_error_bag))
    print('std depth_error',np.std(depth_error_bag))
    print('avg joint_error',np.mean(joint_error_bag))
    print('std joint_error',np.std(joint_error_bag))
    
    f=open(savefolder+'../test_error.txt','w')
    f.write('average of depth error:  %s\n'%np.mean(depth_error_bag))
    f.write('std of depth error:      %s\n'%np.std(depth_error_bag))
    f.write('average of joint error:  %s\n'%np.mean(joint_error_bag))
    f.write('std of joint error:      %s\n'%np.std(joint_error_bag))
    f.close()
    
    #plot
    percentage_of_frame_bag=[]
    for th in range(100):
        percentage_of_frame=100*np.count_nonzero(np.asarray(depth_error_bag)<th)/len(depth_error_bag)
        percentage_of_frame_bag.append(percentage_of_frame)
    plt.grid()
    plt.xlabel('D:Max allowed distance to GT [mm]')
    plt.ylabel('percentage of frames with average error within D')
    plt.plot(percentage_of_frame_bag)
    fig=plt.gcf()
    fig.savefig(savefolder+'../'+'test result.png')
    
    '''
        