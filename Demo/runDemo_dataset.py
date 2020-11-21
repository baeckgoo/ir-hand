import sys

sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from HPE.src.model.deepPrior import dpnet
from HIG.src.model.network import Pix2pix
from Model.fusionNet import FusionNet

from datasetloader.datasetLoader_uvr import DatasetLoader as datasetloader_UVR
from GrabCut import Grabcut

sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/Utility/')
from visualize import Visualize_combined_outputs

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

from Udp import UDP
    
def calculateDepthError(depth_pred,depth_train,d_min,d_max):
    #error between depth images (original/predicted by pix2pix)
    d_recon_pred=d_min+(d_max-d_min)*(depth_pred+1)/2
    d_recon_gt=d_min+(d_max-d_min)*(depth_train+1)/2
                
    mask=d_recon_gt!=np.max(d_recon_gt)
    dif_abs=abs(d_recon_gt[mask]-d_recon_pred[mask])
    #dif_abs=abs(d_recon_gt-d_recon_pred)
    depth_error=np.sum(dif_abs)/np.count_nonzero(mask)
            
    return depth_error

def saveFingertip(filename,position):
    f = open(filename, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for fr in range(len(position)):
        wr.writerow(position[fr])
    f.close() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch hand pose Training')
    args=parser.parse_args()

    #--user setting
    #trained_modelFile_hpe1_orig='/home/yong/hdd/HPE/output/2020_2_26_11_29/trainedModel_epoch99.pth.tar'
    trained_modelFile_hpe1_orig='/home/yong/hdd/HPE_refined/output/2020_4_20_13_25/trainedModel_epoch99.pth.tar'
    
    trained_modelFile_hpe1='/home/yong/hdd/HIG/output/2020_3_31_22_4/trainedModel_epoch98.pth.tar'
    trained_modelFile_hig='/home/yong/hdd/HIG/output/2020_3_31_22_4/trainedModel_epoch98.pth.tar'
    trained_modelFile_hpe2='/home/yong/hdd/HPE2/output/2020_4_28_20_55/trainedModel_epoch199.pth.tar'
    
    args.gpu_ids='1'
    
    dataset_name='v2' #'v1', 'v2' , 'test', 'test_blur'
    show_enabled=False
    
    filename_method={}
    filename_method['hpe1_orig']='hpe1_refined.txt'#'hpe1_orig.txt'
    filename_method['hig_hpe1']='hig_hpe1.txt'
    filename_method['hpe2']='hpe2.txt'
    filename_method['average']='average.txt'
    
    #--common setting
    args.use_net=['hpe1_orig','hig','hpe1','hpe2']
    args.train_net=[]
    args.cgan=True
    
    
    if dataset_name=='v1':
        data_num=2370
    elif dataset_name=='v2':
        data_num=8000
    elif dataset_name=='test':
        data_num=1000
    elif dataset_name=='test_blur':
        data_num=1000
    
    
    #--load file path
    pcafile_name='pca_52_by_957032_augmented_X2.pickle'
    
    if dataset_name=='v1' or dataset_name=='v2':
        load_filepath_imgs='/home/yong/ssd/dataset/blurHand20/'+dataset_name+'/images/'
        load_filepath_preprocess='/home/yong/ssd/dataset/preprocessed_blurHand20/'+dataset_name+'/'
    elif dataset_name=='test' or dataset_name=='test_blur':
        load_filepath_imgs="/home/yong/ssd/dataset/depth_ir/"+dataset_name+"/" 
        load_filepath_preprocess='/home/yong/ssd/dataset/preprocessed_HIG/'+dataset_name+'/'


    #--save file path
    if dataset_name=='v1' or dataset_name=='v2':
        savefolder_fingertip='/home/yong/hdd/fingertip_test/sota_comparison/'+dataset_name+'/'
        savefolder='/home/yong/hdd/fingertip_test/sota_comparison/'+dataset_name+'/result_imgs/'
    elif dataset_name=='test' or dataset_name=='test_blur':
        savefolder_fingertip='/home/yong/hdd/fingertip_test/ablative/'+dataset_name+'/'
        savefolder='/home/yong/hdd/fingertip_test/ablative/'+dataset_name+'/result_imgs/'
        
    #--common setting
    args.is_train=False
    args.gan_mode='vanilla'
    #args.lambda_L1=50
    args.trainImageSize=128
    trainImageSize=128
    args.skeleton_pca_dim=52
    args.skeleton_orig_dim=63
    args.discriminator_reconstruction=False 
    args.test_batch=1
    args.cgan=True
    
    hpe_numBlocks=5
    
    #device
    device = torch.device("cuda:%s"%args.gpu_ids if torch.cuda.is_available() else "cpu")
    args.device=device
    cudnn.benchmark = True  
    
    
   #--HIG net setting
    if not 'pix2pix' in locals():
        pix2pix=Pix2pix(args) 
        checkpoint=torch.load(trained_modelFile_hig)

        if 'hig' in args.use_net:
            pix2pix.netG.load_state_dict(checkpoint['state_dict_hig'])

        if 'hid' in args.use_net:
            pix2pix.netD.load_state_dict(checkpoint['state_dict_hid'])      
         
    #--HPE net setting 
    if 'trained_modelFile_hpe1' in locals():
        if not 'hpe1' in locals():
            hpe_numBlocks=5
            hpe1=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
            checkpoint=torch.load(trained_modelFile_hpe1)
            hpe1.load_state_dict(checkpoint['state_dict_hpe1'])
    else:
        if not 'hpe1' in locals():
            hpe_numBlocks=5
            hpe1=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
            checkpoint=torch.load(trained_modelFile_hpe1_orig)
            hpe1.load_state_dict(checkpoint['state_dict'])
            
    #--HPE_orig net setting 
    if not 'hpe1_orig' in locals():
        hpe_numBlocks=5
        hpe1_orig=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
        checkpoint=torch.load(trained_modelFile_hpe1_orig)
        hpe1_orig.load_state_dict(checkpoint['state_dict'])
                
    #--HPE2 net setting
    if not 'hpe2' in locals():
        hpe2=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
        checkpoint=torch.load(trained_modelFile_hpe2)
        hpe2.load_state_dict(checkpoint['state_dict_hpe2'])
                   
    with open('/home/yong/ssd/dataset/HANDS17/training/preprocessed/'+pcafile_name,'rb') as f:
        pca=pickle.load(f)
    
    #--fusion net setting
    fusionnet=FusionNet(hpe1_orig,pix2pix,hpe1,hpe2,None,args)    
     
    if 'hpd' in args.use_net:       
        w=np.transpose(pca.components_)
        b=pca.mean_
        fusionnet.set_reconstruction_net(w,b)
    fusionnet=fusionnet.to(device) 
            
    #--dataset of uvr (for utils)
    camera_info={}
    if dataset_name=='v1':
        camera_info['fx']=475.065948/2
        camera_info['fy']=475.065857/2
        camera_info['cx']=315.944855/2
        camera_info['cy']=245.287079 /2
        camera_info['camerawidth']=320
        camera_info['cameraheight']=240       
    else:
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
     
    if dataset_name=='v1' or dataset_name=='v2':
        loadpath_dict="/home/yong/ssd/dataset/preprocessed_blurHand20/dataset_dict_"+dataset_name+".pickle"
    elif dataset_name=='test' or dataset_name=='test_blur':
        loadpath_dict="/home/yong/ssd/dataset/preprocessed_HIG/dataset_dict_"+dataset_name+".pickle"
        
    with open(loadpath_dict,'rb') as f:
        dataset_load=pickle.load(f)   
        
    ##--bag of fingertip position 
    fingertip={}
    fingertip['hpe1_orig']=[]
    fingertip['hig_hpe1']=[]
    fingertip['hpe2']=[]
    fingertip['hpe1_blur']=[]
    fingertip['average']=[]
    

    
    #--start
    fusionnet.set_mode('eval')
    ir_batch = np.zeros((1,1,args.trainImageSize,args.trainImageSize), dtype = np.float32)
    depth_batch = np.zeros((1,1,args.trainImageSize,args.trainImageSize), dtype = np.float32)
    
    vis=Visualize_combined_outputs(utils,4,1,camerawidth,cameraheight)
    

    progressbar=trange(data_num,leave=True)    
    for i in progressbar:
    #for i in range(data_num):
        frame=i
        ##--input
        if dataset_name=='v1':
            depth_orig=cv2.imread(load_filepath_imgs+'depth-%07d.png'%frame,2)
            ir_orig=cv2.imread(load_filepath_imgs+'ir-%07d.png'%frame,2)
        else:#elif dataset_name=='v2':
            depth_orig=cv2.imread(load_filepath_imgs+'depth%d.png'%frame,2)
            ir_orig=cv2.imread(load_filepath_imgs+'ir%d.png'%frame,2)
            
            
        #--preprocess depth/ir
        train_imgs=np.load(load_filepath_preprocess+'%d.npy'%frame)
        depth_train=np.copy(train_imgs[:,0:trainImageSize])
        ir_train=np.copy(train_imgs[:,trainImageSize:2*trainImageSize])
        com=dataset_load['com'][frame]
        window=dataset_load['window'][frame]
                
        depth_crop=datasetloader_uvr.utils.crop(depth_orig,window[0],window[1],window[2],window[3],window[4],window[5])
                        
        # set input
        ir_batch[0,0,:,:]=ir_train.copy()
        depth_batch[0,0,:,:]=depth_train.copy()       
        fusionnet.set_input(ir_batch,depth_batch)
         
        #forward
        #TIME=time.time()
        fusionnet.forward_('hpe1_orig')
        fusionnet.forward_('hig_hpe1')
        fusionnet.forward_('hpe2')
        #fusionnet.forward_('hpe1_blur')
        #print(time.time()-TIME)
            
        #reconstruct
        com3d=utils.unproject2Dto3D(com)  
        out1=fusionnet.reconstruct_joints(pca,com3d,cube,'hpe1_orig','tocpu')
        out2=fusionnet.reconstruct_joints(pca,com3d,cube,'hig_hpe1','tocpu')
        out3=fusionnet.reconstruct_joints(pca,com3d,cube,'hpe2','tocpu')   
        #out4=fusionnet.reconstruct_joints(pca,com3d,cube,'hpe1_blur','tocpu')   
            
        out_average=(out2+out3)/2
        #out_average=(out1+out2+out3)/3                
        
        if show_enabled==True:
            #show result
            depth_pred=fusionnet.out['hig'].detach().cpu().numpy()
            depth_pred=depth_pred[0,0,:,:]
            
            vis.setWindow('hpe1_orig',[0,0])
            vis.setWindow('hig_hpe1',[0,1])
            vis.setWindow('hpe2',[0,2])
            #vis.setWindow('hpe1_blur',[0,3])
            vis.setWindow('average',[0,3])
            #vis.setWindow('hpd',[0,4])
                
            '''
            vis.set_inputimg('hpe1_orig',depth_train)
            vis.convertTo3cInputImages('hpe1_orig')
            vis.set_inputimg('hig_hpe1',ir_train)
            vis.set_inputimg('hig_hpe1',depth_pred)
            vis.convertTo3cInputImages('hig_hpe1')
            vis.set_inputimg('hpe2',ir_train)
            vis.convertTo3cInputImages('hpe2')
            '''
            
            
            '''
            if frame==2002 or frame==2080 or frame==2121 or frame==6245 or frame==7198:
                #depth_seg (for visualization)
                depth_orig[depth_orig>d_maximum]=0
                mask=np.zeros(depth_orig.shape,np.uint16)
                mask[window[2]:window[3],window[0]:window[1]]=1
                depth_orig=depth_orig*mask
            
                #ir_seg (for visualization)         
                ir_orig=Grabcut.segment_grabcut(ir_orig,depth_orig,com,window,d_maximum)#0.6      
            '''
                       
            vis.set_outputimg('hpe1_orig',depth_orig)
            vis.set_outputimg('hig_hpe1',ir_orig)
            vis.set_outputimg('hpe2',ir_orig)
            vis.set_outputimg('average',ir_orig)
            #vis.set_outputimg('hpe1_blur',ir_orig)
            #vis.set_outputimg('hpd',ir_orig)
                
            vis.set_finalimg('hpe1_orig',out1)
            vis.set_finalimg('hig_hpe1',out2)
            vis.set_finalimg('hpe2',out3)
            #vis.set_finalimg('hpe1_blur',out4)
            vis.set_finalimg('average',out_average) 
            #vis.set_finalimg('hpd',out_hpd_best)
            
            '''
            img_save=vis.get_img_save()
            cv2.imshow('result',cv2.resize(img_save,(camerawidth*4,cameraheight)))
            cv2.imwrite(savefolder+'%d.png'%frame,img_save)   
            cv2.waitKey(1)
            '''
            
                       
        #store fingertip position
        fingertip['hpe1_orig'].append(out1[0,17*3:18*3])
        fingertip['hig_hpe1'].append(out2[0,17*3:18*3])
        fingertip['hpe2'].append(out3[0,17*3:18*3])
        #fingertip['hpe1_blur'].append(out4[0,17*3:18*3])
        fingertip['average'].append(out_average[0,17*3:18*3])
        #fingertip['hpd'].append(out_hpd_best[0,11*3:12*3])
        
                  
    #save fingertip position 
    for i,method in enumerate(filename_method):
        filename=filename_method[method]
        print(method,filename)
        saveFingertip(savefolder_fingertip+filename,fingertip[method])        
                  
