import sys

sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from HPE.src.model.deepPrior import dpnet
from HIG.src.model.network import Pix2pix
from Model.fusionNet import FusionNet

#del
from HPD.src.model.skeletonDiscriminator import SkeletonDiscriminator

from datasetloader.datasetLoader_uvr import DatasetLoader as datasetloader_UVR
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
from datasetLoader_icvl import DatasetLoader as datasetloader_ICVL


#from options import train_options
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

#for refine
import torch
import torch.nn as nn
import torch.nn.functional as F

import loss as losses

import copy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch hand pose Training')
    args=parser.parse_args()
    args_pix2pix=parser.parse_args()

    #--user setting
    trained_modelFile_hpe1_orig='/home/yong/hdd/HPE/output/2020_2_26_11_29/trainedModel_epoch99.pth.tar'
    trained_modelFile_hpe1_refine='/home/yong/hdd/HPE/output/2020_2_26_11_29/trainedModel_epoch47.pth.tar'
    #trained_modelFile_hpe1_refine='/home/yong/hdd/HPE_refined/output/2020_4_13_20_18/trainedModel_epoch99.pth.tar'
    
    
    args.gpu_ids='1'
    args.use_net=['hpe1_orig','hpe1_refine']
    #args.use_net=['hpe2','hpe1_orig','hpe1_blur']
    args.train_net=['hpe1_refine']
    args.train_with_inter=False
    
    if args.train_with_inter==False:
        loss_names=['hpe1_refine_mse']
    else:
        loss_names=['hpe1_refine_mse','hpe1_refine_inter']
    
    #--something to test
    args.lambda_loss_hpe1_refine_mse=1
    args.lambda_loss_hpe1_refine_inter=1
    
    args.lr_hpe1_refine=1.0e-5 #1.0e-5
    args.lr_hpd=1.0e-5
    args.weight_decay=1.0e-4 #1.0e-4
    args.cgan=False
    
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
    args.solver='adam'
    args.epochs=100
    args.train_batch=128
    args.momentum=0
    args.gamma=0.9
    args.beta1=0.5
    args.is_train=True
    args.gan_mode='vanilla' #'vanilla'
    args.trainImageSize=128
    args.skeleton_pca_dim=52
    args.skeleton_orig_dim=63
    
    #device
    device = torch.device("cuda:%s"%args.gpu_ids if torch.cuda.is_available() else "cpu")
    args.device=device
    cudnn.benchmark = True
       
    #--HPE_orig net setting 
    if not 'hpe1_orig' in locals():
        hpe_numBlocks=5
        hpe1_orig=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
        checkpoint=torch.load(trained_modelFile_hpe1_orig)
        hpe1_orig.load_state_dict(checkpoint['state_dict'])
        
    #--HPE_blur net setting 
    if not 'hpe1_refine' in locals():
        hpe_numBlocks=5
        hpe1_refine=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
        checkpoint=torch.load(trained_modelFile_hpe1_refine)
        hpe1_refine.load_state_dict(checkpoint['state_dict'])
               
    #--pca
    with open("/home/yong/ssd/dataset/HANDS17/training/preprocessed/pca_52_by_957032_augmented_X2.pickle",'rb') as f:
        pca=pickle.load(f)
        
        
     #--fusion net setting
    fusionnet=FusionNet(hpe1_orig,None,None,None,hpe1_refine,args) 
    #fusionnet=FusionNet(hpe1_orig,None,None,hpe2,hpe1_blur,args) 
    
    if 'hpd' in args.use_net:       
        w=np.transpose(pca.components_)
        b=pca.mean_
        fusionnet.set_reconstruction_net(w,b)
    fusionnet=fusionnet.to(device)  
    
    
    #--number of datasets
    traindataNum_uvr=175000#175000
    traindataNum_vr20=0#215*1  #vr dataset (no blur): 215
    
    validateNum_uvr=10000
    validateNum_vr20=0   #vr test dataset: 2370 

    #--dataset of uvr
    dataset_path_uvr='/home/yong/ssd/dataset/preprocessed_HIG/'
    datasetloader_uvr={}
    datasetloader_uvr['train']=datasetloader_UVR(dataset_path_uvr,traindataNum_uvr,camera_info,'train','generator')
    datasetloader_uvr['train'].set_datasetLoader(args.train_batch)
    
    #--dataset of VR20
    camera_info_vr20={}
    camera_info_vr20['fx']=475.065948/2
    camera_info_vr20['fy']=475.065857/2
    camera_info_vr20['cx']=315.944855/2
    camera_info_vr20['cy']=245.287079/2
    camera_info_vr20['camerawidth']=int(640/2)
    camera_info_vr20['cameraheight']=int(480/2)
    camera_info_vr20['cube']=np.asarray([250,250,250])
    camera_info_vr20['trainImageSize']=128
    dataset_path_vr20='/home/yong/ssd/dataset/preprocessed_blurHand20/'
    datasetloader_uvr['train_vr20']=datasetloader_UVR(dataset_path_vr20,traindataNum_vr20,camera_info_vr20,'v1_noblur_shuffle','generator')    
    datasetloader_uvr['train_vr20'].set_datasetLoader(args.train_batch)
    
    #--validation set of VR20
    if validateNum_vr20>0:
        datasetloader_uvr['test']=datasetloader_UVR(dataset_path_vr20,validateNum_vr20,camera_info_vr20,'v1','generator')
        datasetloader_uvr['test'].set_datasetLoader(args.train_batch)
    else:
        datasetloader_uvr['test']=datasetloader_UVR(dataset_path_uvr,validateNum_uvr,camera_info,'test','generator')
        datasetloader_uvr['test'].set_datasetLoader(args.train_batch)
        
        
    #--dataset of bighand
    if not 'datasetloader_icvl' in locals():
        datasetloader_icvl=datasetloader_ICVL(args.skeleton_pca_dim)
        filepath_icvl={}
        filepath_icvl['pca']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/pca_52_by_957032_augmented_X2.pickle"
        filepath_icvl['gt_train']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/dataset_dict_train_full_52_by_957032_augmented_X2.pickle"
        filepath_icvl['gt_test']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/dataset_dict_test_full_52_by_957032_augmented_X2.pickle"
        filepath_icvl['gt_full_train']="/home/yong/ssd/dataset/HANDS17/training/original/Training_Annotation.txt"
        filepath_icvl['gt_full_test']="/home/yong/ssd/dataset/HANDS17/training/original/Test_Annotation.txt"
            
        filepath_icvl['image_train']=None
        filepath_icvl['image_train_each']="/home/yong/ssd/dataset/preprocessed_HPE/images_train_augmented_X2/%d.npy"
        filepath_icvl['image_test']=None
        filepath_icvl['image_test_each']="/home/yong/ssd/dataset/preprocessed_HPE/images_test/%d.npy"
        
        datasetloader_icvl.load(filepath_icvl)
    
    ###---create result folder---###
    now=datetime.now()
    savefolder='/home/yong/hdd/HPE_refined/output/%d_%d_%d_%d_%d/'%(now.year,now.month,now.day,now.hour,now.minute)
    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)
    print('save folder:',savefolder)
    
    ###--save configuration--###
    f=open(savefolder+'setting.txt','w')
    for arg in vars(args):
        f.write('%s:%s\n'%(arg,getattr(args,arg)))
    f.write('loaded HPE#1_orig net:%s\n'%trained_modelFile_hpe1_orig)
    f.write('loaded HPE#1_refine net:%s\n'%trained_modelFile_hpe1_refine)
                    
    f.write('traindataNum_uvr:%d\n'%traindataNum_uvr)
    f.close()
    
    #--start 
    progress_train=progress.Progress(loss_names,pretrain=False)
    progress_validate=progress.Progress(loss_names,pretrain=False)
    print('start..')
    for epoch in range(args.epochs):
        #--train
        fusionnet.set_mode('train')
        
        #dataset generator
        datasetloader_uvr['train'].shuffle()
        if traindataNum_vr20>0:
            datasetloader_uvr['train_vr20'].shuffle()
        generator_train_icvl=datasetloader_icvl.generator_learningData(args.train_batch,'train','noaug',0)
        
        #trange number
        trange_num=int(175000)//args.train_batch
        
        if traindataNum_vr20>0:
            trange_num+=int(traindataNum_vr20)//args.train_batch
        if  traindataNum_uvr>0:
            trange_num+=int(traindataNum_uvr)//args.train_batch
                
        progressbar=trange(trange_num,leave=True)   
        for i in progressbar:
            #select dataset
            get_uvrdata,get_icvldata,get_vr20data=False,False,False
            fusionnet.use_blurdata=False
            
            if i%2==0:
                get_uvrdata=True
            else:
                get_icvldata=True
                
            
            #input
            if get_icvldata==True:
                img_icvl,hpose_icvl,_,_=next(generator_train_icvl)
                fusionnet.set_input_icvl(img_icvl,hpose_icvl)
            elif get_uvrdata==True:
                img_depth,img_ir=datasetloader_uvr['train'].load_data()
                fusionnet.set_input(img_ir,img_depth)
            else:
                img_depth,img_ir=datasetloader_uvr['train_vr20'].load_data()
                fusionnet.set_input(img_ir,img_depth)
                
            #optimize
            if get_icvldata==True:
                fusionnet.optimize_parameters_hpe1_bighand()
            else:
                fusionnet.optimize_parameters_hpe1_depthIR()
            
            #forward and backward
            if get_icvldata==True:
                fusionnet.calculateloss_hpe1_bighand()
            else:
                fusionnet.calculateloss_hpe1_depthIR()
                
            loss_=fusionnet.getloss(loss_names)
            progress_train.append_local(loss_)
            
            #print progress
            if i<trange_num-1:
                message='epc:%d: '%epoch
                for i in range(len(loss_names)):
                    message+='(%.5f)'%loss_[i]
                #message='epc:%d (loss) hig_hpe1:%.4f %.4f %.4f %.4f ||| hpe2:%.4f %.4f %.4f %.4f'%(epoch,loss_[0],loss_[1],loss_[2],loss_[3],loss_[4],loss_[5],loss_[6],loss_[7])
            else:
                loss_avg=progress_train.get_average()
                message='epc:%d: '%epoch
                for i in range(len(loss_names)):
                    message+='(%.5f)'%loss_avg[i]
                #message='epc:%d (loss) hig_hpe1:%.4f %.4f %.4f %.4f ||| hpe2:%.4f %.4f %.4f %.4f'%(epoch,loss_avg[0],loss_avg[1],loss_avg[2],loss_avg[3],loss_avg[4],loss_avg[5],loss_avg[6],loss_avg[7])
            progressbar.set_description(message)
            #print(message)
            
        progress_train.append_loss()    
        
        #--test
        
        fusionnet.set_mode('eval')
        
        generator_test_icvl=datasetloader_icvl.generator_learningData(args.train_batch,'validate',False,0)
        
        if traindataNum_vr20>0:
            trange_num=validateNum_vr20//args.train_batch 
        else:
            trange_num=validateNum_uvr//args.train_batch
            
        progressbar2=trange(trange_num,leave=True) 
        for i in progressbar2:
        #for i in range(validateNum_uvr//args.train_batch):
            #input
            fusionnet.use_blurdata=False
            
            #v1(vr20) or v2 test dataset
            img_depth,img_ir=datasetloader_uvr['test'].load_data()
            fusionnet.set_input(img_ir,img_depth)
            fusionnet.calculateloss_hpe1_depthIR()
            
            # bighand test dataset
            '''
            img_icvl,hpose_icvl,_,_=next(generator_train_icvl)
            fusionnet.set_input_icvl(img_icvl,hpose_icvl)
            fusionnet.calculateloss_hpe1_bighand()
            '''
            
            #get loss
            loss_=fusionnet.getloss(loss_names)
            progress_validate.append_local(loss_)
            
            #print progress
            if i<trange_num-1:
                message='   (val) epc:%d: '%epoch
                for i in range(len(loss_names)):
                    message+='-%.5f-'%loss_[i]
                #message='--epc:%d (loss) hig_hpe1:%.4f %.4f %.4f %.4f ||| hpe2:%.4f %.4f %.4f %.4f'%(epoch,loss_[0],loss_[1],loss_[2],loss_[3],loss_[4],loss_[5],loss_[6],loss_[7])
            else:
                loss_avg=progress_validate.get_average()
                message='   (val) epc:%d: '%epoch
                for i in range(len(loss_names)):
                    message+='-%.5f-'%loss_avg[i]
                #message='--epc:%d (loss) hig_hpe1:%.4f %.4f %.4f %.4f ||| hpe2:%.4f %.4f %.4f %.4f'%(epoch,loss_avg[0],loss_avg[1],loss_avg[2],loss_avg[3],loss_avg[4],loss_avg[5],loss_avg[6],loss_avg[7])
            progressbar2.set_description(message)
            #print(message)
            
        progress_validate.append_loss()  
        
        
        #save model
        if progress_validate.losses['hpe1_refine_mse'][-1]<progress_validate.loss_best:
            progress_validate.loss_best=progress_validate.losses['hpe1_refine_mse'][-1]
            state_best={
                    'epoch':epoch,
                    'best_loss':progress_validate.loss_best,
                    'state_dict':fusionnet.net_hpe1_refine.state_dict(),
                    'optimizer':fusionnet.optimizer_hpe1_refine.state_dict()}
            torch.save(state_best,savefolder+'trainedModel_epoch%d.pth.tar'%epoch)
        else:
            if epoch%10==0:
                state_best={
                    'epoch':epoch,
                    'best_loss':progress_validate.loss_best,
                    'state_dict':fusionnet.net_hpe1_refine.state_dict(),
                    'optimizer':fusionnet.optimizer_hpe1_refine.state_dict()}
                torch.save(state_best,savefolder+'trainedModel_epoch%d.pth.tar'%epoch)
                
        
            
        ###--save loss plot (G_L1,G_gAN)--###        
        progress_train.save_plot(savefolder,'hpe1_refine','train','train_hpe1_refine.png')
        progress_validate.save_plot(savefolder,'hpe1_refine','val','val_hpe1_refine.png')
        
        
    torch.save(state_best,savefolder+'trainedModel_epoch%d.pth.tar'%epoch)
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
