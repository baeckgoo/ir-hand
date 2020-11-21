import sys

sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from HPE.src.model.deepPrior import dpnet
from HIG.src.model.network import Pix2pix
from Model.fusionNet import FusionNet


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
import time

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
    trained_modelFile_hpe1_orig='/home/yong/hdd/HPE/output/2020_2_26_11_29/trainedModel_epoch99.pth.tar'
    trained_modelFile_hpe1='/home/yong/hdd/HIG/output/2020_3_31_22_4/trainedModel_epoch98.pth.tar'
    trained_modelFile_hig='/home/yong/hdd/HIG/output/2020_3_31_22_4/trainedModel_epoch98.pth.tar'
    trained_modelFile_hpe2='/home/yong/hdd/HPE2/output/2020_4_4_0_36/trainedModel_epoch90.pth.tar'
    
    args.gpu_ids='0'  #'0','1'
    args.use_net=['hpe1_orig','hig','hpe1','hpe2']
    args.train_net=['hpe2']
    args.train_with_inter=False
    args.cgan=True
    
    
    if args.train_with_inter==True:
        loss_names=['hpe2_mse','hpe2_inter']    
    else:
        loss_names=['hpe2_mse']
    
    #--something to test             
    args.lambda_loss_hpe2_mse=10
    args.lambda_loss_hpe2_inter=10
    args.lambda_loss_hpe2_gan=0#0.1      #if discriminator_skeleton==true
    args.lambda_loss_hpd_fake=0#0.7       #if discriminator_skeleton==true
    args.lambda_loss_hpd_real_icvl=0#0.3#0.5 #if discriminator_skeleton==true
    args.lr_hpe2=1.0e-5#5.0e-6 #1.0e-5
    args.weight_decay=5.0e-4#1.0e-4
    
    args.lr_hpd=1.0e-6
    
    #--common setting
    args.solver='adam'
    args.epochs=200
    args.train_batch=64
    args.momentum=0
    args.gamma=0.9
    args.beta1=0.5
    args.is_train=True
    args.gan_mode='vanilla' #'vanilla'
    args.trainImageSize=128
    args.skeleton_pca_dim=52
    args.skeleton_orig_dim=63
          
    #--device
    device = torch.device("cuda:"+args.gpu_ids if torch.cuda.is_available() else "cpu")
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
    if not 'hpe1' in locals():
        hpe_numBlocks=5
        hpe1=dpnet(args.skeleton_pca_dim,hpe_numBlocks,device)
        checkpoint=torch.load(trained_modelFile_hpe1)
        hpe1.load_state_dict(checkpoint['state_dict_hpe1'])

    
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

    #--pca
    with open("/home/yong/ssd/dataset/HANDS17/training/preprocessed/pca_52_by_957032_augmented_X2.pickle",'rb') as f:
        pca=pickle.load(f)
         
    #--fusion net setting
    fusionnet=FusionNet(hpe1_orig,pix2pix,hpe1,hpe2,None,args)    
    if 'hpd' in args.use_net:       
        w=np.transpose(pca.components_)
        b=pca.mean_
        fusionnet.set_reconstruction_net(w,b)
    fusionnet=fusionnet.to(device)  
    
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
    
    #--number of datasets
    traindataNum_uvr=175000#170000
    traindataNum_blur_uvr=25000
    traindataNum_vr20=0  #vr dataset (slow motion): 215
    
    validateNum=10000       
    validateNum_vr20=0   #vr test dataset: 2370 
    
    #--dataset of uvr
    dataset_path_uvr='/home/yong/ssd/dataset/preprocessed_HIG/'
    datasetloader_uvr={}
    datasetloader_uvr['train']=datasetloader_UVR(dataset_path_uvr,traindataNum_uvr,camera_info,'train','generator')
    datasetloader_uvr['train_blur']=datasetloader_UVR(dataset_path_uvr,traindataNum_blur_uvr,camera_info,'train_blur','generator')
    

    datasetloader_uvr['train'].set_datasetLoader(args.train_batch)
    datasetloader_uvr['train_blur'].set_datasetLoader(args.train_batch)
    

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
    
    #--validation set
    if validateNum_vr20>0:
        datasetloader_uvr['test']=datasetloader_UVR(dataset_path_vr20,validateNum_vr20,camera_info_vr20,'v1','generator')
        datasetloader_uvr['test'].set_datasetLoader(args.train_batch)
    else:
        datasetloader_uvr['test']=datasetloader_UVR(dataset_path_uvr,validateNum,camera_info,'test','generator')
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
    savefolder='/home/yong/hdd/HPE2/output/%d_%d_%d_%d_%d/'%(now.year,now.month,now.day,now.hour,now.minute)
    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)
    print('save folder:',savefolder)
    
    ###--save configuration--###
    f=open(savefolder+'setting.txt','w')
    for arg in vars(args):
        f.write('%s:%s\n'%(arg,getattr(args,arg)))
    f.write('loaded HPE#1 net:%s\n'%trained_modelFile_hpe1_orig)
    f.write('traindataNum_uvr:%s\n'%traindataNum_uvr)
    f.write('traindataNum_blur_uvr:%s\n'%traindataNum_blur_uvr)
    f.write('traindataNum_VR20:%s\n'%traindataNum_vr20)
    f.write('validateNum_uvr:%s\n'%validateNum)
    f.close()
    
    #--start
    progress_train=progress.Progress(loss_names,pretrain=False)
    progress_validate=progress.Progress(loss_names,pretrain=False)

    iternum_train=traindataNum_uvr//args.train_batch
    iternum_train_blur=traindataNum_blur_uvr//args.train_batch
    iternum_vr20=traindataNum_vr20//args.train_batch
    print('start..')
    
    for epoch in range(args.epochs):
        #--train
        fusionnet.set_mode('train')
        
        datasetloader_uvr['train'].shuffle()
        datasetloader_uvr['train_blur'].shuffle()
        
        #trange_num=traindataNum_uvr//args.train_batch + traindataNum_blur_uvr//args.train_batch +traindataNum_vr20//args.train_batch
        if traindataNum_vr20>0:
            trange_num=2*traindataNum_vr20//args.train_batch
        elif traindataNum_blur_uvr>0:
            trange_num=2*traindataNum_blur_uvr//args.train_batch
        else:
            trange_num=traindataNum_uvr//args.train_batch
            
        progressbar=trange(trange_num,leave=True)   
        for i in progressbar:
            #select dataset
            get_noblurdata,get_blurdata=False,False
            fusionnet.use_blurdata=False
                   
            if traindataNum_blur_uvr>0:
                if i%2==0:
                    get_blurdata=True
                    fusionnet.use_blurdata=True
                else:
                    get_noblurdata=True
            else:
                get_noblurdata=True
        
            #load data from the selected dataset
            if get_blurdata==True:
                img_depth,img_ir=datasetloader_uvr['train_blur'].load_data()
            elif get_noblurdata==True:
                img_depth,img_ir=datasetloader_uvr['train'].load_data()
                
            #optimize
            fusionnet.set_input(img_ir,img_depth) #0.0005 sec
            fusionnet.optimize_parameters_hpe2() #0.25
            
                     
            #forward and backward
            fusionnet.calculateloss_hpe2() #0.08 sec
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
            trange_num=validateNum//args.train_batch
            
        progressbar2=trange(trange_num,leave=True) 
        for i in progressbar2:
        #for i in range(validateNum_uvr//args.train_batch):
            #input
            img_depth,img_ir=datasetloader_uvr['test'].load_data()
            
            if 'hpd' in args.train_net:
                img_icvl,hpose_icvl,_,_=next(generator_test_icvl)
                fusionnet.set_input_icvl(img_icvl,hpose_icvl)
            
            fusionnet.set_input(img_ir,img_depth)
            
            
            #forward
            fusionnet.calculateloss_hpe2()
        
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
        if progress_validate.losses['hpe2_mse'][-1]<progress_validate.loss_best:
            progress_validate.loss_best=progress_validate.losses['hpe2_mse'][-1]
            if 'hpd' in args.train_net:
                state_best={
                    'epoch':epoch+1,
                    'best_loss':progress_validate.loss_best,
                    'state_dict_hpe2':fusionnet.net_hpe2.state_dict(),
                    'optimizer_hpe2':fusionnet.optimizer_hpe2.state_dict(),
                    'state_dict_hpd':fusionnet.net_hpd.state_dict(),
                    'optimizer_hpd':fusionnet.optimizer_hpd.state_dict()}
            else:
                state_best={
                    'epoch':epoch+1,
                    'best_loss':progress_validate.loss_best,
                    'state_dict_hpe2':fusionnet.net_hpe2.state_dict(),
                    'optimizer_hpe2':fusionnet.optimizer_hpe2.state_dict()}              
            torch.save(state_best,savefolder+'trainedModel_epoch%d.pth.tar'%epoch)
            
        ###--save loss plot (G_L1,G_gAN)--###
        if 'hig' in args.train_net:
            progress_train.save_plot(savefolder,'hig','train','train_hig.png')
            progress_validate.save_plot(savefolder,'hig','val','val_hig.png')
        
        if 'hid' in args.train_net:
            progress_train.save_plot(savefolder,'hid','train','train_hid.png')
            progress_validate.save_plot(savefolder,'hid','val','val_hid.png')
        
        progress_train.save_plot(savefolder,'hpe2','train','train_hpe2.png')
        progress_validate.save_plot(savefolder,'hpe2','val','val_hpe2.png')
        
        if 'hpd' in args.train_net:
            progress_train.save_plot(savefolder,'hpd','train','train_hpd.png')
            progress_validate.save_plot(savefolder,'hpd','val','val_hpd.png')
                    
            
    
    ###--save model--###
    torch.save(state_best,savefolder+'trainedModel_epoch%d.pth.tar'%epoch)
    
    print('used HPE net:',trained_modelFile_hpe1_orig)
    print('save folder:',savefolder)
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
