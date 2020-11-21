''' cite:   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import torch.backends.cudnn as cudnn
    
#from model.loss import GANLoss
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
import loss as losses

from loss.ganloss import GANLoss

from segNet import Pix2pix

    
def getdataBatch(args,it,frame_idx,dataNum_list,filepath_list):
    '''batch'''
    data_batch=np.zeros((args.batch_size,1,args.cnnimg_size,args.cnnimg_size))
    label_batch=np.zeros((args.batch_size,3,args.cnnimg_size,args.cnnimg_size))
    
    for j in range(args.batch_size):
        frame=frame_idx[it*args.batch_size+j]        
        if frame>=0 and frame<dataNum_list[0]:
            datasetid=0
            frame_local=frame
        elif frame>=dataNum_list[0] and frame<sum(dataNum_list[:2]):
            datasetid=1
            frame_local=frame-dataNum_list[0]
        else:
            datasetid=2
            frame_local=frame-sum(dataNum_list[:2])
            
        depth=cv2.imread(filepath_list[datasetid]+"depth\\image_%05u_depth.png"%frame_local,2)
        depth=(depth-depth.min())/(depth.max()-depth.min())
        #depth=depth/1000
        
        label=cv2.imread(filepath_list[datasetid]+"label_filtered\\image_%05u_label.bmp"%frame_local,2)
        label2=cv2.resize(label,(args.cnnimg_size,args.cnnimg_size),interpolation=cv2.INTER_AREA)
                
        data_batch[j,0]=cv2.resize(depth,(args.cnnimg_size,args.cnnimg_size))
        
        label_batch[j,0][label2==255]=1
        label_batch[j,1][label2==0]=1
        label_batch[j,2][label2==1]=1
        
        '''
        label_batch[j,0][label2==255]=0
        label_batch[j,0][label2==0]=1
        label_batch[j,0][label2==1]=2
        '''
            
    return data_batch,label_batch
    

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import argparse
    import plotloss
    import tqdm
    from Sensor.Realsense import Realsense
    
    parser = argparse.ArgumentParser()
    args= parser.parse_args()
        
    '''user setting'''
    weight_filepath='..\\..\\weights\\segmentation_20201027\\'
    args.segnet_weightfile=weight_filepath+'segmentation-60.pth.tar'
    debug=False
    data_type='camera' #'camera' or 'image'
    save=False
    
    
    '''common setting'''
    #filepath
    filepath_list=["D:\\research\\handdataset\\TwoPaintedHands\\paintedHands\\ego\\output_user01\\",
                   "D:\\research\\handdataset\\TwoPaintedHands\\paintedHands\\ego\\output_user02\\",
                   "D:\\research\\handdataset\\TwoPaintedHands\\paintedHands\\ego\\output_user04\\"]
    
    
    #parameters
    args.max_epoch=100
    args.is_train=True
    args.cgan=True
    args.gan_mode='vanilla'
    args.lr_generator=1e-3
    args.lr_discriminator=1e-3
    args.weight_decay=0.01
    args.lr_decay_step=20
    args.lr_decay_gamma=0.9
    args.device=0
    args.cnnimg_size=256
    args.batch_size=1

    
    save_epoch_iter=10
   
    dataNum_test=300
    dataNum_list=[3282,3109,3058-dataNum_test]
    
    dataNum_train=sum(dataNum_list)
    
    
    args.loss_weight={}
    args.loss_weight['G_L1']=100
    args.loss_weight['D_real']=0.5
    args.loss_weight['D_fake']=0.5
    
    
    #network
    pix2pix=Pix2pix(args,1,3)
    
    printed_loss_names=['G_L1','G_GAN','D_real','D_fake']
    plotloss_train=plotloss.Plotloss(printed_loss_names)
    plotloss_test=plotloss.Plotloss(printed_loss_names)
    
    #realsense
    realsense=Realsense()
    
    '''run'''
    data_batch=np.zeros((args.batch_size,1,args.cnnimg_size,args.cnnimg_size))
    
    frame_idx_train=np.arange(dataNum_train)
    frame_idx_test=np.arange(dataNum_train,dataNum_train+dataNum_test)
    
    try:
        frame=0
        while(1):
            if data_type=='image':
                data_batch,label_batch=getdataBatch(args,frame,frame_idx_test,dataNum_list,filepath_list)
                
                #evaluate
                pix2pix.set_input(data_batch)
                    
                seg_pred=pix2pix.forward(pix2pix.real_A)
                seg_pred=seg_pred.detach().cpu().numpy()
                
                hands_mask=np.zeros((seg_pred[0]).shape)
                
                hands_mask[0,np.argmax(seg_pred[0],axis=0)!=0]=1
                hands_mask[1,np.argmax(seg_pred[0],axis=0)==1]=1
                hands_mask[2,np.argmax(seg_pred[0],axis=0)==2]=1
                
                vis_img=np.zeros((2*args.cnnimg_size,2*args.cnnimg_size))
                vis_img[0:args.cnnimg_size,0:args.cnnimg_size]=255*data_batch[0,0]
                vis_img[0:args.cnnimg_size,1*args.cnnimg_size:2*args.cnnimg_size]=255*hands_mask[0]
                vis_img[1*args.cnnimg_size:2*args.cnnimg_size,0:args.cnnimg_size]=255*hands_mask[1]
                vis_img[1*args.cnnimg_size:2*args.cnnimg_size,1*args.cnnimg_size:2*args.cnnimg_size]=255*hands_mask[2]
                cv2.imshow('vis',vis_img)
                
                cv2.imwrite(weight_filepath+'test-result\\vis-%d.png'%frame,vis_img)
                
                cv2.waitKey(1)
                frame+=1
                
                     
            elif data_type=='camera':
                #sensor input
                realsense.run()
                depth_orig=realsense.getDepthImage()
                ir_orig=realsense.getIrImage()
                rgb_orig=realsense.getColorImage() 
            
                depth_orig[depth_orig>500]=0
            
                #
                depth=(depth_orig-depth_orig.min())/(depth_orig.max()-depth_orig.min())
                data_batch[0,0]=cv2.resize(depth,(args.cnnimg_size,args.cnnimg_size))
                
                pix2pix.set_input(data_batch)
                seg_pred=pix2pix.forward(pix2pix.real_A)
                seg_pred=seg_pred.detach().cpu().numpy()
                
                hands_mask=np.zeros((seg_pred[0]).shape)
                
                hands_mask[0,np.argmax(seg_pred[0],axis=0)!=0]=1
                hands_mask[1,np.argmax(seg_pred[0],axis=0)==1]=1
                hands_mask[2,np.argmax(seg_pred[0],axis=0)==2]=1
                
                vis_img=np.zeros((2*args.cnnimg_size,2*args.cnnimg_size))
                vis_img[0:args.cnnimg_size,0:args.cnnimg_size]=255*data_batch[0,0]
                vis_img[0:args.cnnimg_size,1*args.cnnimg_size:2*args.cnnimg_size]=255*hands_mask[0]
                vis_img[1*args.cnnimg_size:2*args.cnnimg_size,0:args.cnnimg_size]=255*hands_mask[1]
                vis_img[1*args.cnnimg_size:2*args.cnnimg_size,1*args.cnnimg_size:2*args.cnnimg_size]=255*hands_mask[2]
                cv2.imshow('vis',vis_img)
                
                if save:
                    cv2.imwrite(weight_filepath+'realtime-result\\vis-%d.png'%frame,vis_img)
                
                cv2.waitKey(1)
                frame+=1
                
    finally:
        print('stop device')
        realsense.release()
        
        
        
        