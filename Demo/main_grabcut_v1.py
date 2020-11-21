import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
import cv2
import numpy as np
import pickle

import sys
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
from datasetLoader_uvr import Utils 
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from datasetloader.datasetLoader_uvr import DatasetLoader as datasetloader_UVR

from GrabCut import Grabcut

if __name__=="__main__":
    ##--user setting
    slow_frames=np.arange(0,300)
    fast_frames=np.load('/home/yong/ssd/dataset/preprocessed_blurHand20/fast_frames.npy')
    slow_fast_frames=np.concatenate((slow_frames,fast_frames))
    
    '''
    data_num=2370
    idx_data=np.arange(data_num)
    np.random.shuffle(idx_data)
    '''
    idx_data=slow_fast_frames
    np.random.shuffle(idx_data)

    
    dataset_version='v1_shuffle' #'v1', 'v1_shuffle',  'v2'
    save_filepath_img='/home/yong/ssd/dataset/preprocessed_blurHand20/v1_shuffle/' #'v1', 'v1_shuffle', 'v2'
    
    
    load_filepath_img='/home/yong/ssd/dataset/blurHand20/v1/images/'
    savepath_dict="/home/yong/ssd/dataset/preprocessed_blurHand20/dataset_dict_"+dataset_version+".pickle"
    
    
    
    save_enable=True
    
    ##--common setting
    
    d_minimum=0
    d_maximum=500
    
    camera_info={}
    camera_info['fx']=475.065948/2
    camera_info['fy']=475.065857/2
    camera_info['cx']=315.944855/2
    camera_info['cy']=245.287079/2
    camera_info['camerawidth']=640/2
    camera_info['cameraheight']=480/2
    camera_info['cube']=np.asarray([250,250,250])
    camera_info['trainImageSize']=128
    trainImageSize=camera_info['trainImageSize']
    cube=camera_info['cube']
    
    
    utils=Utils(camera_info)
    imgs_train=np.zeros((trainImageSize,2*trainImageSize),'float32')
    datasetloader_uvr=datasetloader_UVR('..',0,camera_info,'..','..')
    
    dataset_dict={}
    dataset_dict['com']=[]
    dataset_dict['window']=[]
    
    #--start
    
    progressbar=trange(len(idx_data),leave=True)  
    for i in progressbar:
    #for frame in range(242,data_num):    
        #frame=i
        frame=idx_data[i]
        
        depth=cv2.imread(load_filepath_img+'depth-%07d.png'%frame,2)
        ir=cv2.imread(load_filepath_img+'ir-%07d.png'%frame,2)
        
        depth_seg=depth.copy()
        depth_seg[depth_seg>d_maximum]=0
        
        
        
        # preprocess (depth)
        depth_train,depth_crop,com,window=datasetloader_uvr.preprocess_depth(depth_seg,trainImageSize,cube)
        #depth_train,depth_crop,com,window=preprocess_depth(utils,depth_seg,trainImageSize,cube)
            
    
        # preprocess (ir) --(method 1) with background
        #ir_train=datasetloader_uvr.preprocess_ir(ir,window)  
        
        #grabcut (ir)  --(method 2) without background
        ir_seg=Grabcut.segment_grabcut(ir,depth_seg,com,window)
        
        #plt.imshow(ir_seg)
        
        # preprocess (ir) --(method 2) without background
        ir_train=datasetloader_uvr.preprocess_ir(ir_seg,window)  
                
        # save image
        imgs_train[:,0:trainImageSize]=depth_train
        imgs_train[:,trainImageSize:2*trainImageSize]=ir_train
            
        cv2.imshow("imgs_train",imgs_train)
        cv2.waitKey(1)    
            
        #jio
        
        if save_enable==True:
            np.save(save_filepath_img+'%d.npy'%i,imgs_train)
            dataset_dict['com'].append(com)
            dataset_dict['window'].append(window)
            
        
    
    with open(savepath_dict,'wb') as f:
        pickle.dump(dataset_dict,f,pickle.HIGHEST_PROTOCOL)
    
    
        