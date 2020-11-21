from datasetLoader_icvl import DatasetLoader
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from numpy.testing import assert_array_almost_equal

from tqdm import tqdm
from tqdm import trange
import pickle
import time


#def func()    
if __name__=="__main__":   
    #--user setting  
    '''
    option="train"
    ratio="full"
    num_class=52
    pca_name="52_by_957032_augmented_X2"  # 45_by_119629 , 45_by_119629_augmented
    augmentation_num=2
    save_augmented_dataset=True
    saveImage=True
    
    sigma_com = 5.
    rot_range=180.
    sigma_sc = 0.02
    '''
    
    option="test"
    ratio="full"
    num_class=52
    pca_name="52_by_957032_augmented_X2"
    save_augmented_dataset=False
    saveImage=False
    
    
    
    #--common setting
    if option=="train" and ratio=="eighth":
        datanumber=119629 #295510#957032
    if option=="train" and ratio=="full":
        datanumber=957032 #295510#957032
    if option=="test" and ratio=="full":
        datanumber=295510
    
  
    if not 'datasetloader' in locals():
        datasetloader=DatasetLoader(num_class)   
        filepath={}
        filepath['pca']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/pca_"+pca_name+".pickle"
        filepath['gt_train']=None
        filepath['gt_test']=None
        filepath['gt_full_train']="/home/yong/ssd/dataset/HANDS17/training/original/Training_Annotation.txt"
        filepath['gt_full_test']="/home/yong/ssd/dataset/HANDS17/training/original/Test_Annotation.txt"
        
        filepath['image_train']=None
        filepath['image_train_each']=None
        filepath['image_test']=None
        filepath['image_test_each']=None
        
        datasetloader.load(filepath)
        
        
    if save_augmented_dataset==False:
        savepath_img="/home/yong/ssd/dataset/preprocessed_HPE/images_"+option+"/"
    else:
        savepath_img="/home/yong/ssd/dataset/preprocessed_HPE/images_"+option+"_augmented_X%s"%augmentation_num+"/"
    savepath_dict="/home/yong/ssd/dataset/HANDS17/training/preprocessed/dataset_dict_"+option+"_"+ratio+"_"+pca_name+".pickle" 
        
    print('savepath_dict:',savepath_dict)    
    
    #saved dataset
    dataset_dict={}
    dataset_dict['frames_valid']=[]
    dataset_dict['com']=[]
    dataset_dict['com3d']=[]
    dataset_dict['label']=[]
    dataset_dict['label_embed']=[]
    dataset_dict['window']=[]
    
    
    jointNum=datasetloader.jointNum
    #119629
    
    #--start
    frame=0
    
    progressbar=trange(datanumber,leave=True)
    for iii in progressbar:
        if option=="train":  
            img=datasetloader.loadImage("/home/yong/ssd/dataset/HANDS17/training/original/images_train/",iii)
        elif option=="test":
            img=datasetloader.loadImage("/home/yong/ssd/dataset/HANDS17/training/original/images_test/",iii) 
        
        if option=="train":
            joints3d_gt=datasetloader.joint3d_gt_train[iii,:]
        elif option=="test":
            joints3d_gt=datasetloader.joint3d_gt_test[iii,:]
            
        img_seg,isvalid_bool=datasetloader.segmentHand(img,joints3d_gt)
        if isvalid_bool==False:
            dataset_dict['com'].append(None)
            dataset_dict['com3d'].append(None)
            dataset_dict['window'].append(None)
            dataset_dict['label'].append(None)
            dataset_dict['label_embed'].append(None)
            frame+=1
            continue
        
        #without augmentation
        dataset_dict['frames_valid'].append(frame)
            
        cube=datasetloader.cube
       
        #make com
        com=datasetloader.utils.calculateCOM(img_seg)
        com,img_crop,window=datasetloader.utils.refineCOMIterative(img_seg,com,3)
        dataset_dict['com'].append(com)
        dataset_dict['window'].append(window)
        
        #make cnn image          
        img_crop_norm=datasetloader.utils.makeLearningImage(img_crop,com)
        if saveImage==True:
            np.save(savepath_img+"%d.npy"%frame,img_crop_norm)
            #else:
            #    img_crop_norm_bag[frame,:,:]=img_crop_norm

        
        #make train label         
        com3d=datasetloader.utils.unproject2Dto3D(com)   
        joints3d_norm=joints3d_gt-np.tile(com3d,(1,jointNum))
        joints3d_norm2=joints3d_norm/(cube[2]/2.)
        joints3d_norm3=np.clip(joints3d_norm2,-1,1)  
        trainlabel_embed=datasetloader.pca.transform(joints3d_norm3)  
        dataset_dict['label'].append(joints3d_gt)
        dataset_dict['label_embed'].append(trainlabel_embed)
        dataset_dict['com3d'].append(com3d)
            
        
        #with augmentation
        frame+=1
        joints3d_=np.reshape(joints3d_gt,(21,3))
        if save_augmented_dataset==True:
            for ag_id in range(augmentation_num):
                dataset_dict['frames_valid'].append(frame)
                
                datasetloader.utils.set_augmentation_parameters(sigma_com,rot_range,sigma_sc)
                img_seg_aug,img_train_aug,com2d_aug,com3d_aug,window_aug,joints3d_aug,joints3d_norm3_aug=datasetloader.utils.augment(img_seg,joints3d_,com)
                
                #augmented image
                if saveImage==True:
                    np.save(savepath_img+"%d.npy"%frame,img_train_aug)
               
                #augmented label
                trainlabel_embed_aug=datasetloader.pca.transform(joints3d_norm3_aug)  
                
                dataset_dict['com'].append(com2d_aug)
                dataset_dict['window'].append(window_aug)
                dataset_dict['label'].append(joints3d_aug)
                dataset_dict['label_embed'].append(trainlabel_embed_aug)
                dataset_dict['com3d'].append(com3d_aug)
                
                frame+=1
                
                #debug
                '''
                cv2.line(img_seg_aug,(window_aug[0],window_aug[2]),(window_aug[1],window_aug[2]),(255,0,0),5)
                cv2.line(img_seg_aug,(window_aug[0],window_aug[2]),(window_aug[0],window_aug[3]),(255,0,0),5)
                cv2.line(img_seg_aug,(window_aug[1],window_aug[2]),(window_aug[1],window_aug[3]),(255,0,0),5)
                cv2.line(img_seg_aug,(window_aug[0],window_aug[3]),(window_aug[1],window_aug[3]),(255,0,0),5)
                
                joints2d=datasetloader.utils.projectAlljoints2imagePlane(joints3d_aug)
                datasetloader.utils.circle2DJoints2Image(img_seg_aug,joints2d)
                plt.imshow(img_seg_aug)
                
                jio
                '''
                
        
    #save label
    with open(savepath_dict,'wb') as f:
        pickle.dump(dataset_dict,f,pickle.HIGHEST_PROTOCOL)
                 
    #load label
    with open(savepath_dict,'rb') as f:
        dataset_load=pickle.load(f)
     
    
     
    
    
    
    
    
    
    