from datasetLoader_icvl import DatasetLoader
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from numpy.testing import assert_array_almost_equal

from tqdm import tqdm
from tqdm import trange
import pickle

if __name__=="__main__":   
#def main():
    outDim=52 #52
    datanum=957032 #957032#119629
    augmentation_enable=True
    augmentation_num=2
    
    if augmentation_enable==False:
        savefile_path='/home/yong/ssd/dataset/HANDS17/training/preprocessed/pca'+'_%d_by_%d'%(outDim,datanum)+'.pickle'
    else:
        savefile_path='/home/yong/ssd/dataset/HANDS17/training/preprocessed/pca'+'_%d_by_%d'%(outDim,datanum)+'_augmented_'+'X%d'%augmentation_num+'.pickle'
    
    if not 'datasetloader' in locals():
        datasetloader=DatasetLoader(outDim)   
        filepath={}
        filepath['pca']=None#"/home/yong/ssd/dataset/HANDS17/training/preprocessed/pca_45.pickle"
        filepath['gt_train']=None
        filepath['gt_test']=None
        filepath['gt_full_train']="/home/yong/ssd/dataset/HANDS17/training/original/Training_Annotation.txt"
        filepath['gt_full_test']="/home/yong/ssd/dataset/HANDS17/training/original/Test_Annotation.txt"
        
        filepath['image_train']=None
        filepath['image_train_each']=None
        filepath['image_test']=None
        filepath['image_test_each']=None
        
        datasetloader.load(filepath)
            
    jointNum=datasetloader.jointNum
    cube=datasetloader.cube        

    joints3d_norm_bag=[]
    progressbar=trange(datanum,leave=True)
    for frame in progressbar:
        img=datasetloader.loadImage("/home/yong/ssd/dataset/HANDS17/training/original/images_train/",frame)
    
        joints3d_gt=datasetloader.joint3d_gt_train[frame,:]
        img_seg,isvalid_bool=datasetloader.segmentHand(img,joints3d_gt)
        if isvalid_bool==False:
            continue
        
        com=datasetloader.utils.calculateCOM(img_seg)
        com,img_crop,window=datasetloader.utils.refineCOMIterative(img_seg,com,3)    
        
        ##--original--##
        com3d=datasetloader.utils.unproject2Dto3D(com)    
        joints3d_norm=joints3d_gt-np.tile(com3d,(1,jointNum))
        joints3d_norm2=joints3d_norm/(cube[2]/2.)
        joints3d_norm_bag.append(np.clip(joints3d_norm2,-1,1))  
        
        ##--augmentation--##
        if augmentation_enable==False:
            continue
        for i in range(augmentation_num):
            sigma_com = 5.
            off=datasetloader.rng.randn(3)*sigma_com # mean:0, var:sigma_com (normal distrib.)
            '''     
            if i%2==0:
                rot_range=180.
                rot=datasetloader.rng.uniform(0,rot_range)
            else:
                rot_range=-180
                rot=datasetloader.rng.uniform(rot_range,0)
            '''
            rot_range=180
            rot=datasetloader.rng.uniform(-rot_range,rot_range)
            rot=np.mod(rot,360)

            sigma_sc = 0.02
            sc = np.fabs(datasetloader.rng.randn(1) * sigma_sc + 1.)
        
            
            #rotation
            joints3d_=np.reshape(joints3d_gt,(21,3))
            joints2d=datasetloader.utils.projectAlljoints2imagePlane(joints3d_)
            
            data2d=np.zeros_like(joints2d)
            for k in range(data2d.shape[0]):
                data2d[k]= datasetloader.utils.rotatePoint2D(joints2d[k],com,rot)
               
            data3d=np.zeros_like(data2d)
            for k in range(data2d.shape[0]):
                data3d[k]=datasetloader.utils.unproject2Dto3D(data2d[k])
            
            #translation
            com3d_new=com3d+off
        
            #scale
            new_cube = [s*sc for s in cube]
            
            #apply augmentation
            joints3d_norm=np.concatenate(data3d)-np.tile(com3d_new,(1,jointNum))
            joints3d_norm2=joints3d_norm/(new_cube[2]/2.)
            joints3d_norm3=np.clip(joints3d_norm2,-1,1)
            joints3d_norm_bag.append(joints3d_norm3)
            
            
            #show augmented image and joints position
            '''
            img_seg,joints3d=datasetloader.utils.augmentRotation(img_seg,joints3d_gt,com,rot)
            com2d,com3d,window=datasetloader.utils.augmentTranslation(img_seg,joints3d,com,off)
            cube=datasetloader.cube
            cube,window=datasetloader.utils.augmentScale(img_seg,cube,com2d,sc)
            xstart,xend,ystart,yend,zstart,zend=window[0],window[1],window[2],window[3],window[4],window[5]
            img_crop=datasetloader.utils.crop(img_seg,xstart,xend,ystart,yend,zstart,zend)
            img_train=datasetloader.utils.makeLearningImage(img_crop,frame)
        
            joints2d=datasetloader.utils.projectAlljoints2imagePlane(data3d)
            datasetloader.utils.circle2DJoints2Image(img_seg,joints2d)
            plt.imshow(img_seg)
            jio
            '''
            
            

    joints3d_norm_bag2=np.asarray(joints3d_norm_bag)
    joints3d_norm4=joints3d_norm_bag2[:,0,:]
    
    pca=PCA(n_components=outDim)
    pca.fit(joints3d_norm4)


    with open(savefile_path,'wb') as f:
        pickle.dump(pca,f,pickle.HIGHEST_PROTOCOL)

       
    with open(savefile_path,'rb') as f:
        pca_load=pickle.load(f)
        
    print(pca_load.mean_-pca.mean_)    
    print(pca_load.components_-pca.components_)
    
#if __name__=="__main__":   
def test():   
    if not 'datasetloader' in locals():
        datasetloader=DatasetLoader(45)   
        filepath={}
        filepath['pca']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/pca_45_by_119629_augmented.pickle"
        filepath['gt_train']=None
        filepath['gt_test']=None
        filepath['gt_full_train']="/home/yong/ssd/dataset/HANDS17/training/original/Training_Annotation.txt"
        filepath['gt_full_test']="/home/yong/ssd/dataset/HANDS17/training/original/Test_Annotation.txt"
            
        filepath['image_train']=None
        filepath['image_train_each']=None
        filepath['image_test']=None
        filepath['image_test_each']=None
                
        datasetloader.load(filepath)
        
    jointNum=21
    cube=datasetloader.cube
        
    pca=datasetloader.pca    
    imagepath="/home/yong/ssd/dataset/HANDS17/training/original/images_train/image_D%08d.png"
    
    #make train image
    frame=13
    img=cv2.imread(imagepath%(frame+1),2)
    joints3d_gt=datasetloader.joint3d_gt_train[frame,:]
    
    img_seg,isvalid_bool=datasetloader.segmentHand(img,joints3d_gt)
    com=datasetloader.utils.calculateCOM(img_seg)
    print('com_Before',com)
    com,img_crop,window=datasetloader.utils.refineCOMIterative(img_seg,com,1)
        
    #make train label
    com3d=datasetloader.utils.unproject2Dto3D(com)   
    joints3d_norm=joints3d_gt-np.tile(com3d,(1,jointNum))
    joints3d_norm2=joints3d_norm/(cube[2]/2.)
    joints3d_norm3=np.clip(joints3d_norm2,-1,1)  
    trainlabel_embed=datasetloader.pca.transform(joints3d_norm3) 
    
    pca_w=pca.components_
    pca_b=pca.mean_
    output_recon=np.dot(trainlabel_embed,pca_w)+pca_b
        
    #dif
    dif=joints3d_norm3-output_recon
    
    #exer
    X_train = np.random.randn(100, 63)
    
    pca = PCA(n_components=45)
    pca.fit(X_train)

    pca_w=pca.components_
    pca_b=pca.mean_
    
    a=pca.fit_transform(X_train)
    
    a_recon=pca.inverse_transform(a)
    dif=a_recon-X_train
    print(dif.max(),dif.min())
    
    
    #
    pca_w=pca.components_
    pca_b=pca.mean_
    output_recon=np.dot(a,pca_w)+pca_b
    dif=output_recon-X_train
    print(dif.max(),dif.min())
    

    #U, S, VT = np.linalg.svd(X_train - X_train.mean(0))

    #dif=VT[:30], pca.components_
    #assert_array_almost_equal(VT[:30], pca.components_)
    
    
    joints2d=datasetloader.utils.projectAlljoints2imagePlane(np.reshape(joints3d_gt,(21,3)))
    datasetloader.utils.circle2DJoints2Image(img_seg,joints2d)
    plt.imshow(img_seg)   
    
    img_window=datasetloader.showBox(img,window)
    plt.imshow(img_window)
    print(com)
    