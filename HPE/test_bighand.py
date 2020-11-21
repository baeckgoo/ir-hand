import torch
import sys
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np

#from datasetloader.datasetLoader_icvl import DatasetLoader
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
from datasetLoader_icvl import DatasetLoader

from model.deepPrior import dpnet

from utils.evaluation import evaluateEmbeddingError

import torch.backends.cudnn as cudnn

from tqdm import tqdm
from tqdm import trange
import time
import pickle

# select proper device to run
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33


def arguments():
    parser = argparse.ArgumentParser(description='PyTorch hand pose Training')
    
    #dataset
    parser.add_argument('--num-classes', default=52, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--num-blocks', default=5, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    #model structure
    """
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: hg)')
    """

    # training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=1.0e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30,60,90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    
    # data preprocessing
    
    #checkpoint
    
    return parser.parse_args()
 
#frane 24 is strange? compressed error?
if __name__=="__main__":
    option="test"
    batch_size=128
    
    ###---select history---###    
    
    trained_modelFile='/home/yong/hdd/HPE/output/2020_2_26_11_29/trainedModel_epoch99.pth.tar'
    savefolder='/home/yong/hdd/HPE/output/2020_2_26_11_29/result_'+'%s'%option+'/'
    

    save_resultimage=False
    
    notValidFileList=[]
    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)
    
    ###--setting--###
    args=arguments()  
    
    ###--dataset--###
    datasetloader=DatasetLoader(args.num_classes)
    filepath={}
    filepath['pca']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/pca_52_by_957032_augmented_X2.pickle"
    filepath['gt_train']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/dataset_dict_train_full_52_by_957032_augmented_X2.pickle"
    filepath['gt_test']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/dataset_dict_test_full_52_by_957032_augmented_X2.pickle"
    filepath['gt_full_train']="/home/yong/ssd/dataset/HANDS17/training/original/Training_Annotation.txt"
    filepath['gt_full_test']="/home/yong/ssd/dataset/HANDS17/training/original/Test_Annotation.txt"
            
    filepath['image_train']=None#"/home/yong/hdd/dataset/HANDS17/training/preprocessed/images_train_eighth.npy"
    filepath['image_train_each']=None#"../../../preprocessed_HPE/images_train/%d.npy"
    filepath['image_test']=None
    filepath['image_test_each']="/home/yong/ssd/dataset/preprocessed_HPE/images_test/%d.npy"
        
    datasetloader.load(filepath)
    
    
    trainImageSize=datasetloader.trainImageSize
    jointNum=datasetloader.jointNum
    num_class=datasetloader.num_class
    #img_gt = np.zeros((1,1,trainImageSize,trainImageSize), dtype = np.float32)
    joint_gt=np.zeros((1,jointNum))

    frame_start=0
    #frame_end=#119629#295510
    if option=="train":
        frame_end=datasetloader.traindataNum
    elif option=="test":
        frame_end=datasetloader.validateNum
    
    ###--test--###
    if not 'model' in locals():
        model=dpnet(args.num_classes,args.num_blocks,device)
        model=model.to(device)
        checkpoint=torch.load(trained_modelFile)
        model.load_state_dict(checkpoint['state_dict'])
        #model.load_state_dict(torch.load(trained_modelFile))
        model.eval()
    
    error_bag=[]
    error_avg=0
    validframe_num=frame_end-frame_start
    cube=datasetloader.cube  
    cube_torch=torch.FloatTensor(cube)
    cube_torch=cube_torch.to(device)
    
    '''
    cnn_img = np.zeros((batch_size,1,trainImageSize,trainImageSize), dtype = np.float32)
    cnn_gt=np.zeros((batch_size,datasetloader.num_class),np.float32)
    joints3d_gt=np.zeros((batch_size,jointNum*3))
    com3d=np.zeros((batch_size,3))
    '''
    
    output_recon_bag=[]
    with torch.no_grad():
        progressbar=tqdm(range(frame_start,frame_end),leave=True)
        for i in progressbar:
        #for i in range(frame_start,frame_end):       
            batch_idx=i%batch_size
            if batch_idx==0:
                cnn_img_,cnn_gt_,com3d_,joints3d_gt_=[],[],[],[]
                
            if option=='train':    
                frame=datasetloader.idx_train[i]
            elif option=='test':
                frame=datasetloader.idx_validate[i]
                
            
            if option=='train':
                cnn_img_.append(np.load(datasetloader.filepath['image_train_each']%frame))
                cnn_gt_.append(np.copy(datasetloader.dataset_dict_train['label_embed'][frame]))    
                com3d_.append(np.copy(datasetloader.dataset_dict_train['com3d'][frame])) 
                joints3d_gt_.append(datasetloader.joint3d_gt_train[frame,:])
            elif option=='test':
                cnn_img_.append(np.load(datasetloader.filepath['image_test_each']%frame))
                cnn_gt_.append(np.copy(datasetloader.dataset_dict_test['label_embed'][frame]))
                com3d_.append(np.copy(datasetloader.dataset_dict_test['com3d'][frame]))
                joints3d_gt_.append(datasetloader.joint3d_gt_test[frame,:])
            
            #to torch (cuda)
            if batch_idx==batch_size-1 or frame==frame_end-1:
                #
                cnn_img=torch.FloatTensor(np.expand_dims(cnn_img_,axis=1))            
                cnn_gt=torch.FloatTensor(np.asarray(cnn_gt_)[:,0,:])               
                com3d=np.asarray(com3d_)
                joints3d_gt_torch=torch.FloatTensor(joints3d_gt_)
                
                #to network
                #joints3d_gt_torch=torch.FloatTensor(joints3d_gt)
                joints3d_gt_torch=joints3d_gt_torch.to(device)
            
                input=torch.FloatTensor(cnn_img)
                input=input.to(device)
                output_embed=model(input)

                #reconstruct
                pca_w=torch.FloatTensor(datasetloader.pca.components_)
                pca_w=pca_w.to(device)
                pca_b=torch.FloatTensor(datasetloader.pca.mean_)
                pca_b=pca_b.to(device)
                output_recon=torch.mm(output_embed,pca_w)+pca_b
            

                com3d_tile=np.tile(com3d,(1,21))
                com3d_tile=torch.FloatTensor(com3d_tile)          
                com3d_tile=  com3d_tile.to(device)                 
                output_recon=output_recon*(cube_torch[2]/2.)+com3d_tile
                
                for k in range(output_recon.shape[0]):
                    output_recon_bag.append(output_recon[k])


                for k in range(output_recon.shape[0]):
                    error_torch1=(joints3d_gt_torch[k]-output_recon[k])**2
                    error_torch2=torch.sqrt(torch.sum(error_torch1.view(21*1,3),dim=1))       
                    error=torch.sum(error_torch2)/(21*1)
                    error_bag.append(error)
        
                #error_avg+=error/validframe_num
                      
                #show/save image
                
                if save_resultimage==True:
                    for k in range(batch_size):
                        img_save=np.zeros((480,640*2,3))
                        img=datasetloader.loadImage("/home/yong/hdd/dataset/HANDS17/training/original/images_%s/"%option,(frame-batch_size+k+1))
                        
                        output_recon_cpu=output_recon.cpu().numpy()
                        
                        
                        img=np.uint8(cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))           
                        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                        
                        img_save[:,0:img.shape[1],:]=datasetloader.utils.circle3DJoints2Image(img,joints3d_gt_[k])
                        img_save[:,img.shape[1]:2*img.shape[1],:]=datasetloader.utils.circle3DJoints2Image(img,output_recon_cpu[k,:])
                        
                        
                        cv2.putText(img_save,'error:%f'%error_bag[k].item(),(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                        cv2.imwrite(savefolder+option+'_%d.png'%(frame-batch_size+k+1),img_save)
                
            if not 'progressbar' in locals():
                if frame%10000==0:
                    print('frame:',frame,'(','%.f'%(100*frame/frame_end),'%)')
            
    
        #mean and st.dev of error    
        print('error avg..',torch.mean(torch.stack(error_bag)))
        print('error std..',torch.std(torch.stack(error_bag)))
    
        #save positions
        output_recon_bag_cpu=torch.stack(output_recon_bag)
        output_recon_bag_cpu=output_recon_bag_cpu.cpu().numpy()
        
        f=open(savefolder+'../position_estimated.txt','w')
        for i in range(len(output_recon_bag_cpu)):
            s=''
            for j in range(62):
                s+='%.2f,'%output_recon_bag_cpu[i][j]
            s+='%.2f\n'%output_recon_bag_cpu[i][j+1]
            
            f.write(s)
        f.close()
        

        #save/show percentage of frames (graph)     
        error_bag_cpu=torch.stack(error_bag)
        error_bag_cpu=error_bag_cpu.cpu().numpy()

        percentage_of_frame_bag=[]
        for th in range(70):
            percentage_of_frame=100*np.count_nonzero(error_bag_cpu<th)/len(error_bag_cpu)
            percentage_of_frame_bag.append(percentage_of_frame)
        plt.grid()
        plt.xlabel('D:Max allowed distance to GT [mm]')
        plt.ylabel('percentage of frames with average error within D')
        plt.plot(percentage_of_frame_bag)
                
        fig=plt.gcf()
        fig.savefig(savefolder+'../'+'test result_avgerr.png')
        
        #save/show percenta
        position_gt=datasetloader.joint3d_gt_test
        position_estimated=output_recon_bag_cpu
        
    
        position_error=(position_gt-position_estimated)**2 #(295510,63)
        position_error=np.reshape(position_error,(len(position_error),21,3)) #(295510,21,3)
        position_error=np.sqrt(np.sum(position_error,2))  #(295510,21)
        
        pof_joints=[]
        position_error_max=np.max(position_error,1)
        for th in range(80):
            pof_joints.append(100*np.count_nonzero(position_error_max<th)/len(position_estimated))
    
        plt.grid()
        plt.xlabel('error threshold' +'  \u03B5' + ' (mm)')
        plt.ylabel('proportion of frames with all joints error < '+'\u03B5')
        plt.plot(pof_joints)
                
        fig=plt.gcf()
        fig.savefig(savefolder+'../'+'test result_wrosterror.png')  
        
        
        
        
        
        

                    