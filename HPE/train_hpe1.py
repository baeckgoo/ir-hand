import torch
import sys
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle

sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/datasetloader/')
#from datasetloader.datasetLoader_icvl import DatasetLoader
from datasetLoader_icvl import DatasetLoader

from model.deepPrior import dpnet
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
import loss as losses

from utils.evaluation import evaluateEmbeddingError

import torch.backends.cudnn as cudnn

from tqdm import tqdm
from tqdm import trange
import time

from datetime import datetime

'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
'''

# select proper device to run
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    """
    parser.add_argument('--resnet-layers', default=101, type=int, metavar='N',   #50
                        help='Number of resnet layers',
                        choices=[18, 34, 50, 101, 152])
    """
    
    # training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='adam',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=5.0e-5, type=float,#1.0e-4(too high), 5.0e-6()
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default= 1.0e-5, type=float, #1.0e-5
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50,80],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')

    # data preprocessing
    
    
    #checkpoint
    
    return parser.parse_args()

 
def decrease_learning_rate(optimizer,lr,gamma):
    lr *= gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        
def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr
  
def train(model,datasetloader,criterion_loss,lr,args):
    batch_size=args.train_batch
    aug_ratio=1
    
    generator_train=datasetloader.generator_learningData(batch_size,'train','noaug',aug_ratio) 
    iter_num_train=aug_ratio*len(datasetloader.idx_train)//batch_size
    loss_train=[]
    
    progressbar=trange(iter_num_train,leave=True)    
    for i in progressbar:
    #for i in range(iter_num_train):
        #TIME=time.time()
        img_train,gt_train,cnn_gt_joint3d,com3d=next(generator_train)   #0.6sec
        #print(time.time()-TIME)
        
        input=torch.FloatTensor(img_train)
        input=input.to(device)
        target=torch.FloatTensor(gt_train)
        target=target.to(device)
        
        #compute output                  
        output=model(input)  #0.005
        
        #loss
        loss=criterion_loss(output,target)
        loss_train.append(loss.item()) #check if the list size becomes too big.
        
        #error   
        #err=evaluateEmbeddingError(output,target)
        #error_train.append(err.item())
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i<iter_num_train-1:
            message='epoch:%d lr:%.5f loss:%.5f'%(epoch,lr,loss.item())
        else:
            message='epoch(avg):%d lr:%.5f loss:%.5f'%(epoch,lr,np.mean(loss_train))
        
        progressbar.set_description(message)
        
    return np.mean(loss_train)
 
def validate(model,datasetloader,criterion_loss,args):
    batch_size=args.test_batch
    
    with torch.no_grad():
        generator_validate=datasetloader.generator_learningData(batch_size,'validate','noaug',0)
        iter_num_validate=len(datasetloader.idx_validate)//batch_size
            
        loss_validate=[]
        
        progressbar=trange(iter_num_validate,leave=True)
        for i in progressbar:
        #for i in range(iter_num_validate):
            img_validate,gt_validate,cnn_gt_joint3d,com3d=next(generator_validate)   
            input=torch.FloatTensor(img_validate)
            input=input.to(device)
            
            target=torch.FloatTensor(gt_validate)
            target=target.to(device)
            
            #compute output                     
            output=model(input)
            loss = criterion_loss(output, target)
                
            #store loss/accuracy 
            #err=evaluateEmbeddingError(output,target)
            #error_validate.append(err.item())
            loss_validate.append(loss.item()) #caution if the size becomes too big.

            if i<iter_num_validate-1:
                message='loss:%.5f'%(loss.item())
            else:
                message='loss(avg):%.5f'%(np.mean(loss_validate))
            
            progressbar.set_description(message)
            
    return np.mean(loss_validate)
  
def validate_error(model,datasetloader,criterion_error,args):
    batch_size=args.test_batch
    with torch.no_grad():
        generator_validate=datasetloader.generator_learningData(batch_size,'test_error','noaug',0)
        iter_num_validate=len(datasetloader.idx_validate)//batch_size
            
        error_bag=[]
        
        progressbar=trange(iter_num_validate,leave=True)
        for i in progressbar:
            img_validate,_,cnn_gt_joint3d,com3d=next(generator_validate)   
            input=torch.FloatTensor(img_validate)
            input=input.to(device)
            
            target=torch.FloatTensor(cnn_gt_joint3d)
            target=target.to(device)
            
            #compute output                     
            output_embed=model(input)
                       
            #reconstruct
            error_batch=criterion_error._forward(output_embed,target,com3d)
            error_bag.append(error_batch)
            #error_validate.append(err.item())
            
            if i<iter_num_validate-1:
                message='--error:%.5f std:%.f '%(np.mean(error_batch),np.std(error_batch))
            else:
                message='--error(avg):%.5f std:%.5f'%(np.mean(error_bag),np.std(error_bag))
            
            progressbar.set_description(message)
              
    return np.mean(error_bag)            
        
        
        
        
    
    
#frane 24 is strange? compressed error?
if __name__=="__main__":
    print('train..')
    ###--setting--###
    pretrain=False
    trained_modelFile='/home/yong/ssd/HPE/output/2019_10_26_20_54/trainedModel_epoch33.pth.tar'
    historyFile='/home/yong/ssd/HPE/output/2019_10_26_20_54/history.pickle'
    #trained_modelFile='../output/2019_10_1_21_33/trainedModel_epoch2.pth'
    args=arguments()  

    ###---create result folder---###
    now=datetime.now()
    savefolder='/home/yong/hdd/HPE/output/%d_%d_%d_%d_%d/'%(now.year,now.month,now.day,now.hour,now.minute)
    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)
    
     
    ###--dataset--###
    if not 'datasetloader' in locals():
        
        datasetloader=DatasetLoader(args.num_classes)
        filepath={}
        filepath['pca']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/pca_52_by_957032_augmented_X2.pickle"
        filepath['gt_train']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/dataset_dict_train_full_52_by_957032_augmented_X2.pickle"
        filepath['gt_test']="/home/yong/ssd/dataset/HANDS17/training/preprocessed/dataset_dict_test_full_52_by_957032_augmented_X2.pickle"
        filepath['gt_full_train']="/home/yong/ssd/dataset/HANDS17/training/original/Training_Annotation.txt"
        filepath['gt_full_test']="/home/yong/ssd/dataset/HANDS17/training/original/Test_Annotation.txt"
            
        filepath['image_train']=None
        filepath['image_train_each']="/home/yong/ssd/dataset/preprocessed_HPE/images_train_augmented_X2/%d.npy"
        filepath['image_test']=None
        filepath['image_test_each']="/home/yong/ssd/dataset/preprocessed_HPE/images_test/%d.npy"
        
        datasetloader.load(filepath)
        

    ###--model--###
    model = dpnet(args.num_classes,args.num_blocks,device)
    model=model.to(device)
    #model = torch.nn.DataParallel(model).to(device)
    
    ###--optimizer--###
    '''
    if args.solver == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    '''
    if args.solver =='adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay,
                                        betas=(0.5,0.999))
        
    ###---loss/error---###
    criterion_loss= losses.JointsMSELoss().to(device)
    
    pca_w=torch.FloatTensor(datasetloader.pca.components_)
    pca_w=pca_w.to(device)
    pca_b=torch.FloatTensor(datasetloader.pca.mean_)
    pca_b=pca_b.to(device)
    
    cube=datasetloader.cube  
    cube_torch=torch.FloatTensor(cube)
    cube_torch=cube_torch.to(device)
    
    criterion_error=losses.Accuracy(pca_w,pca_b,cube_torch,datasetloader.jointNum,args.test_batch,device).to(device)
         
    ###---resume--###
    if pretrain==True:
        checkpoint=torch.load(trained_modelFile)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])  
        
        #model.load_state_dict(checkpoint)
        
    ###---progress---###
    lr = args.lr
    if pretrain==True:
        progress={}
        with open(historyFile, 'rb') as f:
            progress['loss_train']=pickle.load(f)
            progress['loss_validate']=pickle.load(f)
            progress['error_validate']=pickle.load(f)
        loss_validate_best=999999         
    else:
        progress={}
        progress['loss_train']=[]
        progress['loss_validate']=[]
        progress['error_validate']=[]
        loss_validate_best=999999   

    ###---run---###
    framenum_not_improved=0
    for epoch in range(args.start_epoch,args.epochs): 
        #lr = adjust_learning_rate(optimizer, epoch,lr, args.schedule, args.gamma)
        
        #train loss
        
        model.train()
        loss_train=train(model,datasetloader,criterion_loss,lr,args)
        progress['loss_train'].append(loss_train)
        
        #validate loss   
        
        model.eval()
        loss_validate=validate(model,datasetloader,criterion_loss,args)
        progress['loss_validate'].append(loss_validate)
        
        #validate accuracy
        if epoch%5==0:
            error_validate=validate_error(model,datasetloader,criterion_error,args)
            progress['error_validate'].append(error_validate)
            
        #if better, save model
        if np.mean(loss_validate)<loss_validate_best:
            loss_validate_best=np.mean(loss_validate)
            state_best={
                'epoch':epoch+1,
                'state_dict':model.state_dict(),
                'best_loss':loss_validate_best,
                'optimizer':optimizer.state_dict()}  
            torch.save(state_best,savefolder+'trainedModel_epoch%d.pth.tar'%state_best['epoch'])
        else:
            framenum_not_improved+=1
            
        if framenum_not_improved>4:
            lr = decrease_learning_rate(optimizer, lr, args.gamma)
            framenum_not_improved=0
        
        
    ###--save configuration--###
    f=open(savefolder+'setting.txt','w')
    
    f.write('initial learning rate: %f\n'%args.lr)
    f.write('final   learning rate: %f\n'%lr)
    f.write('weight decay: %f\n'%args.weight_decay)
    f.write('train batch:%d\n'%args.train_batch)
    f.write('optimizer:%s\n'%optimizer)
    f.write('start epoch:%d\n'%args.start_epoch)
    f.write('---filepath---\n')
    f.write('pca:%s\n'%filepath['pca'])
    f.write('gt_train:%s\n'%filepath['gt_train'])
    f.write('gt_test:%s\n'%filepath['gt_test'])
    f.write('gt_full_train:%s\n'%filepath['gt_full_train'])
    f.write('gt_full_test:%s\n'%filepath['gt_full_test'])
    f.write('image_train:%s\n'%filepath['image_train'])
    f.write('image_train_each:%s\n'%filepath['image_train_each'])
    f.write('image_test:%s\n'%filepath['image_test'])
    f.write('image_test_each:%s\n'%filepath['image_test_each'])
    f.write('train data num%s\n'%len(datasetloader.idx_train))
    f.write('test data num%s\n'%len(datasetloader.idx_validate))
    for arg in vars(args):
        f.write('%s:%s\n'%(arg,getattr(args,arg)))
    f.close()   
    
    ###--save loss--###
    f=open(savefolder+'train_loss.txt','w')
    for i in range(len(progress['loss_train'])):
        f.write('%d: %f\n'%(i,progress['loss_train'][i]))
    f.close()
    
    f=open(savefolder+'val_loss.txt','w')
    for i in range(len(progress['loss_validate'])):
        f.write('%d: %f\n'%(i,progress['loss_validate'][i]))
    f.close()
    
    f=open(savefolder+'val_error.txt','w')
    for i in range(len(progress['error_validate'])):
        f.write('%d: %f\n'%(i,progress['error_validate'][i]))
    f.close()
        
    ###--save progress---###
    history_filepath=savefolder+'history.pickle'
    with open(history_filepath,'wb') as f:
        pickle.dump(progress['loss_train'],f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(progress['loss_validate'],f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(progress['error_validate'],f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(lr,f,pickle.HIGHEST_PROTOCOL)
       
    
    ###--show/save loss plot--###
    plt.plot([x for x in progress['loss_train']],label='train')
    plt.plot([x for x in progress['loss_validate']],label='val')
    #plt.ylim(0,0.08)
    plt.ylim(0,0.02)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train/val loss')
    plt.legend()
    plt.grid()
    fig=plt.gcf()
    plt.show()
    fig.savefig(savefolder+'loss.png')
    
                    