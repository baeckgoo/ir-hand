import time
import matplotlib.pyplot as plt

import sys

import numpy as np

from datetime import datetime
import os
import pickle

class Progress():
    def __init__(self,loss_names,pretrain=False):
        self.loss_names=loss_names
        if pretrain==True:
            historyFile='/home/yong/hdd/HPE/output/2019_10_26_20_54/history.pickle'
            self.losses={}
            with open(historyFile, 'rb') as f:
                for nm in loss_names:
                    self.losses[nm]=pickle.load(f)

            self.loss_best=999999         
        else:
            self.losses={}
            for nm in loss_names:
                self.losses[nm]=[]
            self.loss_best=999999

        self.initialize_local()
  
    def initialize_local(self):
        #initialize local
        self.loss_local={}
        for nm in self.loss_names:
            self.loss_local[nm]=[]

        
    #the input order should be considered.
    def append_local(self,loss_):
        for i,nm in enumerate(self.loss_names):
            self.loss_local[nm].append(loss_[i])
            
        
    def append_loss(self):
        for nm in self.loss_names:
            self.losses[nm].append(np.mean(self.loss_local[nm]))
        self.initialize_local()

      
    def get_average(self):
        out=[]
        for nm in self.loss_names:
            out.append(np.mean(self.loss_local[nm]))
        return out

        
    def save_plot(self,savefolder,opt,plt_title,name):
        plt.figure()
        if opt=='hig':
            plt.plot([x for x in self.losses['hig_L1']],label='hig_L1')
            plt.plot([x for x in self.losses['hig_gan']],label='hig_gan')
            plt.plot([x for x in self.losses['hig_hpe1_mse']],label='hig_hpe1_mse') 
            if 'hig_hpe1_gan' in self.losses:
                plt.plot([x for x in self.losses['hig_hpe1_gan']],label='hig_hpe1_gan')
            if 'hig_hpe1_inter' in self.losses:
                plt.plot([x for x in self.losses['hig_hpe1_inter']],label='hig_hpe1_inter')
        elif opt=='hid':
            plt.plot([x for x in self.losses['hid_real']],label='hid_real')
            plt.plot([x for x in self.losses['hid_fake']],label='hid_fake')
        elif opt=='hpe2':
            plt.plot([x for x in self.losses['hpe2_mse']],label='hpe2_mse')
            if 'hpe2_gan' in self.losses:
                plt.plot([x for x in self.losses['hpe2_gan']],label='hpe2_gan')
            if 'hpe2_inter' in self.losses:
                plt.plot([x for x in self.losses['hpe2_inter']],label='hpe2_inter')
        elif opt=='hpd':
            plt.plot([x for x in self.losses['hpd_real']],label='hpd_real')
            plt.plot([x for x in self.losses['hpd_fake']],label='hpd_fake')
        elif opt=='hpe1_refine':
            plt.plot([x for x in self.losses['hpe1_refine_mse']],label='hpe1_refine_mse')
            if 'hpe1_refine_gan' in self.losses:
                plt.plot([x for x in self.losses['hpe1_refine_gan']],label='hpe1_refine_gan')
            if 'hpe1_refine_inter' in self.losses:
                plt.plot([x for x in self.losses['hpe1_refine_inter']],label='hpe1_refine_inter')

    
        plt.autoscale(enable=True,axis='y')
        
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(plt_title)
        plt.legend()
        plt.grid()
        fig=plt.gcf()
        #plt.show()
        fig.savefig(savefolder+name)
        plt.close()
        
        
        
        
        
        
    


    
    
    
    
    
    
    