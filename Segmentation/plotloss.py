import time
import matplotlib.pyplot as plt

import sys

import numpy as np

from datetime import datetime
import os
import pickle

class Plotloss():
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

    def append_local(self,loss_,nm):
        self.loss_local[nm].append(loss_)
        '''        
        for i,nm in enumerate(self.loss_names):
            self.loss_local[nm].append(loss_[i])
        '''    
        
    def append_loss(self):
        for nm in self.loss_names:
            self.losses[nm].append(np.mean(self.loss_local[nm]))
        self.initialize_local()

      
    def get_average(self,nm):
        return np.mean(self.loss_local[nm])
    
    def get_std(self,nm):
        return np.std(self.loss_local[nm])
        
    def save_plot(self,plt_title,fname):
        plt.figure()
        
        min_list=[]
        max_list=[]
        for nm in self.loss_names:
            vals=[x for x in self.losses[nm]]
            plt.plot(vals,label=nm)
    
            min_list.append(min(vals))
            max_list.append(max(vals))
        
        plt.autoscale(enable=True,axis='y')
        #plt.ylim(0,min(min_list)*100) 
        #plt.ylim(0,0.01) 
  
        
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(plt_title)
        plt.legend()
        plt.grid()
        fig=plt.gcf()
        #plt.show()
        fig.savefig(fname)
        plt.close()
        '''
        plt.figure()
        
        for nm in self.loss_names:
            plt.plot([x for x in self.losses[nm]],label=nm)
        
        plt.autoscale(enable=True,axis='y')
        
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(plt_title)
        plt.legend()
        plt.grid()
        fig=plt.gcf()
        #plt.show()
        fig.savefig(fname)
        plt.close()
        '''