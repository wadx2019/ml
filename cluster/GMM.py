# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:38:35 2020

@author: HP
"""

import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from scipy import stats

class GMM(object):
    
    def __init__(self,n_classes,n_features,threshold=1e-5,epsilon=0,max_epoches=1000,
                 params=None,verbose=False,init_method="kmeans++"):
        
        self.n_classes=n_classes
        self.n_features=n_features
        self.threshold=threshold
        self.epsilon=epsilon
        self.max_epoches=max_epoches
        
        if params:
            self.centers_init=params['centers_init']
            self.sigmas_init=params['sigmas_init']
            self.pis_init=params['pis_init']
        else:
            self.centers_init=None
            self.sigmas_init=None
            self.pis_init=None
            
        self.verbose=verbose
        self.init_method=init_method
        self.centers_his=[]
        self.sigmas_his=[]
        self.pis_his=[]
        
    def train(self,X):
        
        assert X.shape[0]>=self.n_classes, "number of samples is less than number of classes!"
        assert X.shape[1]==self.n_features, "dimension of features not match!"
        
        self.init_params(X)
        
        assert self.centers_init.shape==(self.n_classes,self.n_features) , "centers shape error!"
        assert self.sigmas_init.shape==(self.n_classes,self.n_features,self.n_features) , "sigmas shape error!"
        assert self.pis_init.shape==(self.n_classes,) , "pis shape error!"
        
        if self.verbose:
            self.centers_his.append(self.centers_init.copy())
            self.sigmas_his.append(self.sigmas_init.copy())
            self.pis_his.append(self.pis_init.copy())
                
        for epoch in range(self.max_epoches):
            
            probs=np.zeros((X.shape[0],self.n_classes))
            for i in range(self.n_classes):
                probs[:,i]=self.pis_init[i]*stats.multivariate_normal(mean=self.centers_init[i],cov=self.sigmas_init[i],allow_singular=True).pdf(X)
            gammas=probs/np.sum(probs,axis=1).reshape(-1,1)
              
            old_centers=self.centers_init.copy()
            old_sigmas=self.sigmas_init.copy()
            old_pis=self.pis_init.copy()
            
            N_gamma=np.sum(gammas,axis=0)
            X_norm=(X.reshape(-1,1,self.n_features)-self.centers_init).transpose((1,0,2))
            self.centers_init=np.sum(gammas.T.reshape((self.n_classes,-1,1))*X,axis=1)/N_gamma.reshape((-1,1))
            self.sigmas_init=np.matmul((gammas.T.reshape(self.n_classes,-1,1)*X_norm).transpose((0,2,1)),X_norm)/N_gamma.reshape((-1,1,1))+self.epsilon
            self.pis_init=N_gamma/X.shape[0]
            
            loss=np.sum(np.abs(old_centers-self.centers_init))/self.n_classes+np.sum(np.abs(old_sigmas-self.sigmas_init))/self.n_classes**2+np.sum(np.abs(old_pis-self.pis_init))/self.n_classes
            
            if self.verbose:
                self.centers_his.append(self.centers_init.copy())
                self.sigmas_his.append(self.sigmas_init.copy())
                self.pis_his.append(self.pis_init.copy())
            
            if loss<self.threshold:
                break
        return epoch

    def init_params(self,X):
        
        if self.centers_init!=None or self.sigmas_init!=None or self.pis_init!=None or self.init_method=="random":
            if self.centers_init == None:
                index=np.arange(X.shape[0])
                np.random.shuffle(index)
                index=index[:self.n_classes]
                self.centers_init=X[index].copy()
            
            if self.sigmas_init == None and self.pis_init == None:
                dis2center=np.linalg.norm(X.reshape(-1,1,self.n_features)-self.centers_init,axis=2)
                gammas=dis2center/np.sum(dis2center,axis=1).reshape(-1,1)
                N_gamma=np.sum(gammas,axis=0)
                X_norm=(X.reshape(-1,1,self.n_features)-self.centers_init).transpose((1,0,2))
                self.sigmas_init=np.matmul((gammas.T.reshape(self.n_classes,-1,1)*X_norm).transpose((0,2,1)),X_norm)/N_gamma.reshape((-1,1,1))+self.epsilon
                self.pis_init=N_gamma/X.shape[0]
            
        elif self.init_method=="kmeans++":
            self.centers_init=[]
            for i in range(self.n_classes):
                if self.centers_init:
                    centers_tmp=np.array(self.centers_init)
                    dis=np.min(np.linalg.norm(X.reshape((-1,1,self.n_features))-centers_tmp,axis=1),axis=1)**2
                    dis_norm=dis/np.sum(dis)
                    self.centers_init.append(X[np.random.choice(np.arange(len(dis_norm)),p=dis_norm)])
                else:
                    X_center=np.sum(X,axis=0)/X.shape[0]
                    dis=np.linalg.norm(X-X_center,axis=1)**2
                    dis_norm=dis/np.sum(dis)
                    #self.centers_init.append(X[np.random.choice(np.arange(len(dis_norm)),p=dis_norm)])
                    self.centers_init.append(X[np.random.randint(X.shape[0])])
            
            self.centers_init=np.array(self.centers_init)
            dis2center=np.linalg.norm(X.reshape(-1,1,self.n_features)-self.centers_init,axis=2)
            gammas=dis2center/np.sum(dis2center,axis=1).reshape(-1,1)
            N_gamma=np.sum(gammas,axis=0)
            X_norm=(X.reshape(-1,1,self.n_features)-self.centers_init).transpose((1,0,2))
            self.sigmas_init=np.matmul((gammas.T.reshape(self.n_classes,-1,1)*X_norm).transpose((0,2,1)),X_norm)/N_gamma.reshape((-1,1,1))+self.epsilon
            self.pis_init=N_gamma/X.shape[0]
        else:
            raise Exception("init_method is illegal!")
            
            
    
    def predict(self,X,prob=False):
        
        assert self.centers_init.shape==(self.n_classes,self.n_features) , "centers shape error!"
        assert self.sigmas_init.shape==(self.n_classes,self.n_features,self.n_features) , "sigmas shape error!"
        assert self.pis_init.shape==(self.n_classes,) , "pis shape error!"
        
        gammas=np.zeros((X.shape[0],self.n_classes))
        probs=np.zeros(gammas.shape)
        for i in range(self.n_classes):
            probs[:,i]=self.pis_init[i]*stats.multivariate_normal(mean=self.centers_init[i],cov=self.sigmas_init[i],allow_singular=True).pdf(X)
        gammas=probs/np.sum(probs,axis=1).reshape(-1,1)
        
        if prob:
            return gammas
        else:
            return np.argmax(gammas,axis=1)
    
    def train_predict(self,X,prob=False):
        
        self.train(X)
        return self.predict(X,prob)
    
    def intermediate_results(self):
        
        GMM_his=[]
        for i in range(len(self.centers_his)):
            GMM_his.append(GMM(self.n_classes,self.n_features,
                   params={"centers_init":self.centers_his[i],"sigmas_init":self.sigmas_his[i],"pis_init":self.pis_his[i]},
                   threshold=self.threshold,epsilon=self.epsilon,max_epoches=self.max_epoches,verbose=self.verbose))
        
        return GMM_his if GMM_his else None
    
    def params(self):
        
        return {"centers_init":self.centers_init,"sigmas_init":self.sigmas_init,"pis_init":self.pis_init}
    
    def copy(self):
        
        return GMM(self.n_classes,self.n_features,
                   params={"centers_init":self.centers_init,"sigmas_init":self.sigmas_init,"pis_init":self.pis_init},
                   threshold=self.threshold,epsilon=self.epsilon,max_epoches=self.max_epoches,verbose=self.verbose)
    
    def validate(self,X,times=1,error_threshold=1e-2):
        
        mathced=0
        for i in range(times):
            val=GMM(self.n_classes,self.n_features,
                   threshold=self.threshold,epsilon=self.epsilon,max_epoches=self.max_epoches)
            val.train(X)
            
            error=np.sum(np.abs(val.centers_init-self.centers_init))+np.sum(np.abs(val.sigmas_init-self.sigmas_init))+np.sum(np.abs(val.pis_init-self.pis_init))
            
            if error<error_threshold:
                mathced+=1
        
        return mathced/times
    
        
        
if __name__=="__main__":
    X,y=make_classification(n_samples=100000,n_features=3,n_informative=2,n_redundant=0,n_repeated=0,n_clusters_per_class=1,n_classes=3,)
    plt.figure(1)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()
    model=GMM(n_classes=3,n_features=3,verbose=True,init_method='kmeans++')
    y_hat=model.train_predict(X)
    plt.figure(2)
    plt.scatter(X[:,0],X[:,1],c=y_hat)
    plt.show()