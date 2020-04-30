import numpy as np
#import cv2
#import glob
from scipy.linalg import eigh
#from types import *

class MyPCA:
    def __init__ (self,n_components = 1.0) :
        self.n_components = n_components
        
    def fit(self, data):
        self.data = np.asarray(data)
        self.n_samples, self.n_features = self.data.shape
        std_dev = np.zeros(np.shape(self.data))
        for i in range(self.n_features):
            std_dev[:,i] =  (self.data[:,i] - np.mean(self.data[:,i]))/self.data[:,i].std()
        cov = np.cov(std_dev.T)# Eigen Values
        self.EigVal, self.EigVec = eigh(cov)
        order = self.EigVal.argsort()[::-1]
        self.EigVal = self.EigVal[order]
        self.EigVec = self.EigVec[:,order]
        if isinstance(self.n_components, float):
            #print(self.n_components)
            if self.n_components <= 1.0 :
                percent=0.0
                count=0
                for i in self.EigVal:
                    percent+= i/sum(self.EigVal)
                    count+=1
                    if percent >= self.n_components:
                        break;
                self.n_components = count
                #print(count)
                #print((np.dot(std_dev, self.EigVec[:,:count])).shape)
                return (np.dot(std_dev, self.EigVec[:,:count]))
            else :
                print("Float value must between 0.0 and 1.0")
                return
        elif isinstance(self.n_components, int):
            if(self.n_components <= self.EigVec.shape[0]) :
                #print((np.dot(std_dev, self.EigVec)).shape)
                return (np.dot(std_dev, self.EigVec[:,:self.n_components]))
            else :
                print("Number of componets greater than feature size")
                return
        else :
            print("Numeber of components must be float or int")
            return
        
    def transform(self, test_data):
        std_dev = np.zeros(np.shape(test_data))
        for i in range(self.n_features):
            std_dev[:,i] =  (test_data[:,i] - np.mean(self.data[:,i]))/self.data[:,i].std()
        return (np.dot(std_dev, self.EigVec[:,:self.n_components]))