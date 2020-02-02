import numpy as np
import pandas as pd

def byfirst(elem):
    return elem[0]

class KNNClassifier:
    k = 3
    def _init_(self):
        self.train_df = pd.Dataframe()
        self.train_data = np.empty
        self.train_label = np.empty
        self.validation_df = pd.Dataframe()
        self.validation_data = np.empty
        self.validation_label = np.empty
        self.test_df = pd.Dataframe()
        self.test_data = np.empty
        self.test_label = np.empty
        
    def train(self, train_path):
        self.train_df = pd.read_csv(train_path, header=None)
        self.train_data = self.train_df.to_numpy()
        self.train_label = self.train_data[:,0]
        self.train_data = self.train_data[:,1:]
   
    def predict(self, test_path):
        self.test_df = pd.read_csv(test_path, header=None)
        self.test_data = self.test_df.to_numpy()
        euclid_success=[]
        manhatt_success=[]
        euclid_dist_mat = []
        manhatt_dist_mat=[]
        for q in range(0,len(self.test_data)):
            manhatt_dist_vec=[]
            euclid_dist_vec=[]
            for r in range(0,len(self.train_data)):
                x = self.test_data[q] - self.train_data[r]
                x = np.absolute(x)
                manhatt_dist = np.sum(x)
                manhatt_dist_vec.append((manhatt_dist, self.train_label[r]))
                manhatt_dist_vec.sort(key=byfirst)
                if (len(manhatt_dist_vec)>self.k):
                    manhatt_dist_vec.pop()
                
                x = x*x
                euclid_dist = np.sum(x)
                euclid_dist_vec.append((euclid_dist, self.train_label[r]))
                euclid_dist_vec.sort(key=byfirst)
                if (len(euclid_dist_vec)>self.k):
                    euclid_dist_vec.pop()
    
            manhatt_dist_mat.append(manhatt_dist_vec)
            euclid_dist_mat.append(euclid_dist_vec)
        
        manhatt_succ_count=0
        euclid_succ_count=0
        predict_label = []
        for j in range(0, len(self.test_data)):
            man_tmp_dist = manhatt_dist_mat[j][0:self.k]
            man_max_label=[x[1] for x in man_tmp_dist]
            man_predict_label=max(set(man_max_label), key=man_max_label.count)
            #if (self.test_label[j]==man_predict_label):
            #    manhatt_succ_count+=1
            
            euc_tmp_dist = euclid_dist_mat[j][0:self.k]
            euc_max_label=[x[1] for x in euc_tmp_dist]
            euc_predict_label=max(set(euc_max_label), key=euc_max_label.count)
            predict_label.append(euc_predict_label)
            #if (test_label[j]==euc_predict_label):
            #    euclid_succ_count+=1
        manhatt_success.append(manhatt_succ_count)
        euclid_success.append(euclid_succ_count)
        return predict_label
