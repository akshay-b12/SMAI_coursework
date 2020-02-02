import numpy as np
import pandas as pd

def byfirst(elem):
    return elem[0]

class KNNClassifier:
    k = 3
    col_names = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attach', 'gill_spacing', 
                                'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surf_above', 
                                'stalk_surf_below', 'stalk_color_above', 'stalk_color_below', 'veil_type', 'veil_color',
                                'ring_number', 'ring_type', 'spore_color', 'population', 'habitat']

    col_attributes = [['b', 'c', 'x', 'f', 'k', 's'], ['f', 'g', 's', 'y'],['b', 'c', 'e', 'g', 'n', 'p', 'r', 'u', 'w', 'y'],
                  ['f', 't'], ['a', 'c', 'f', 'l', 'm', 'n', 'p', 's', 'y'], ['a', 'f', 'd', 'n'],['c', 'w', 'd'],
                  ['b', 'n'], ['b', 'e', 'g', 'h', 'k', 'n', 'o', 'p', 'r', 'u', 'w', 'y'], ['e', 't'],
                  ['b', 'c', 'e', 'r', 'u', 'z'], ['f', 'k', 's', 'y'], ['f', 'k', 's', 'y'], ['b', 'c', 'e', 'g', 'n', 'o', 'p', 'w', 'y'],
                  ['b', 'c', 'e', 'g', 'n', 'o', 'p', 'w', 'y'], ['p', 'u'], ['n', 'o', 'w', 'y'], ['n', 'o', 't'],
                  ['e', 'f', 'l', 'n', 'p', 'c', 's', 'z'], ['b', 'h', 'k', 'n', 'o', 'r', 'u', 'w', 'y'], ['a', 'c', 'n', 's', 'v', 'y'],
                  ['d', 'g', 'l', 'm', 'p', 'u', 'w']]
    
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
        train_label_df = self.train_df.iloc[:,0]
        self.train_label = train_label_df.to_numpy()
        self.train_df = self.train_df.iloc[:,1:]
        
        self.train_df.columns = self.col_names
        for i in range(0,len(self.col_names)):
            self.train_df[self.col_names[i]] = pd.Categorical(self.train_df[self.col_names[i]],categories = self.col_attributes[i])

        self.train_df = self.train_df.replace(to_replace ="?", value =np.nan)
        for col in self.train_df.columns:
            self.train_df[col].fillna(self.train_df[col].mode()[0], inplace=True)
        
        self.train_df = pd.get_dummies(self.train_df)
        self.train_data = self.train_df.to_numpy()
        
    def predict(self, test_path):
        self.test_df = pd.read_csv(test_path, header=None)
        self.test_df.columns = self.col_names
        for i in range(0,len(self.col_names)):
            self.test_df[self.col_names[i]] = pd.Categorical(self.test_df[self.col_names[i]],categories = self.col_attributes[i])

        self.test_df = self.test_df.replace(to_replace ="?", value =np.nan)
        for col in self.test_df.columns:
            self.test_df[col].fillna(self.test_df[col].mode()[0], inplace=True)
            
        self.test_df = pd.get_dummies(self.test_df)
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