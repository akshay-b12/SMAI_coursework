import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

attrType = {}


##   Binary Tree class with all the atributes required by the Decision Tree node.
##   Df: Stores the dtaaframe at current node
##   left, right : Pointers to left and right children
##   leaf: Bool to denote whether a node is leaf or non-leaf
##   col: The column at which decision to split has been made at the current node
##   col_val: The categorical or numerical value at which the split took place at current node.
##   
##   Define the getter and setter methods for all the attributes.

class BinaryTree:
    def __init__(self):
        self.df = pd.DataFrame()
        self.left = None
        self.right = None
        self.leaf = True
        self.col = None
        self.col_val = None
        self.mean_val = 0.0
        
    def setdata(self, data):
        self.df = data
    
    def setleft(self, node):
        self.left = node
            
    def setright(self, node):
        self.right = node
    
    def setleaf(self, val):
        self.leaf = val;
    
    def setmean_val(self, val):
        self.mean_val = val;
        
    def getleft(self):
        return self.left
            
    def getright(self):
        return self.right

    def getmean_val(self):
        return self.mean_val
    
    def setcol(self, col_name):
        self.col = col_name

    def getcol(self):
        return self.col

    def setcol_val(self, val):
        self.col_val = val

    def getcol_val(self):
        return self.col_val


## Decsion tree class which takes dataframe as input and creates a decision tree.

class DecisionTree:
    
    def __init__(self):
        self.rootNode = None


##  The train() function first performs preprocessing on the data.
##  This includes :
##  - dropping the list which have values int less than the 30% of the columns
##  - Replace the "NA" values with the mean of the column for numerical data
##  - Replace the "NA" values with the mode of the column for categorical data

    def train(self, train_path):
        df = pd.read_csv(train_path, index_col=0)
        drop_list = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
        df.drop(drop_list, axis='columns', inplace=True)

        mean_replace = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
        for i in range(0, len(mean_replace)): 
            df[mean_replace[i]].fillna(df[mean_replace[i]].mean(), inplace=True)

        mode_replace = ['GarageCond', 'GarageType', 'GarageFinish', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                       'BsmtFinType2', 'Electrical', 'MasVnrType']

        for i in range(0, len(mode_replace)): 
            df[mode_replace[i]].fillna(df[mode_replace[i]].mode()[0], inplace=True)
        
        for cols in df.columns:
            attrType[cols]=[len(df[cols].unique()),df.shape[0],False]
            if df[cols].dtype == 'object':
                attrType[cols][2] = True
        self.rootNode = BinaryTree()
        self.rootNode.setdata(df)
        self.get_split_col(df, self.rootNode, 0)


##  -The function recursively splits the dataframe into left and right nodes by making decision based on the MSE value
##  -The stopping condition for the tree are : Either the no. of columns in the current node are less than 20 or,
##  if the height of the tree becomes 6.
##  -Based on the type of column, numerical or categorical, the MSE is found.
##  -The split is made at the column with least mean and the dataframe is split into left and right nodes.
        
    def get_split_col(self, data, root, ht):
        
        if(len(data) <=20):
            root.setleaf(True)
            root.setdata(data)
            root.setmean_val(self.findmse(data))
            return

        if(ht >= 6):
            root.setleaf(True)
            root.setdata(data)
            root.setmean_val(self.findmse(data))
            return
            
        root.setleaf(False)
        root.setdata(data)
        unique_val = {}

        #sale_price_col = data.iloc[:,-1]
        col_names = set(data.columns)
        col_names.remove('SalePrice')
        #col_names.remove('Id')
        #data = data.iloc[:,0:]
        for i in col_names:
            unique_val[i] = data[i].unique()
            #print(unique_val[i])
        col_mse_lis = []
        for col in unique_val.keys():
            #print(col)
            #print(attrType[col][2])
            #lis = unique_val[col]
            if attrType[col][2] == True:       # Categorica
                ret_val = self.get_cat_mse(data, col)
                col_mse_lis.append((col, ret_val[0], ret_val[1]))
            else:
                ret_val = self.get_num_mse(data, col)
                col_mse_lis.append([col, ret_val[0], ret_val[1]])
        col_mse_lis.sort(key=lambda x: x[2])
        #sorted(col_mse_lis, key=lambda x: x[2]) 
        root.setmean_val(col_mse_lis[0][1])
        if attrType[col_mse_lis[0][0]][2] == True:       # Categorical
            filt = (data[col_mse_lis[0][0]] == col_mse_lis[0][1])
            left = BinaryTree()
            right = BinaryTree()
            root.setleft(left) 
            root.setright(right)
            df_left = data[filt]
            df_right = data[~filt]
            root.setcol(col_mse_lis[0][0])
            root.setcol_val(col_mse_lis[0][1])
            root.setmean_val(data['SalePrice'].mean())
            #print(col_mse_lis[0][0], len(df_left), len(df_right))
            if len(df_left) > 0:
                self.get_split_col(df_left, root.getleft(), ht+1)
            if len(df_right) > 0:
                self.get_split_col(df_right, root.getright(), ht+1)
        else:
            filt = (data[col_mse_lis[0][0]] < col_mse_lis[0][1])
            left = BinaryTree()
            right = BinaryTree()
            root.setleft(left) 
            root.setright(right)
            df_left = data[filt]
            df_right = data[~filt]
            root.setcol(col_mse_lis[0][0])
            root.setcol_val(col_mse_lis[0][1])
            root.setmean_val(data['SalePrice'].mean())
            #print(col_mse_lis[0][0], len(df_left), len(df_right))
            if len(df_left) > 0:
                self.get_split_col(df_left, root.getleft(), ht+1)
            if len(df_right) > 0:
                self.get_split_col(df_right, root.getright(), ht+1)


##  This function gets the value of the column which gives the least MSE in sale price 
##  for the categorical type of data.
    
    def get_cat_mse(self, data, col_name):
        mse_lis = []
        lis =  data[col_name].unique()
        #print(col_name)
        for i in lis:
            filt = (data[col_name] == i)
            rows1 = data[filt]['SalePrice'].to_numpy()
            rows2 = data[~filt]['SalePrice'].to_numpy()
            
            msr1 = (rows1 - np.mean(rows1))**2
            msr2 = (rows2 - np.mean(rows2))**2
            total = len(msr1)+len(msr2)
            mse = int(((sum(msr1)*len(rows1))+(sum(msr2)*len(rows2)))/total)
            
            mse_lis.append([i,mse])
        mse_lis.sort(key = lambda x: x[1])
        #sorted(mse_lis, key=lambda x: x[1]) 
        return mse_lis[0]
    

##  This function gets the value of the column which gives the least MSE in sale price 
##  for the numerical type of data

    def get_num_mse(self, data, col_name):
        mse_lis = []
        lis =  data[col_name].to_numpy()
        lis = np.sort(lis)
        sorted(lis)
        for i in range(0, len(lis)-1):
            thresh = (lis[i]+lis[i+1])/2
            #print(type(data[col_name]), type(thresh))
            filt = (data[col_name] < thresh)
            rows1 = data[filt]['SalePrice'].to_numpy()
            rows2 = data[~filt]['SalePrice'].to_numpy()
            
            msr1 = (rows1 - np.mean(rows1))**2
            msr2 = (rows2 - np.mean(rows2))**2
            total = len(msr1)+len(msr2)
            mse = int(((sum(msr1)*len(rows1))+(sum(msr2)*len(rows2)))/total)
            
            mse_lis.append([thresh,mse])
        mse_lis.sort(key = lambda x: x[1])
        #sorted(mse_lis, key=lambda x: x[1]) 
        return mse_lis[0]

##  Finds the mean of the sale price for the dataframe.

    def findmse(self, data):
        return data['SalePrice'].mean()


##  For a given sample data with the property attributes, the predict() traverses the decision tree and
##  predicts the price of the property as the mean of the training samples present at that leaf node.
##  It calls the recursive function find_prediction() which returns the predicted sale price.

    def predict(self, test_path):
        df = pd.read_csv(test_path, index_col=0)

        drop_list = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']
        df.drop(drop_list, axis='columns', inplace=True)

        mean_replace = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
        for i in range(0, len(mean_replace)): 
            df[mean_replace[i]].fillna(df[mean_replace[i]].mean(), inplace=True)

        mode_replace = ['GarageCond', 'GarageType', 'GarageFinish', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                       'BsmtFinType2', 'Electrical', 'MasVnrType']

        for i in range(0, len(mode_replace)): 
            df[mode_replace[i]].fillna(df[mode_replace[i]].mode()[0], inplace=True)

        mean_predict_val = []
        for row in range(0,len(df)):
            mean_predict_val.append(self.find_prediction(self.rootNode, df.iloc[row]))
        return mean_predict_val


##  The function traverses the decision tree recursively by making decision based on the mean price at that node.
##  Once it reaches a leaf, it returns the mean price of the dataframe present at the leaf node

    def find_prediction(self,root, test_row):
        if( (root.getleft() == None) and (root.getright() == None)):
            return root.getmean_val()

        col_name = root.getcol()
        col_value = root.getcol_val()
        if attrType[col_name][2] == True:
            if col_value == test_row[col_name]:
                return self.find_prediction(root.getleft(), test_row)
            else:
                return self.find_prediction(root.getright(), test_row)
        else:
            if test_row[col_name] < col_value:
                return self.find_prediction(root.getleft(), test_row)
            else:
                return self.find_prediction(root.getright(), test_row)
            
