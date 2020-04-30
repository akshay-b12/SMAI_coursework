import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

class Airfoil:
    def __init__(self):
        self.b = 0
        self.learning_rate = 0.01
        self.num_iters = 1000
        
        #self.test_data = test
        #self.test_op = test_label
    
    def train(self, path):
        data = pd.read_csv(path, header=None)
        self.train_data = data.iloc[:,:-1]
        self.train_op = np.array(data.iloc[:, -1])
        self.train_data = preprocessing.scale(self.train_data)
        self.feature_size = len(self.train_data[0])
        self.m = [0]*len(self.train_data[0])
        self.gradient_descent()
        
    def compute_error(self, data, data_op):
        total_error = 0
        for i in range(data):
            total_error += (data_op - (np.dot(self.m,data[i])+self.b))**2
        return total_error
    
    def gradient_descent(self):
        self.train_data = list(self.train_data)
        self.train_op = list(self.train_op)
        for i in range(self.num_iters):
            for j in range(len(self.train_data)):
                self.stoch_grad(self.train_data[j], self.train_op[j], len(self.train_data))
        #print(self.m)
        #print(self.b)
        
    def stoch_grad(self, data_row, data_op, train_size):
        m_tmp = self.m
        b_tmp = self.b
        dotprod = [a*b for a,b in zip(m_tmp,data_row)]
        dotprod = np.sum(dotprod)
        tmp = (self.learning_rate*( dotprod + b_tmp - data_op))/train_size
        for k in range(self.feature_size):
            self.m[k] = m_tmp[k] - tmp*data_row[k]
        self.b = b_tmp - tmp
            
    def predict(self, path):
        data = pd.read_csv(path, header=None)
        #print(np.shape(data))
        self.test_data = data
        #print(np.shape(self.test_data))
        #self.test_op = np.array(data.iloc[:, -1])
        self.test_data = preprocessing.scale(self.test_data)
        predict_op = []
        
        for i in range(len(self.test_data)):
            dotprod = [a*b for a,b in zip(self.m,self.test_data[i])]
            dotprod = np.sum(dotprod)
            pred = (dotprod+self.b)
            #print(pred)
            predict_op.append(pred)
        #print(np.shape(predict_op))
        return np.array(predict_op).flatten()
