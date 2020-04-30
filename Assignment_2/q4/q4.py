import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

class Weather:
    def __init__(self):
        self.b = 0
        self.learning_rate = 0.001
        self.num_iters = 1000
        
        #self.test_data = test
        #self.test_op = test_label
    
    def train(self, path):
        df = pd.read_csv(path)
        attribute_list = ['Summary',	'Precip Type', 	'Temperature (C)', 	'Humidity', 	'Wind Speed (km/h)',
 	                'Wind Bearing (degrees)', 	'Visibility (km)', 	'Pressure (millibars)']
        self.categorical_list = ['Summary',	'Precip Type',]
        self.drop_list = ['Formatted Date', 'Daily Summary']
        df.drop(self.drop_list, axis='columns', inplace=True)
        df = df.dropna(how='any',axis=0)

        self.cat_col_attributes = [['Partly Cloudy', 'Mostly Cloudy', 'Overcast', 'Foggy',
                        'Breezy and Mostly Cloudy', 'Clear', 'Breezy and Partly Cloudy',
                        'Breezy and Overcast', 'Humid and Mostly Cloudy', 'Humid and Partly Cloudy',
                        'Windy and Foggy', 'Windy and Overcast', 'Breezy and Foggy', 'Breezy',
                        'Dry and Partly Cloudy', 'Windy and Partly Cloudy',
                        'Windy and Mostly Cloudy', 'Dangerously Windy and Partly Cloudy', 'Dry',
                        'Windy', 'Humid and Overcast', 'Light Rain', 'Drizzle',
                        'Dry and Mostly Cloudy', 'Breezy and Dry', 'Rain'],
                        ['rain', 'snow']]
        for i in range(0,len(self.categorical_list)):
            df[self.categorical_list[i]] = pd.Categorical(df[self.categorical_list[i]],categories = self.cat_col_attributes[i])
        
        df = pd.get_dummies(df)
        data_op = df['Apparent Temperature (C)']
        data_op = data_op.to_numpy()
        df = df.drop(columns=['Apparent Temperature (C)'])
        df_np = df.to_numpy()
        self.train_data = df_np
        self.train_op = data_op
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
        df = pd.read_csv(path)
        df.drop(self.drop_list, axis='columns', inplace=True)
        df = df.dropna(how='any',axis=0)
        for i in range(0,len(self.categorical_list)):
            df[self.categorical_list[i]] = pd.Categorical(df[self.categorical_list[i]],categories = self.cat_col_attributes[i])
        df = pd.get_dummies(df)
        data_op = df['Apparent Temperature (C)']
        data_op = data_op.to_numpy()
        df = df.drop(columns=['Apparent Temperature (C)'])
        df_np = df.to_numpy()
        #print(np.shape(data))
        self.test_data = df_np
        #print(np.shape(self.test_data))
        self.test_op = np.array(data_op)
        
        predict_op = []
        
        for i in range(len(self.test_data)):
            dotprod = [a*b for a,b in zip(self.m,self.test_data[i])]
            dotprod = np.sum(dotprod)
            pred = (dotprod+self.b)
            #print(pred)
            predict_op.append(pred)
        #print(np.shape(predict_op))
        return np.array(predict_op).flatten(), self.test_op