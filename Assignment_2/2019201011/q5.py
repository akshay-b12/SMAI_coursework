import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

class AuthorClassifier:
    def __init__ (self, C = 0.1, kernel = 'linear'):
        self.c = C
        self.kernel = kernel
        
    def train(self, path):
        print('Training started')
        df = pd.read_csv(path)
        self.train_data = df['text'].to_numpy()
        author_np = df['author'].to_numpy()
        self.Encoder = LabelEncoder()
        self.train_label = self.Encoder.fit_transform(author_np)
        print('Encoding done')
        tfidf_vectorizer=TfidfVectorizer(use_idf=True)
        self.fit_data=tfidf_vectorizer.fit(self.train_data)#_transform(Train_X)
        Train_X_Tfidf = self.fit_data.transform(self.train_data)
        
        self.SVM = svm.SVC(C=self.c, kernel=self.kernel, gamma='auto')
        self.SVM.fit(Train_X_Tfidf,self.train_label)# predict the labels on validation dataset
        print('SVM training done')
        
    def predict(self, path):
        print('Prediction started ')
        df = pd.read_csv(path)
        self.test_data = df['text'].to_numpy()
        #author_np = df['author'].to_numpy()
        #self.test_label = self.Encoder.fit_transform(author_np)
        Test_X_Tfidf = self.fit_data.transform(self.test_data)
        predictions_SVM = self.SVM.predict(Test_X_Tfidf)# Use accuracy_score function to get the accuracy
        print('Prediction done ')
        return predictions_SVM
