import numpy as np
import pandas as pd
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import homogeneity_score
from scipy.spatial import distance

class Cluster:
    def __init__(self, k=5, iterate = 100):
        self.k = k
        self.iterations = iterate
                          
    def kmeans(self):
        pred_cluster = []
        sum_for_mean = [np.zeros(self.dim)]*self.k
        pred_count = np.zeros(self.k)
        #print(np.shape(self.centroid))
        for i in range(self.N):
            min_dist = float("inf")
            min_dist_cluster = -1
            for j in range(self.k):
                euc_dist = distance.euclidean(self.data[i],np.array(self.centroid[j]))
                if euc_dist < min_dist:
                    min_dist = euc_dist
                    min_dist_cluster = j
            sum_for_mean[min_dist_cluster] = np.add(sum_for_mean[min_dist_cluster], self.data[i])
            pred_count[min_dist_cluster] += 1
            pred_cluster.append(min_dist_cluster)
        for i in range(self.k):
            self.centroid[i] = np.divide(sum_for_mean[i], pred_count[i])
        return pred_cluster
    
    def cluster(self, path):
        df = self.get_data(path)
        train_data = df['text'].to_numpy()
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)#, preprocessor=my_preprocessor, tokenizer=my_tokenizer)#, ngram_range=(1,2),)
        self.fit_data=tfidf_vectorizer.fit(train_data)
        self.data = self.fit_data.transform(train_data)
        self.dim = np.shape(self.data)[1] #assuming data stored row-wise
        self.N = np.shape(self.data)[0]
        self.data = self.data.toarray()
        self.centroid = []  #[ [data[random.randint(0, self.N-1)]] for _ in range(self.k)]
        for i in range(self.k):
            centre = np.random.uniform(1,10, self.dim) #for j in range(self.dim)]
            norm_centre = np.linalg.norm(centre)
            self.centroid.append(np.divide(centre, norm_centre))
        print(self.centroid)
        pred_cluster = [-1]*self.N
        label_np = df['label'].to_numpy()
        for i in range(self.iterations):
            self.pred_cluster = self.kmeans()
            #print(self.pred_cluster)
            print("Iteration ", i+1, " done")
        return self.pred_cluster, label_np
    
    def get_data(self,path):
        path = path+"*.txt"
        files = glob.glob(path)
        cols = ['text', 'label']
        df = pd.DataFrame(columns = cols)
        #print(df)
        df_list = []
        for file in files:
            f = open(file, mode = 'r', errors = 'ignore')
            txt_data = f.read()
            txt_data = ''.join([i if ord(i) < 128 else ' ' for i in txt_data])
            #name_split = file.split('.')
            #name_split = name_split[0].split('_')
            #print(name_split)
            label = int(file[-5])-1
            df_list.append([txt_data, label])
            f.close()
            df2 = pd.DataFrame(df_list, columns = cols)
            df = df.append(df2)
        return df