from sklearn.metrics.cluster import homogeneity_score

from q6 import Cluster as cl
cluster_algo = cl()
# You will be given path to a directory which has a list of documents. You need to return a list of cluster labels for those documents
predictions, actual = cluster_algo.cluster('./Datasets/') 
print(homogeneity_score(actual,predictions))
'''SCORE BASED ON THE ACCURACY OF CLUSTER LABELS'''
