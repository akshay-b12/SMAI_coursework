import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVC
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getImage(img_mat, plot=False):
    assert img_mat.shape in [(3072,), (3072, 1)] # sanity check
    r_channel = img_mat[:1024].reshape(32, 32)
    g_channel = img_mat[1024: 2 * 1024].reshape(32, 32)
    b_channel = img_mat[2 * 1024:].reshape(32, 32)
    image_repr = np.stack([r_channel, g_channel, b_channel], axis=2)
    assert image_repr.shape == (32, 32, 3) # sanity check
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(image_repr), plt.show(block=False)

    return image_repr

def getSIFT(img):
    sift = cv2.xfeatures2d.SIFT_create()
    if img.shape in [(3072, 1), (3072,)]: img = getImage(img)
    kps, des = sift.detectAndCompute(img, None)
    return des if des is not None else np.array([]).reshape(0, 128)

def load_images_SIFT(path):
    data_np = []
    labels_np = []
    for i in range(5):
        data_dict = unpickle(path+"/data_batch_"+str(i+1))
        data_np.append(data_dict[b'data'])
        labels_np.append(data_dict[b'labels'])
        print("Batch ", i+1, " data loaded")
    #label_list = ['airplane', 'automobile', 'bird', 'cat', 
    #              'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    descriptor_list = []
    sift_vectors = {str(key): [] for key in range(10)} 
    #categorywise_sift = np.empty(dtype=np.float64)
    for i in range(len(data_np)):
        for j in range(len(data_np[i])):
            feature = getSIFT(data_np[i][j])
            #categorywise_sift[labels_np[i]].append(feature)
            descriptor_list.extend(feature)
            sift_vectors[str(labels_np[i][j])].append(feature)
        print("SIFT of batch ",i+1," calculated")
    #for i in range(len(label_list)):
    #    images[label_list[i]] = categorywise_sift[i]
    return [descriptor_list, sift_vectors, labels_np]

def kmeans(k, descriptor_list):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=666)
    kmeans.fit(descriptor_list) #kmeans.partial_fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words
    
def image_class(all_bovw, centers):
    #label_list = ['airplane', 'automobile', 'bird', 'cat', 
    #              'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dict_feature = {key: [] for key in range(10)} 
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature
  
def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i]) 
           #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i]) 
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind

def load_images_hog(path):
    data_np = []
    labels_np = []
    for i in range(1):
        data_dict = unpickle(path+"/data_batch_"+str(i+1))
        data_np.append(data_dict[b'data'])
        labels_np.append(data_dict[b'labels'])
        print("Batch ", i+1, " data loaded")
    #label_list = ['airplane', 'automobile', 'bird', 'cat', 
    #              'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    descriptor_list = []
    hog_vectors = {str(key): [] for key in range(10)} 
    #categorywise_sift = np.empty(dtype=np.float64)
    for i in range(len(data_np)):
        for j in range(len(data_np[i])):
            feature = getHOG(data_np[i][j])
            #categorywise_sift[labels_np[i]].append(feature)
            descriptor_list.extend(feature)
            hog_vectors[str(labels_np[i][j])].append(feature)
        print("SIFT of batch %d calculated", i+1)
    #for i in range(len(label_list)):
    #    images[label_list[i]] = categorywise_sift[i]
    return [descriptor_list, hog_vectors, labels_np]
def getHOG(image):
    winSize = (16,16)
    blockSize = (8,8)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 4
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (0,0)#(8,8)
    locations = ((10,20),)
    hist = hog.compute(image,winStride,padding,locations)
    return hist


label_dict = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 
                  'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

########------ Using SIFT + Bag of words -------##########
feature_list = load_images_SIFT("/content/drive/My Drive/SMAI/Assignment-2/q1/cifar-10-python/cifar-10-batches-py")
print("SIFT Feature list loaded and computed")
descriptor_list = feature_list[0] 
all_bovw_feature = feature_list[1]
data_labels = feature_list[2]


# Takes the sift features that is seperated class by class for test data
visual_words = kmeans(800, descriptor_list)
print("KMeans computed")

bovw_train = image_class(all_bovw_feature, visual_words)
print("Visual words computed")
with open("bovw_train.txt", "wb") as fp:   #Pickling
    pickle.dump(bovw_train, fp)
#np.savez("bovw_train", bovw_train)
print("Visual words written") 


bovw_train = np.load("bovw_train.txt", allow_pickle=True)
final_label = []
final_feature = []
for key, value in bovw_train.items():
    for val in value:
      final_label.append(int(key))
      final_feature.append(val)
print(np.shape(final_label))
print(np.shape(final_feature))

X_train, X_test, y_train, y_test = train_test_split(final_feature, final_label, test_size=0.2, random_state=666)
print(len(X_train), len(X_test), len(y_train), len(y_test))
#clf = linear_model.SGDClassifier(max_iter=10, alpha = 0.01, loss='hinge', random_state = 666)
clf = SVC()
clf.fit(X_train, y_train)

clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print("Accuracy using SIFT + BoVW : ",accuracy)


########------ Using HoG + Bag of words -------##########

feature_list = load_images_hog("/content/drive/My Drive/SMAI/Assignment-2/q1/cifar-10-python/cifar-10-batches-py")
print("HOG Feature list loaded and computed")
descriptor_list = feature_list[0] 
# Takes the HoG features that is seperated class by class for train data
all_bovw_feature = feature_list[1]
data_labels = feature_list[2]

visual_words = kmeans(400, descriptor_list)
print("KMeans computed")

bovw_train = image_class(all_bovw_feature, visual_words)
print("Visual words computed")
with open("/content/drive/My Drive/SMAI/Assignment-2/q1/bovw_train_hog.txt", "wb") as fp:   #Pickling
    pickle.dump(bovw_train, fp)
print("Visual words written") 

bovw_train = np.load("/content/drive/My Drive/SMAI/Assignment-2/q1/bovw_train_hog.txt", allow_pickle=True)
final_label = []
final_feature = []
for key, value in bovw_train.items():
    for val in value:
      final_label.append(int(key))
      final_feature.append(val)
print(np.shape(final_label))
print(np.shape(final_feature))

X_train, X_test, y_train, y_test = train_test_split(final_feature, final_label, test_size=0.2, random_state=666)
print(len(X_train), len(X_test), len(y_train), len(y_test))
#clf = linear_model.SGDClassifier(max_iter=10, alpha = 0.01, loss='hinge', random_state = 666)
clf = SVC()
clf.fit(X_train, y_train)

clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print("Accuracy using HoG + BoVW : ",accuracy)