import numpy as np
import cv2
import glob
from MyPCA import MyPCA
from sklearn.model_selection import train_test_split
import sys

class LogisticRegression:
    def __init__(self, learn_rate = 0.001, num_iters = 100):
        self.learning_rate = learn_rate
        self.n_iters = num_iters
        self.weights = None
        self.bias = None
        
    def train(self, data, labels):
        self.data = self.add_bias_col(data)
        self.n_samples, self.n_features = self.data.shape 
        self.classes = np.unique(labels)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        #print(self.class_labels)
        labels = self.one_hot_encode(labels)
        self.weights = np.zeros(shape=(len(self.classes),self.data.shape[1]))
        for _ in range(self.n_iters):
            y = np.dot(self.data, self.weights.T).reshape(-1,len(self.classes)) ## y = m*x + c
            ## apply softmax
            y_predicted = self.softmax(y)
            #y_predicted = self.sigmoidfn(y)
            
            # compute gradients
            dw = np.dot((y_predicted - labels).T, self.data)
            # update parameters
            self.weights -= self.learning_rate * dw
        #print(self.weights)
    
    def add_bias_col(self,X):
        return np.insert(X, 0, 1, axis=1)
    
    def one_hot_encode(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
    '''
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    '''
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)
    
    def predict(self, X):
        X = self.add_bias_col(X)
        pred_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
        self.probs_ = self.softmax(pred_vals)
        pred_classes = np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))
        return pred_classes
        #return np.mean(pred_classes == y)
    
    
def read_data(file_name, train):
    data_file = open(file_name)
    img_files = data_file.readlines()
    #print(img_files)
    gray_images = []
    labels = []
    for file in img_files:
        file = file.split()
        img = cv2.imread(file[0])
        img = cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA) #None,fx=0.5,fy=0.5
        flat_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
        gray_images.append(flat_img)
        if train :
            labels.append(file[1])

    if train :
        return np.asarray(gray_images), labels
    else :
        return np.asarray(gray_images)
  
if len(sys.argv) != 3 :
    print("Invalid number of arguments. Expected two arguments")
    exit()

data, labels = read_data(sys.argv[1], train = True)
pca = MyPCA(n_components = 0.95)#n_components = 0.95
pca_data = pca.fit(data)
#print(pca_data.shape)

#train_X, test_X, train_y, test_y = train_test_split(pca_data, labels, train_size=0.8, random_state=666)
#print(np.shape(train_X), np.shape(test_X), np.shape(train_y))
logreg = LogisticRegression()
logreg.train(np.asarray(pca_data), np.asarray(labels))
test_images = read_data(sys.argv[2], train = False)
test_pca = pca.transform(test_images)
pred_label = logreg.predict(np.asarray(test_pca))
print(*pred_label, sep='\n')