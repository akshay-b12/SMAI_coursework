from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from try_q5 import AuthorClassifier as ac
auth_classifier = ac()
auth_classifier.train('./Datasets/q5/train.csv') # Path to the train.csv will be provided
predictions, actual = auth_classifier.predict('./Datasets/q5/test.csv') # Path to the test.csv will be provided
print("SVM Accuracy Score -> ",accuracy_score(actual, predictions)*100)
print("confussion Matrix:->", confusion_matrix(actual, predictions))
print("F1-score -> ", f1_score(actual, predictions, average = None))