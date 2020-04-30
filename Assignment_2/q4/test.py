from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from q4 import Weather as wr
model4 = wr()
model4.train('./Datasets/q4/train.csv') # Path to the train.csv will be provided 
prediction4, y_test = model4.predict('./Datasets/q4/test.csv')
print("Mean-Square-error: ", mean_squared_error(y_test, prediction4), end = "\t")
print ("R2 score : ", r2_score(y_test, prediction4))
