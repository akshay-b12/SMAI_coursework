from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


from q3 import Airfoil as ar
model3 = ar()
model3.train('./Datasets/q3/train.csv') # Path to the train.csv will be provided
prediction3, y_test = model3.predict('./Datasets/q3/test.csv')
print("Mean-Square-error: ", mean_squared_error(y_test, prediction3), end = "\t")
print ("R2 score : ", r2_score(y_test, prediction3))