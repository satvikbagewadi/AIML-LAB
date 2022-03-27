from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
print("Dataset loaded")

x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, test_size = 0.1)

print("Size of training data and its label: ", x_train.shape, y_train.shape)
print("Size of testing data and its label: ", x_test.shape, y_test.shape)

for i in range (len(iris.target_names)) :
   print("Label ", i , " - ", str(iris.target_names[i]))

neighbors = KNeighborsClassifier(n_neighbors= 1)
neighbors.fit(x_train, y_train)
y_pred = neighbors.predict(x_test)

for i in range (0, len(x_test)) :
   print("Sample: ", str(x_test[i]), "Actual Label: ", str(y_test[i]), "Prediction: ", str(y_pred[i]))

print("Accuracy: ", neighbors.score(x_test, y_test))