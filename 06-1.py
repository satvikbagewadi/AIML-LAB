import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

msg = pd.read_csv('DataSet6.csv', names=['message','label'])
msg['labelnum'] = msg.label.map({'pos':1,'neg':0})
X = msg.message
Y = msg.labelnum

print('First 5 instances: ')
for x,y in zip(X[:5],Y[:5]) :
   print(x , ', ', y)

x_train, x_test, y_train, y_test = train_test_split(X,Y)
print("Total training instances: ", x_train.shape)
print('Total testing instances: ', x_test.shape)

countVect = CountVectorizer()
xtrain_dtm = countVect.fit_transform(x_train)
x_test_dtm = countVect.transform(x_test)

print("Total feature extracted: ", xtrain_dtm.shape)
print("Features of first 5 are listed below: ")

df = pd.DataFrame(xtrain_dtm.toarray(), columns=countVect.get_feature_names())
print(df[0:5])

clf = MultinomialNB().fit(xtrain_dtm, y_train)
predicted = clf.predict(x_test_dtm)

print('Classification results: ')
for x,y in zip(x_test, predicted):
   pred = 'Pos' if y==1 else 'Neg'
   print(x ," -> ", pred)

print("Accuracy metrics: ")
print("Classifier Accuracy: ", metrics.accuracy_score(y_test, predicted))
print("Recall: ", metrics.recall_score(y_test, predicted))
print("Precision: ", metrics.precision_score(y_test, predicted))
print("Confusion matrix: ")
print(metrics.confusion_matrix(predicted,y_test))
