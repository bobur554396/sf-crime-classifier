import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from time import time
import numpy as np
import init_lib

 
train, category_name_with_index = init_lib.initialise_train()
test = init_lib.initialise_test()


print "train data: ", train.shape
print "test data: ", test.shape


features = ["Descript"]

clf = DecisionTreeClassifier()

t0 = time()
d_tree = clf.fit(train[features], train["Category"])
print "training time: ", round(time()-t0, 8), "s"

t1 = time()
pred = clf.predict(test)
print "predicting time: ", round(time()-t1, 8), "s"




test_original = pd.read_csv('test.csv', header=0)
test_original["Category"] = pd.Series([category_name_with_index[train["Category"][pred[x]]] for x in range(len(pred))])


result = pd.DataFrame(test_original)

result.to_csv('predicted_test.csv', index = False)


# test accuracy

X_train, X_test, y_train, y_test = train_test_split(train[features], train["Category"], train_size=.60)
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

pred_accuracy = clf.predict(X_test)
acc = accuracy_score(y_test, pred_accuracy)
print "accuracy: ", acc
