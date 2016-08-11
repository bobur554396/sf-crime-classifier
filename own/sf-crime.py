import pandas as pd
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import init_lib
from dt import MyDecisionTree


train, category_name_with_index = init_lib.initialise_train()
test = init_lib.initialise_test()


print "train data: ", train.shape
print "test data: ", test.shape


features = ["Descript"]


clf = MyDecisionTree()

# print train[features].as_matrix()

# print train["Category"].as_matrix()

t0 = time()
clf.fit(train[features].as_matrix(), train["Category"].as_matrix())
print "training time: ", round(time()-t0, 8), "s"

# t1 = time()
# pred = clf.predict(test.as_matrix())
# print "predicting time: ", round(time()-t1, 8), "s"


# test_original = pd.read_csv('test.csv', header=0)
# test_original["Category"] = pd.Series([category_name_with_index[train["Category"][pred[x]]] for x in range(len(pred))])


# result = pd.DataFrame(test_original)

# result.to_csv('predicted_test.csv', index = False)


# # test accuracy

# X_train, X_test, y_train, y_test = train_test_split(train[features], train["Category"], train_size=.60)

# clf.fit(X_train.as_matrix(), y_train.as_matrix())

# pred_accuracy = clf.predict(X_test.as_matrix())
# acc = accuracy_score(y_test, pred_accuracy)
# print "accuracy: ", acc
