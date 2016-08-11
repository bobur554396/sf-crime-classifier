import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from time import time
import numpy as np
import init_lib


init_lib.cleanData()

 
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


test_original = pd.read_csv('norm_test.csv', header=0)
test_original["Category"] = pd.Series([category_name_with_index[train["Category"][pred[x]]] for x in range(len(pred))])


# result = pd.DataFrame(test_original)

# result.to_csv('predicted_test.csv', index = False)


# print test_original

submission = pd.DataFrame({
	"Category": test_original['Category'],
	"Descript": test_original['Descript'],
	"DayOfWeek": test_original['DayOfWeek'],
	"PdDistrict": test_original['PdDistrict'],
	"Resolution": test_original['Resolution'],
	"Address": test_original['Address'],
	"X": test_original['X'],
	"Y": test_original['Y']
})

submission.to_csv("test_submit1.csv", index=False, columns=['Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y'])

# test accuracy

# X_train, X_test, y_train, y_test = train_test_split(train[features], train["Category"], train_size=.60)
# clf = DecisionTreeClassifier()

# clf.fit(X_train, y_train)

# pred_accuracy = clf.predict(X_test)
# acc = accuracy_score(y_test, pred_accuracy)
# print "accuracy: ", acc
