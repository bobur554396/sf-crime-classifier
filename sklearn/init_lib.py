import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
    


def initialise_train():
    train = pd.read_csv('train.csv', header=0)
    train = train.drop(['DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y'], axis=1)

    # 1. determine all values
    Categories = list(enumerate(sorted(np.unique(train['Category']))))
    Descriptions = list(enumerate(sorted(np.unique(train['Descript']))))

    # 2. set up dictionaries
    category_name_with_index = {i: name for i, name in Categories}

    CategoriesDict = {name: i for i, name in Categories}
    DescriptionsDict = {name: i for i, name in Descriptions}

    # 3. Convert all strings to int
    train.Category = train.Category.map(lambda x: CategoriesDict[x]).astype(int)
    train.Descript = train.Descript.map(lambda x: DescriptionsDict[x]).astype(int)

    return train, category_name_with_index

def initialise_test():
    test = pd.read_csv('norm_test.csv', header=0)
    test = test.drop(['DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y'], axis=1)

    test['Descript'] = test['Descript'].fillna('WARRANT ARREST')
    
    Descriptions = list(enumerate(sorted(np.unique(test['Descript']))))

    DescriptionsDict = {name: i for i, name in Descriptions}

    test.Descript = test.Descript.map(lambda x: DescriptionsDict[x]).astype(int)
    return test

def cleanData():
    file_data = []
    with open("test.csv") as f:
        lis = [line for line in f]
        for i, x in enumerate(lis):
            if i ==0:
                x = x[8:]
            if x[0] == '\"':
                x = x[1:]
            if x[-1] == '\"':
                x = x[:-1]
            if x[0] == ',':
                x = x[1:]
                x = x.replace('""', '"')
            file_data.append(x)

    with open("norm_test.csv", "w") as f:
        for i in file_data:
            f.write(i)
