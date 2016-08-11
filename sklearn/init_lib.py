import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve
    


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
    test = pd.read_csv('test.csv', header=0)
    test = test.drop(['Category', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y'], axis=1)

    test['Descript'] = test['Descript'].fillna('WARRANT ARREST')
    
    Descriptions = list(enumerate(sorted(np.unique(test['Descript']))))

    DescriptionsDict = {name: i for i, name in Descriptions}

    test.Descript = test.Descript.map(lambda x: DescriptionsDict[x]).astype(int)
    return test