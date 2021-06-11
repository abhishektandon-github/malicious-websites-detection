import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ipaddress

def dec_to_bin(x):
    y = []
    while(x != 0):
        y.append(x%2)
        x = x//2
    while(len(y) < 8):
        y.append(0)
    y.reverse()
    return y


def f(ip):
    ip_3, ip_2, ip_1, ip_0 = map(int, ip.split('.'))
    ip_addr = dec_to_bin(ip_3) + dec_to_bin(ip_2) + dec_to_bin(ip_1) + dec_to_bin(ip_0) 
    return ip_addr

# load dataset
df = pd.read_csv('data/dataset.csv')

# ip and target (numpy arrays)
df_ip = df['IP'].values
target = df['target'].values

arr = np.array(list(map(lambda x: f(x), df_ip)))
print(arr)

df2 = pd.DataFrame(data=arr).values

# Combining malicious and benign data for train/test

# X_train, y_train = 14,000 malicious + 10,000 benign (IPs) for training
# X_test, y_test = 1000 malicious + 22,000 benign (IPs) for testing


X_train = np.concatenate((df2[:14001, :24], df2[24438:34440, :24]), axis = 0)
print(X_train.shape)
y_train = np.concatenate((target[:14001], target[24438:34440]), axis = 0)

X_test = np.concatenate((df2[14002:15002, :24], df2[34441:56441, :24]), axis = 0)
y_test = np.concatenate((target[14002:15002], target[34441:56441]), axis = 0)



print("Accuracy:")

# RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("RandomForestClassifier:", accuracy_score(y_pred, y_test))

# Decision Tree

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("DecisionTreeClassifier:", accuracy_score(y_pred, y_test))

# SVM
model = SVC(gamma=0.01)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("SVM:", accuracy_score(y_pred, y_test))





# print(df['target'].iloc[24437]) // end of mal
# print(df['target'].iloc[24439]) // begin of benign