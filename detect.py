import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('data/dataset.csv')

df_ip = df['IP'].values
target = df['target'].values

arr = np.array(list(map(lambda x: x.split('.'), df_ip))).astype(np.int)

new_df = pd.DataFrame(data = arr, columns = ['X1', 'X2', 'X3', 'X4'])

# Octet
new_df['N2'] = 256 + new_df['X2']
new_df['N3'] = 512 + new_df['X3']
new_df['N4'] = 768 + new_df['X4']

# Ex-Octet
new_df['N5'] = 768 + (new_df['X1']+new_df['X2'])%256
new_df['N6'] = 1024 + (new_df['X1']+new_df['X2']+new_df['X3'])%256

df2 = new_df[['X1', 'N2', 'N3', 'N5', 'N6']].values

X_train = np.concatenate((df2[:14001], df2[24438:34440]), axis = 0)
y_train = np.concatenate((target[:14001], target[24438:34440]), axis = 0)

X_test = np.concatenate((df2[10002:11002], df2[34441:49441]), axis = 0)
y_test = np.concatenate((target[10002:11002], target[34441:49441]), axis = 0)


model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.score(X_test, y_test))




# print(df['target'].iloc[24437]) // end of mal
# print(df['target'].iloc[24439]) // begin of benign