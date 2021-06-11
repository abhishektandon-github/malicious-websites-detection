
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def analyse(df, string):

	# X_train = np.concatenate((df[:14001], df[24438:34440]), axis = 0)
	# y_train = np.concatenate((target[:14001], target[24438:34440]), axis = 0)

	# X_test = np.concatenate((df[10002:14002], df[34441:56441], df[24438:29438]), axis = 0)
	# y_test = np.concatenate((target[10002:14002], target[34441:56441], target[24438:29438]), axis = 0)
	
	X_train, X_test, y_train, y_test = train_test_split(df, target, test_size = 0.20, random_state = 25)

	print("Accuracy for: ",string)

	# RandomForestClassifier
	model = RandomForestClassifier()

	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	print("RandomForestClassifier:", accuracy_score(y_pred, y_test))

	# Decision Tree

	model = DecisionTreeClassifier()

	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	print("DecisionTreeClassifier:", accuracy_score(y_pred, y_test),'\n')

	# # SVM
	# model = SVC(gamma=0.01)

	# model.fit(X_train, y_train)

	# y_pred = model.predict(X_test)

	# print("SVM:", accuracy_score(y_pred, y_test))


# load dataset
df = pd.read_csv('data/dataset.csv')

# ip and target (numpy arrays)
df_ip = df['IP'].values
target = df['target'].values

# separate the 4 octets
arr = np.array(list(map(lambda x: x.split('.'), df_ip))).astype(np.int)

# convert numpy array `arr` to pandas DataFrame
new_df = pd.DataFrame(data = arr, columns = ['X1', 'X2', 'X3', 'X4'])

# ---- Feature Extraction ----

# Adding new features according to Octet Method
new_df['N2'] = 256 + new_df['X2']
new_df['N3'] = 512 + new_df['X3']
new_df['N4'] = 768 + new_df['X4']

# Additional features according to Ex-Octet Method
new_df['N5'] = 768 + (new_df['X1']+new_df['X2'])%256
new_df['N6'] = 1024 + (new_df['X1']+new_df['X2']+new_df['X3'])%256
new_df['N7'] = 1280 + (new_df['X1']+new_df['X2']+new_df['X3']+new_df['X4'])%256

# final dataframe used for train/test

df0 = new_df[['X1']].values
df1 = new_df[['X1', 'N2']].values
df2 = new_df[['X1', 'N2', 'N3']].values
df3 = new_df[['X1', 'N2', 'N3', 'N4']].values
df4 = new_df[['X1', 'N2', 'N3', 'N4', 'N5']].values
df5 = new_df[['X1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
df6 = new_df[['X1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']].values

df7 = new_df[['X1', 'X2']].values
df8 = new_df[['X1', 'X2', 'X3']].values
df9 = new_df[['X1', 'X2', 'X3', 'X4']].values

df10 = new_df[['X1', 'X2', 'N2']].values
df11 = new_df[['X1', 'X2', 'N3']].values
df12 = new_df[['X1', 'X2', 'N4']].values
df13 = new_df[['X1', 'X2', 'N2', 'N3']].values
df14 = new_df[['X1', 'X2', 'N2', 'N3', 'N4']].values

df15 = new_df[['X1', 'X2', 'X3', 'N2', 'N3']].values

df16 = new_df[['N2']].values
df17 = new_df[['N2', 'N3']].values
df18 = new_df[['N2', 'N3', 'N4']].values
df19 = new_df[['N2', 'N3', 'N4', 'N5']].values
df20 = new_df[['N2', 'N3', 'N4', 'N5', 'N6']].values

df21 = new_df[['N3']].values
df22 = new_df[['N3', 'N4']].values
df23 = new_df[['N3', 'N4', 'N5']].values
df24 = new_df[['N3', 'N4', 'N5', 'N6']].values

df25 = new_df[['N5', 'N6']].values
df26 = new_df[['N4', 'N5', 'N6']].values

df27 = new_df[['X1', 'X2', 'N5']].values
df28 = new_df[['X1', 'X2', 'N6']].values
df29 = new_df[['X1', 'X2', 'N7']].values

df30 = new_df[['X1', 'X2', 'X3', 'N6']].values
df31 = new_df[['X1', 'X2', 'X3', 'X4', 'N7']].values

df32 = new_df[['X1', 'N2', 'N5']].values
df33 = new_df[['X1', 'N2', 'N6']].values
df34 = new_df[['X1', 'N2', 'N7']].values

df35 = new_df[['X1', 'N2', 'N3', 'N6']].values
df36 = new_df[['X1', 'N2', 'N3', 'N4', 'N7']].values

analyse(df0, 'X1')
analyse(df1, 'X1, N2')
analyse(df2, 'X1, N2, N3')
analyse(df3, 'X1, N2, N3, N4')
analyse(df4, 'X1, N2, N3, N4, N5')
analyse(df5, 'X1, N2, N3, N4, N5, N6')
analyse(df6, 'X1, N2, N3, N4, N5, N6, N7')

analyse(df7, 'X1, X2')
analyse(df8, 'X1, X2, X3')
analyse(df9, 'X1, X2, X3, X4')

analyse(df10, 'X1, X2, N2')
analyse(df11, 'X1, X2, N3')
analyse(df12, 'X1, X2, N4')
analyse(df13, 'X1, X2, N2, N3')
analyse(df14, 'X1, X2, N2, N3, N4')

analyse(df15, 'X1, X2, X3, N2, N3')

analyse(df16, 'N2')
analyse(df17, 'N2, N3')
analyse(df18, 'N2, N3, N4')
analyse(df19, 'N2, N3, N4, N5')
analyse(df20, 'N2, N3, N4, N5, N6')


analyse(df21, 'N3')
analyse(df22, 'N3, N4')
analyse(df23, 'N3, N4, N5')
analyse(df24, 'N3, N4, N5, N6')

analyse(df25, 'N5, N6')
analyse(df26, 'N4, N5, N6')

analyse(df27, 'X1, X2, N5')
analyse(df28, 'X1, X2, N6')
analyse(df29, 'X1, X2, N7')

analyse(df30, 'X1, X2, X3, N6')
analyse(df31, 'X1, X2, X3, X4, N7')

analyse(df32, 'X1, N2, N5')
analyse(df33, 'X1, N2, N6')
analyse(df34, 'X1, N2, N7')

analyse(df35, 'X1, N2, N3, N6')
analyse(df36, 'X1, N2, N3, N4, N7')















# Combining malicious and benign data for train/test

# X_train, y_train = 14,000 malicious + 10,000 benign (IPs) for training
# X_test, y_test = 1000 malicious + 22,000 benign (IPs) for testing






# print(df['target'].iloc[24437]) // end of mal
# print(df['target'].iloc[24439]) // begin of benign