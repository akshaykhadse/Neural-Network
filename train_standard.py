import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier

# Loading Data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/kaggle_test_data.csv")
train_data = train_data.drop('id', axis=1)
test_ids = test_data['id']
test_data = test_data.drop('id', axis=1)

# Replacing String values by numbers
coloumn_headers = list(train_data)
to_drop = ['age', 'fnlwgt', 'education-num', 'capital-gain',
           'capital-loss', 'hours-per-week', 'salary']
coloumn_headers = [v for i, v in enumerate(coloumn_headers)
                   if v not in to_drop]
for coloumn in coloumn_headers:
    for i, v in enumerate(train_data[coloumn].unique()):
        train_data[coloumn] = train_data[coloumn].replace(v, int(i))
        test_data[coloumn] = test_data[coloumn].replace(v, int(i))

# Dropping Output from Train Data
output = np.array(train_data)[:, -1].reshape(1, len(train_data))
train_data = train_data.drop('salary', axis=1)
# Dropping Education Coloumn Train Data
train_data = train_data.drop("education", axis=1)
test_data = test_data.drop("education", axis=1)
coloumn_headers.remove('education')
# One Hot encoding for discontinous data
for coloumn in coloumn_headers:
    for k in range(len(train_data[coloumn].unique())):
        train_data[coloumn+str(k)] = (train_data[coloumn] == k)*1
        test_data[coloumn+str(k)] = (test_data[coloumn] == k)*1
    train_data = train_data.drop(coloumn, axis=1)
    test_data = test_data.drop(coloumn, axis=1)

# Min Max Scaling
min_cols = train_data.min()
max_cols = train_data.max()
train_data -= min_cols
test_data -= min_cols
train_data /= max_cols
test_data /= max_cols

# 1. Logistic Regression
model = LogisticRegression()
model.fit(train_data, output.ravel().T)
score_lr = model.score(train_data, output.ravel().T)
print('Logistic Regression: %f' % score_lr)
out = model.predict(test_data)
np.savetxt("results/predictions_1.csv", np.concatenate((test_ids.values.reshape(len(test_ids), 1), out.reshape(len(test_ids), 1)), axis=1), delimiter=',', fmt=['%d', '%d'], header='id,salary', comments='')

# 2. Decision Tree
model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(train_data, output.ravel().T)
score_dt = model.score(train_data, output.ravel().T)
print('Decision Tree: %f' % score_dt)
out = model.predict(test_data)
np.savetxt("results/predictions_2.csv", np.concatenate((test_ids.values.reshape(len(test_ids), 1), out.reshape(len(test_ids), 1)), axis=1), delimiter=',', fmt=['%d', '%d'], header='id,salary', comments='')

# 3. Support Vector Classifier
model = svm.SVC()
model.fit(train_data, output.ravel().T)
score_sv = model.score(train_data, output.ravel().T)
print('Support Vector Classifier: %f' % score_sv)
out = model.predict(test_data)
np.savetxt("results/predictions_3.csv", np.concatenate((test_ids.values.reshape(len(test_ids), 1), out.reshape(len(test_ids), 1)), axis=1), delimiter=',', fmt=['%d', '%d'], header='id,salary', comments='')

# K- Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=6)
model.fit(train_data, output.ravel().T)
score_kn = model.score(train_data, output.ravel().T)
print('K- Nearest Neighbors: %f' % score_kn)
#out = model.predict(test_data)
#np.savetxt("results/predictions_1.csv", np.concatenate((test_ids.values.reshape(len(test_ids), 1), out.reshape(len(test_ids), 1)), axis=1), delimiter=',', fmt=['%d', '%d'], header='id,salary', comments='')
