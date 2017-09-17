import pandas as pd
import numpy as np

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

# Loading Saved Weights
w1 = np.empty([100, train_data.shape[1]+1])
w2 = np.empty([25, 101])
w3 = np.empty([1, 26])
with open('weights.txt') as weights:
    for line in weights:
        elements = line.split('|')
        exec(elements[0]+'['+elements[1]+',:] = np.fromstring("'+elements[2].rstrip().replace('[', '').replace(']', '')+'",sep=",")')

train_data = np.array(train_data)
train_data = np.hstack((train_data, np.ones([len(train_data), 1])))
test_data = np.array(test_data)
test_data = np.hstack((test_data, np.ones([len(test_data), 1])))

# Evaluate on test data
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return(np.exp(-x)/(1 + np.exp(-x))**2)


z1 = np.dot(w1, test_data.T)
y1 = sigmoid(z1)

y1 = np.vstack((y1, np.ones([1, len(y1[1, :])])))
z2 = np.dot(w2, y1)
y2 = sigmoid(z2)

y2 = np.vstack((y2, np.ones([1, len(y2[1, :])])))
z3 = np.dot(w3, y2)
y3 = sigmoid(z3)
out = np.greater(y3, 0.5) * 1

np.savetxt("results/predictions.csv", np.concatenate((test_ids.values.reshape(len(test_ids), 1), out.T), axis=1), delimiter=',', fmt=['%d', '%d'], header='id,salary', comments='')

