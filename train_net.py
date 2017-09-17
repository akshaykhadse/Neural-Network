import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/kaggle_test_data.csv")
train_data = train_data.drop('id', axis=1)
test_ids = test_data['id']
test_data = test_data.drop('id', axis=1)

# Data Visualization
'''
# Plotting Distribution of each coloumn
for i, col in enumerate(train_data.columns):
    fig = plt.figure(i)
    fig.suptitle(col)
    if train_data.dtypes[col] == np.object:
        train_data[col].value_counts().plot(kind="bar")
    else:
        train_data[col].hist()
        plt.xticks(rotation="vertical")
    plt.savefig("output/"+col+".png")

# Printing Details of Native Country
print("Native Country Details")
print((train_data["native-country"].value_counts()/train_data.shape[0]).head())
'''
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
'''
# plotting Correlation matrix
sb.plt.figure()
sb.heatmap(train_data.corr(), square=True)
sb.plt.xticks(rotation="vertical")
sb.plt.yticks(rotation="horizontal")
sb.plt.savefig("output/corr.png")
'''

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


# Neural Network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return(np.exp(-x)/(1 + np.exp(-x))**2)


# Initialize Weights
w1 = np.random.normal(0, 0.01, [100, train_data.shape[1]+1])
w2 = np.random.normal(0, 0.01, [25, 101])
w3 = np.random.normal(0, 0.01, [1, 26])

step = 0.2
iterations = 25000
# Full training took 50000 iterations
lambd = 0.01

# Train Neural Net
p_cost = 1

train_data = np.array(train_data)
train_data = np.hstack((train_data, np.ones([len(train_data), 1])))
test_data = np.array(test_data)
test_data = np.hstack((test_data, np.ones([len(test_data), 1])))

for i in range(iterations):
    # Forward Propogation
    z1 = np.dot(w1, train_data.T)
    y1 = sigmoid(z1)

    y1 = np.vstack((y1, np.ones([1, len(y1[1, :])])))
    z2 = np.dot(w2, y1)
    y2 = sigmoid(z2)

    y2 = np.vstack((y2, np.ones([1, len(y2[1, :])])))
    z3 = np.dot(w3, y2)
    y3 = sigmoid(z3)

    cost = (-1 / len(train_data)) * np.sum(output * np.log(y3) + (1 - output) * (np.log(1 - y3))) + lambd/(2*len(train_data)) * (np.sum(np.square(w1[:, 0:-1])) + np.sum(np.square(w2[:, 0:-1])) + np.sum(np.square(w3[:, 0:-1])))
    print('cost: ' + str(cost) + ' step: ' + str(step))

    # Backward Propogation of gradient
    delta_3 = (-1 / len(train_data)) * np.multiply(output/y3 - (1 - output)/(1 - y3), d_sigmoid(z3))
    dc_dw3 = np.dot(delta_3, y2.T)
    dc_dw3[:, 0:-1] = dc_dw3[:, 0:-1] + lambd / len(train_data) * w3[:, 0:-1]

    delta_2 = np.multiply(np.dot(w3.T, delta_3)[0:-1, :], d_sigmoid(z2))
    dc_dw2 = np.dot(delta_2, y1.T)
    dc_dw2[:, 0:-1] = dc_dw2[:, 0:-1] + lambd / len(train_data) * w2[:, 0:-1]

    delta_1 = np.multiply(np.dot(w2.T, delta_2)[0:-1, :], d_sigmoid(z1))
    dc_dw1 = np.dot(delta_1, np.array(train_data))
    dc_dw1[:, 0:-1] = dc_dw1[:, 0:-1] + lambd / len(train_data) * w1[:, 0:-1]

    # Update Weights
    w3 = w3 - step * dc_dw3
    w2 = w2 - step * dc_dw2
    w1 = w1 - step * dc_dw1

    # Update step size based on previous cost
    if p_cost > cost:
        step = 1.1*step
    else:
        step = 0.9*step
    p_cost = cost

# Calculate Train Accuracy
z1 = np.dot(w1, train_data.T)
y1 = sigmoid(z1)

y1 = np.vstack((y1, np.ones([1, len(y1[1, :])])))
z2 = np.dot(w2, y1)
y2 = sigmoid(z2)

y2 = np.vstack((y2, np.ones([1, len(y2[1, :])])))
z3 = np.dot(w3, y2)
y3 = sigmoid(z3)

accuracy = np.count_nonzero(np.greater(y3, 0.5) == output)/len(train_data)
print('Train acuracy: ' + str(accuracy))

# Saving Weights
with open('weights.txt', 'w') as weights:
    for j, w in enumerate([w1, w2, w3]):
        for i in range(len(w[:, 1])):
            weights.write('w'+str(j+1)+'|'+str(i)+'|'+np.array2string(w[i,:], max_line_width=10000, separator=',')+'\n')

