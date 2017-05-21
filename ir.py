import numpy as np
import csv


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid function
def deriv_sigmoid(x):
    return x * (1 - x)


data_store = []
file_name = 'dataset/ir.csv'
# open file
file = open(file_name)
data_read = csv.reader(file)

# set input and output size
input_size = 10
output_size = 2

# init data model
data = np.empty((0, input_size), float)
goal = np.empty((0, output_size), float)

for row in data_read:
    data = np.append(data, np.array([[float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                                      float(row[5]), float(row[6]), float(row[7]), float(row[8]),
                                      float(row[9])]]), axis=0)
    goal = np.append(goal, np.array([[float(row[10]), 1.0 - float(row[10])]]), axis=0)

goalT = goal.T
true_goal = np.sum(goalT[0])
false_goal = np.sum(goalT[1])

# find data size
data_size = len(data)

# set neural config
lr = 0.3
# hidden_node_size =
goal_sse = 1e-2
max_epoch = 600
hidden_node_size = 5

# random seed
np.random.seed(1)

max_acc = 0
best_epoch = 0
best_node_size = 0
best_sse = 0
best_conf = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'PT': 0, 'PF': 0}

# print("Node: %d" % hidden_node_size)
# random weight
w0 = np.random.random((input_size, hidden_node_size))
w1 = np.random.random((hidden_node_size, output_size))

# init measure value
iter = 0
sse = 100
rme = 100

# train data
while sse > goal_sse and iter < max_epoch:
    l0 = data

    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    l2_error = goal - l2
    l2_delta = l2_error * deriv_sigmoid(l2)

    l1_error = np.dot(l2_delta, w1.T)
    l1_delta = l1_error * deriv_sigmoid(l1)

    sse = np.sum(l2_error ** 2)

    w0 += lr * np.dot(l0.T, l1_delta)
    w1 += lr * np.dot(l1.T, l2_delta)

    iter += 1

test = np.array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 1], [0, 1, 1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                 [0, 0, 1, 1, 0, 0, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0, 0, 0, 0, 1]])
l0 = test

l1 = sigmoid(np.dot(l0, w0))
l2 = sigmoid(np.dot(l1, w1))

result = np.round(l2)
print(result)
