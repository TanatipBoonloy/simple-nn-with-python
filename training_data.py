import numpy as np
import csv


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid function
def deriv_sigmoid(x):
    return x * (1 - x)


# open file
file = open('dataset/normalize_data_fon.csv')
data_read = csv.reader(file)

# set input and output size
input_size = 13
output_size = 1

# init data model
data = np.empty((0, input_size), float)
goal = np.empty((0, output_size), float)

for row in data_read:
    data = np.append(data, np.array([[float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                                      float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]),
                                      float(row[11]), float(row[12]), float(row[13])]]), axis=0)
    goal = np.append(goal, np.array([[float(row[14])]]), axis=0)

# find data size
data_size = len(data)

# set neural config
lr = 0.2
# hidden_node_size =
goal_sse = 1e-2
max_epoch = 10000
hidden_node_begin = 1
hidden_node_end = 20

# random seed
np.random.seed(1)

hidden_node_size = hidden_node_begin
while hidden_node_size <= hidden_node_end:
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
        if iter == 0 or iter == max_epoch-1:
            print("Node Size: %d, Iter: %d" % (hidden_node_size,iter))
            rme = np.mean(np.absolute(l2_error))
            print("Node Size: %d, Epoches: %d, SSE: %f, RME: %f" % (hidden_node_size,iter, sse, rme))

        w0 += lr * np.dot(l0.T, l1_delta)
        w1 += lr * np.dot(l1.T, l2_delta)

        iter += 1

    hidden_node_size += 1

# print("---- Complete ----")
# print("Output Value:")
# print(l2)
# print("Total Epoches: %d,\nSSE: %f,\nRME: %f" % (iter, sse, rme))
