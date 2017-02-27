import numpy as np
import csv


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid function
def deriv_sigmoid(x):
    return x * (1 - x)


# open file
file = open('dataset/train_normalize.csv')
data_read = csv.reader(file)

# set input and output size
inp_size = 8
outp_size = 1

# init data model
data = np.empty((0, inp_size), float)
goal = np.empty((0, outp_size), float)

# read data from file
for row in data_read:
    data = np.append(data, np.array([[float(row[2]), float(row[4]), float(row[5]), float(row[6]), float(row[7]),
                                      float(row[9]), float(row[10]), float(row[11])]]), axis=0)
    goal = np.append(goal, np.array([[float(row[1])]]), axis=0)

# find data_size
data_size = len(data)

# set neural config
lr = 0.2
hidden_node_size = 4
max_epoch = 10000000
goal_sse = 1e-2 * data_size

# random seed
np.random.seed(1)

# random weight
w0 = 2 * np.random.random((inp_size, hidden_node_size)) - 1
w1 = 2 * np.random.random((hidden_node_size, outp_size)) - 1

# random treshold
tresh0 = 2 * np.random.random((1, hidden_node_size)) - 1
tresh1 = 2 * np.random.random((1, outp_size)) - 1

# initial measure value
sse = 100
rme = 1
iter = 0

# begin training
while sse > goal_sse and iter < max_epoch:
    l0 = data

    # get output with treshold
    # l1 = sigmoid(np.dot(l0, w0) - tresh0)
    # l2 = sigmoid(np.dot(l1, w1) - tresh1)

    # get output without treshold
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    l2_error = goal - l2
    l2_delta = l2_error * deriv_sigmoid(l2)

    l1_error = np.dot(l2_delta, w1.T)
    l1_delta = l1_error * deriv_sigmoid(l1)

    w0 += lr * np.dot(l0.T, l1_delta)
    w1 += lr * np.dot(l1.T, l2_delta)

    # adjust treshold from error
    # tresh0 += lr * np.sum(l1_delta, axis=0)
    # tresh1 += lr * np.sum(l2_delta, axis=0)

    sse = np.sum(l2_error ** 2)
    rme = np.mean(np.absolute(l2_error))
    if iter % 5000 == 0:
        print("Epoches: %d, SSE: %f, RME: %f" % (iter, sse, rme))
    iter += 1

print("---- Complete ----")
print("Output Value:")
print(l2)
print("Total Epoches: %d,\nSSE: %f,\nRME: %f" % (iter, sse, rme))
