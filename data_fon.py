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

to_shuff = np.empty((0, (input_size + output_size)), float)
for row in data_read:
    to_shuff = np.append(to_shuff, np.array([[float(row[1]),
                                              float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                                              float(row[7]), float(row[8]), float(row[9]), float(row[10]),
                                              float(row[11]), float(row[12]), float(row[13]), float(row[14])]]), axis=0)

# shuffle data
np.random.shuffle(to_shuff)

# init data model
data = np.empty((0, input_size), float)
goal = np.empty((0, output_size), float)

for row in to_shuff:
    data = np.append(data,
                     np.array([[row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10],
                                row[11], row[12]]]), axis=0)
    goal = np.append(goal, np.array([[row[13]]]), axis=0)
    # data = np.append(data, np.array([[float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
    #                                   float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]),
    #                                   float(row[11]), float(row[12]), float(row[13])]]), axis=0)
    # goal = np.append(goal, np.array([[float(row[14])]]), axis=0)

# find data size
data_size = len(data)

# set neural config
lr = 0.2
hidden_node_size = 4
goal_sse = 1e-2

epoch_list_size = 300
epoch_step = 10000

# set validation
k_fold = 10

# random seed
np.random.seed(1)

# error_variable
validation_error = []

# begin training
# each max_epoch
for each_epoch in range(epoch_list_size):
    print("EPOCH: %d" % each_epoch)
    validation_error.append([])
    # each validation
    for i in range(k_fold):
        print("K : %d" % i)
        # setup cross validation
        data_validate = np.empty((0, input_size), float)
        goal_validate = np.empty((0, output_size), float)
        data_train = np.empty((0, input_size), float)
        goal_train = np.empty((0, output_size), float)

        for j in range(data_size):
            if i * k_fold <= j < (i + 1) * k_fold:
                data_validate = np.append(data_validate, np.array([data[j]]), axis=0)
                goal_validate = np.append(goal_validate, np.array([goal[j]]), axis=0)
            else:
                data_train = np.append(data_train, np.array([data[j]]), axis=0)
                goal_train = np.append(goal_train, np.array([goal[j]]), axis=0)

        # init measure value
        iter = 0
        sse = 100
        rme = 100

        # random weight
        w0 = 2 * np.random.random((input_size, hidden_node_size)) - 1
        w1 = 2 * np.random.random((hidden_node_size, output_size)) - 1

        # train data
        while sse > goal_sse and iter < (each_epoch + 1) * epoch_step:
            l0 = data_train

            l1 = sigmoid(np.dot(l0, w0))
            l2 = sigmoid(np.dot(l1, w1))

            l2_error = goal_train - l2
            l2_delta = l2_error * deriv_sigmoid(l2)

            l1_error = np.dot(l2_delta, w1.T)
            l2_delta = l1_error * deriv_sigmoid(l1)

            sse = np.sum(l2_error ** 2)
            if iter % 10000 == 0:
                print("Iter: %d" % iter)
            # rme = np.mean(np.absolute(l2_error))
            #     print("Epoches: %d, SSE: %f, RME: %f" % (iter, sse, rme))

            iter += 1

        # test data
        l0 = data_validate
        l1 = sigmoid(np.dot(l0, w0))
        l2 = sigmoid(np.dot(l1, w1))
        l2_error = goal_validate - l2

        sse = np.sum(l2_error ** 2)
        validation_error[each_epoch].append(sse)

print("---- Complete ----")
print(validation_error)
# print("Output Value:")
# print(l2)
# print("Total Epoches: %d,\nSSE: %f,\nRME: %f" % (iter, sse, rme))
