import numpy as np
import csv


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid function
def deriv_sigmoid(x):
    return x * (1 - x)


data_store = []

for volunteer_id in range(1, 11, 1):
    file_name = 'dataset/two_labels/volunteer_' + str(volunteer_id) + '.csv'
    print("\nVolunteer: %d" % volunteer_id)

    # open file
    file = open(file_name)
    data_read = csv.reader(file)

    # set input and output size
    input_size = 13
    output_size = 2

    # init data model
    # class liked = true, class disliked = false
    data_true = np.empty((0, input_size), float)
    data_false = np.empty((0, input_size), float)
    goal_true = np.empty((0, output_size), float)
    goal_false = np.empty((0, output_size), float)

    for row in data_read:
        if int(row[14]) == 1:
            data_true = np.append(data_true,
                                  np.array([[float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                                             float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]),
                                             float(row[11]), float(row[12]), float(row[13])]]), axis=0)
            goal_true = np.append(goal_true, np.array([[float(row[14]), float(row[15])]]), axis=0)
        else:
            data_false = np.append(data_false,
                                   np.array([[float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                                              float(row[6]), float(row[7]), float(row[8]), float(row[9]),
                                              float(row[10]),
                                              float(row[11]), float(row[12]), float(row[13])]]), axis=0)
            goal_false = np.append(goal_false, np.array([[float(row[14]), float(row[15])]]), axis=0)

    # find data size
    data_true_size = len(data_true)
    data_false_size = len(data_false)

    data = []
    goal = []
    data.append(data_true)
    data.append(data_false)
    goal.append(goal_true)
    goal.append(goal_false)

    # data_size = data_true_size + data_false_size

    # set neural config
    lr = 0.2

    hidden_node_size = 11
    goal_sse = 1e-2
    epoch_step = 2000
    start_epoch = 2000
    max_epoch_size = 10000
    # max_epoch = 10000
    hidden_node_begin = 1
    hidden_node_end = 20

    fold_range = 10

    # random seed
    np.random.seed(1)

    # setup weight
    w0_init = np.random.random((input_size, hidden_node_size))
    w1_init = np.random.random((hidden_node_size, output_size))

    max_acc = 0
    best_epoch = 0
    best_node_size = 0
    best_sse = 0
    best_conf = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'PT': 0, 'PF': 0}

    epoch_list = []
    for max_epoch in range(start_epoch, max_epoch_size, epoch_step):
        if max_epoch % 1000 == 0:
            print("Max Epoches: %d" % max_epoch)

        fold_list = []
        for fold_iter in range(0, fold_range, 1):
            # print("Node: %d" % hidden_node_size)
            # random weight

            data_validate = np.empty((0, input_size), float)
            goal_validate = np.empty((0, output_size), float)
            data_train = np.empty((0, input_size), float)
            goal_train = np.empty((0, output_size), float)

            for data_set_iter in range(0, 2, 1):
                for data_iter in range(0, len(data[data_set_iter]), 1):
                    data_len = len(data[data_set_iter])
                    start = int((data_len / 10) * fold_iter)
                    stop = int((data_len / 10) * (fold_iter + 1))
                    if start <= data_iter < stop:
                        data_validate = np.append(data_validate, np.array([data[data_set_iter][data_iter]]), axis=0)
                        goal_validate = np.append(goal_validate, np.array([goal[data_set_iter][data_iter]]), axis=0)
                    else:
                        data_train = np.append(data_train, np.array([data[data_set_iter][data_iter]]), axis=0)
                        goal_train = np.append(goal_train, np.array([goal[data_set_iter][data_iter]]), axis=0)

            # init measure value
            iter = 0
            sse = 100
            rme = 100

            w0 = np.random.random((input_size, hidden_node_size))
            w1 = np.random.random((hidden_node_size, output_size))

            # train data
            while sse > goal_sse and iter < max_epoch:
                l0 = data_train

                l1 = sigmoid(np.dot(l0, w0))
                l2 = sigmoid(np.dot(l1, w1))

                l2_error = goal_train - l2
                l2_delta = l2_error * deriv_sigmoid(l2)

                l1_error = np.dot(l2_delta, w1.T)
                l1_delta = l1_error * deriv_sigmoid(l1)

                sse = np.sum(l2_error ** 2)

                w0 += lr * np.dot(l0.T, l1_delta)
                w1 += lr * np.dot(l1.T, l2_delta)

                iter += 1

            # test data
            l0 = data_validate
            l1 = sigmoid(np.dot(l0, w0))
            l2 = sigmoid(np.dot(l1, w1))
            l2_error = goal_validate - l2

            predicted_true = 0
            predicted_false = 0
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0
            l2_iter = 0
            for l2_iter in range(0, len(l2), 1):
                true_label = int(goal_validate[l2_iter][0])
                if l2[l2_iter][0] >= l2[l2_iter][1]:
                    predicted_true += 1
                    if true_label == 1:
                        true_positive += 1
                    else:
                        false_positive += 1
                else:
                    predicted_false += 1
                    if true_label == 1:
                        false_negative += 1
                    else:
                        true_negative += 1

            # rme = np.mean(np.absolute(l2_error))
            acc = (true_positive + true_negative) / len(data_validate) * 100

            fold_list.append(acc)
            print("Acc: %f, Class 1: %d, Class 2: %d, TP: %d, TN: %d" % (
            acc, predicted_true, predicted_false, true_positive, true_negative))
        avg_acc = np.average(fold_list)
        print(avg_acc)
        # avg_acc = (np.sum(fold_list))/fold_range
        epoch_list.append(avg_acc)
    data_store.append(epoch_list)
print("\nResult:")
print(data_store)

#     print("Predicted Label: Positive: %d, Negative: %d" % (best_conf['PT'], best_conf['PF']))
#     print("TP: %d, TN: %d, FP: %d, FN: %d" % (best_conf['TP'], best_conf['TN'], best_conf['FP'], best_conf['FN']))
#     print("Node Size: %d, Epoches: %d, SSE: %f, Accuracy: %f\n" % (best_node_size, best_epoch, best_sse, max_acc))
#
#     to_store = []
#     to_store.append(volunteer_id)
#     to_store.append(best_node_size)
#     to_store.append(best_epoch)
#     to_store.append(best_sse)
#     to_store.append(max_acc)
#     to_store.append(best_conf['PT'])
#     to_store.append(best_conf['PF'])
#     to_store.append(best_conf['TP'])
#     to_store.append(best_conf['TN'])
#     to_store.append(best_conf['FP'])
#     to_store.append(best_conf['FN'])
#
#     data_store.append(to_store)
#
# print("All Result:")
# print(data_store)
#
# # print("---- Complete ----")
# # print("Output Value:")
# # print(l2)
# # print("Total Epoches: %d,\nSSE: %f,\nRME: %f" % (iter, sse, rme))
