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
    data = np.empty((0, input_size), float)
    goal = np.empty((0, output_size), float)

    for row in data_read:
        data = np.append(data, np.array([[float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                                          float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]),
                                          float(row[11]), float(row[12]), float(row[13])]]), axis=0)
        goal = np.append(goal, np.array([[float(row[14]), float(row[15])]]), axis=0)

    goalT = goal.T
    true_goal = np.sum(goalT[0])
    false_goal = np.sum(goalT[1])

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

    max_acc = 0
    best_epoch = 0
    best_node_size = 0
    best_sse = 0
    best_conf = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'PT': 0, 'PF': 0}

    hidden_node_size = hidden_node_begin
    while hidden_node_size <= hidden_node_end:
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
            # if iter == 0 or iter == max_epoch - 1:
            if iter % 100 == 0:
                predicted_true = 0
                predicted_false = 0
                true_positive = 0
                true_negative = 0
                false_positive = 0
                false_negative = 0
                l2_iter = 0
                for l2_iter in range(0, len(l2), 1):
                    true_label = int(goal[l2_iter][0])
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

                rme = np.mean(np.absolute(l2_error))
                acc = (true_positive + true_negative) / data_size * 100
                if acc > max_acc:
                    max_acc = acc
                    best_epoch = iter
                    best_node_size = hidden_node_size
                    best_sse = sse
                    best_conf['TP'] = true_positive
                    best_conf['TN'] = true_negative
                    best_conf['FP'] = false_positive
                    best_conf['FN'] = false_negative
                    best_conf['PT'] = predicted_true
                    best_conf['PF'] = predicted_false
                    # print("Predicted Label: Positive: %d, Negative: %d" % (predicted_positive, predicted_negative))
                    # print("TP: %d, TN: %d, FP: %d, FN: %d" % (true_positive, true_negative, false_positive, false_negative))
                    # print(
                    #     "Node Size: %d, Epoches: %d, SSE: %f, RME: %f, Accuracy: %f" % (hidden_node_size, iter, sse, rme, acc))

            w0 += lr * np.dot(l0.T, l1_delta)
            w1 += lr * np.dot(l1.T, l2_delta)

            iter += 1

        hidden_node_size += 1

    print("Predicted Label: Positive: %d, Negative: %d" % (best_conf['PT'], best_conf['PF']))
    print("TP: %d, TN: %d, FP: %d, FN: %d" % (best_conf['TP'], best_conf['TN'], best_conf['FP'], best_conf['FN']))
    print("Node Size: %d, Epoches: %d, SSE: %f, Accuracy: %f\n" % (best_node_size, best_epoch, best_sse, max_acc))

    to_store = []
    to_store.append(volunteer_id)
    to_store.append(best_node_size)
    to_store.append(best_epoch)
    to_store.append(best_sse)
    to_store.append(max_acc)
    to_store.append(best_conf['PT'])
    to_store.append(best_conf['PF'])
    to_store.append(best_conf['TP'])
    to_store.append(best_conf['TN'])
    to_store.append(best_conf['FP'])
    to_store.append(best_conf['FN'])

    data_store.append(to_store)

print("All Result:")
print(data_store)
    # print("---- Complete ----")
    # print("Output Value:")
    # print(l2)
    # print("Total Epoches: %d,\nSSE: %f,\nRME: %f" % (iter, sse, rme))
