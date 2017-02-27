import numpy as np


# sigmoid function
def sig(x):
    return 1 / (1 + np.exp(-x))


# derivation of sigmoid function
def deriv_sig(x):
    return x * (1 - x)


def linear(x):
    x_iter = 0
    while x_iter < len(x):
        if x[x_iter] > 0.2:
            x[x_iter] = 1
        else:
            x[x_iter] = 0
        x_iter += 1
    return x


# input
inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# output
outp = np.array([[0, 0, 0, 1]]).T

# set random seed
np.random.seed(1)

# random initial weight
syn0 = np.random.random((2, 1))

# training data
old_error = 10000
iter = 0
err_inc = 0
l1 = np.random.random((4, 1))
while (l1 != outp).any() and iter < 1000:
    l0 = inp
    # l1 = sig(np.dot(l0, syn0))
    l1 = linear(np.dot(l0, syn0))
    l1_error = outp - l1
    # sum_error = np.dot(np.array([1, 1, 1, 1]), np.absolute(l1_error))
    # if np.absolute(sum_error[0]) < old_error:
    #     old_error = np.absolute(sum_error[0])
    #     err_inc = 0
    # else:
    #     if np.absolute(sum_error[0]) > :
    #         print("error inc : %f" % sum_error[0])
    #         print(l1_error)
    #         err_inc += 1

    # l1_delta = l1_error * deriv_sig(l1)
    l1_delta = l1_error
    # print(l1_delta)
    # sum_err_grad = np.dot(np.array([1, 1, 1, 1]), np.absolute(l1_delta))
    # print(l1_delta)
    # print(sum_err_grad)
    # if sum_err_grad[0] < 1e-1:
    #     break
    syn0 += 0.1 * np.dot(l0.T, l1_delta)

    iter += 1

# print output
print("\nout :")
print(l1)

print("loop count: %d" % iter)
