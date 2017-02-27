import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sig(x):
    return x * (1 - x)


# input
inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# output
outp = np.array([[0, 1, 1, 0]]).T

# random seed
np.random.seed(1)

# random for input weight
w0 = np.random.random((2, 2))
tresh0 = np.random.random((2, 1))

# random for hidden layer
w1 = np.random.random((2, 1))
tresh1 = np.random.random((1, 1))

lr = 0.2
iter = 0
sse = 1
rme = 0
# training data
while sse > 1e-3 and iter < 1000000:
    l0 = inp
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    l2_error = outp - l2
    rme = np.mean(np.abs(l2_error))
    sse = np.sum(l2_error ** 2)
    if iter % 5000 == 0:
        print("Iteration: %d, RME: %f, SSE: %f" % (iter, rme, sse))

    l2_delta = l2_error * deriv_sig(l2)

    l1_error = np.dot(l2_delta, w1.T)
    l1_delta = l1_error * deriv_sig(l1)

    w0 += lr * np.dot(l0.T, l1_delta)
    w1 += lr * np.dot(l1.T, l2_delta)

    iter += 1

print("Output : ")
print(l2)
print("Number of iterator : %d" % iter)
print("Root Mean Square Error : %f" % rme)
print("Sum of Square Error : %f" % sse)
