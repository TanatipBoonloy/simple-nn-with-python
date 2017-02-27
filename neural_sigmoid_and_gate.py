import numpy as np


# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# derivation of sigmoid function
def deriv_sigmoid(x):
    return x * (1.0 - x)


# input
inp = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

# output
outp = np.array([[0.0, 0.0, 0.0, 1.0]]).T

# set random seed
np.random.seed(1)

# random input weigh
w0 = 2*np.random.random((2,1))-1
tresh = [-0.8]

# init loop condition
iter = 0
best_sse = 1
best_iter = 0

lr = 0.8
# train data
while best_sse > 1e-2 and iter < 10000:
    l0 = inp
    l1 = sigmoid(np.dot(l0, w0))

    l1_error = outp - l1
    new_sse = np.sum(np.power(l1_error - np.average(l1_error), 2))
    l1_delta = l1_error * deriv_sigmoid(l1)

    tresh += lr * np.dot(np.array([1.0, 1.0, 1.0, 1.0]), l1_delta)
    w0 += lr * np.dot(l0.T, l1_delta)

    if new_sse < best_sse:
        best_sse = new_sse
        best_iter = iter

    iter += 1

print("result : ")
print(l1)
print("total iteration : %d" % iter)
print("best sum of square error : %f, at iter: %f" % (best_sse,best_iter))
