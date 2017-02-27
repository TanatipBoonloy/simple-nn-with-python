import numpy as np

data = np.array([[3, 3, 3], [4, 4, 4], [5, 5, 5]])
tresh = np.array([[1, 1, 1]])

new_data = np.empty((0, 2), int)

for row in data:
    new_data = np.append(new_data, np.array([[row[0],row[1]]]), axis=0)

print(new_data)

# result = data - tresh
# print(result)
# print(np.sum(result,axis=0))
