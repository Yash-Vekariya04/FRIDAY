import numpy as np

# Flattening a 2*3 matrix into a 1D vector of 6 elements
matrix = np.array([[1, 2, 3], [4, 5, 6]])
flattened = matrix.reshape(6)

# Reshaping it back, but this time into 3 rows and 2 columns
reshaped = matrix.reshape(3, 2)