import numpy as np

# Use NumPy to generate a 3D tensor populated with random numbers. Give it a shape of (4, 5, 6).

tensor_3d = np.random.rand(4, 5, 6)

# Without using a standard for loop, multiply every single number in that entire tensor by 10.

tensor_3d *= 10

# Reshape that modified 3D tensor into a 2D matrix where the number of rows is 10. What is the correct number of columns?

# total elements in 3d tensor = 4 * 5 * 6 = 120
reshaped_matrix = tensor_3d.reshape(10, 12)