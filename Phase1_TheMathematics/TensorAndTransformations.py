import numpy as np

# Creating a 1D Tensor (Vector)
vector = np.array([1, 2, 3, 4])
print(f"1D Vector shape: {vector.shape}")

# Creating a 2D Tensor (Matrix)
matrix = np.array(
    [1, 2, 3],
    [4, 5, 6]
    )
print(f"2D Matrix shape: {matrix.shape}")

# Creating a 3D Tensor (A batch of data)
# Imagine two sentence with 3 words, where each word is 4-number vector
tensor_3d = np.random.rand(2, 3, 4)
print(f"3D Tensor shape: {tensor_3d.shape}")