import numpy as np

vec1 = np.random.rand(10)
vec2 = np.random.rand(10)

# The slow way (Standard loops)
def slow_dot_product(vec1, vec2):
    result = 0
    for i in range(len(vec1)):
        result += vec1[i] * vec2[i]
    return result


# The fast way (Vectorisation)
fast_result = np.dot(vec1, vec2)