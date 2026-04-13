# BACKPROPOGATION FROM SCRATCH

# 1. The setup
weight = 3.0 # weight of the brain
x = 2.0     # The input data
target = 10.0   # The correct answer

# 2. The guess
prediction = weight * x
loss = (prediction - target) ** 2

# 3. The backpropogation

# Now to minimize the loss = (weight*x - target)**2, we need to differentiate the loss function wrt weight
# By using chain rule we have
gradient = 2 * (weight * x - prediction) * x

# 4. The updating Step
learning_rate = 0.05
weight_new = weight - (gradient * learning_rate)

print(f"Old weight: {weight}")
print(f"New smarter weight: {weight_new}")