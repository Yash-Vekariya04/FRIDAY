'''
Assume we have a very simple neural network with a single weight w.
    w = 5.0
    L = w ** 2
    learning_rate = 0.1

Find the gradient and the next value of w.
'''

w = 5.0
loss = w ** 2
learning_rate = 0.1

gradient = 2 * w
new_w = w - (gradient * learning_rate)

print(new_w)