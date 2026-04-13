import numpy as np

# A sample of tensor of raw outputs (logits) from a neural network
# Imagine these are scores for the words: ['Hello', 'Tony', 'System']
logits = np.array([2.5, -1.0, 0.5])

# 1. ReLU Implementation
def relu(x):
    return np.maximum(0, x)

print(f"ReLU output: {relu(logits)}")

# 2. Sigmoid/Softmax Implementation
def softmax(x):
    # We use np.exp() to calculate the e^x for every element
    exponentials = np.exp(x)
    # Then divide by the sum of all the exponentials
    probabilities = exponentials / np.sum(exponentials)
    return probabilities

probs = softmax(logits)
print(f"Softmax Probabilities: {probs}")


'''
CHALLENGE-3: Imagine FRIDAY's brain is trying to predict the next word in the sentence: "Suit power is at ten..."

Her vocabulary is only 3 words: ['percent', 'apples', 'degrees'].
    "percent" = 3.0
    "apples" = 1.0
    "degrees" = 2.0

Without running the python code, calculate the approx denominator of the softmax function.
'''