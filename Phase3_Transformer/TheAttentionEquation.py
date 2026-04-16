import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)

    # Step 1: Multiply Q and K^T (The transpose of K)
    score = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: scale score down
    scaled_score = score / math.sqrt(d_k)

    # Step 3: applying softmax
    attention_weight = F.softmax(scaled_score, dim=-1)

    # Step 4: multiply with V
    output = torch.matmul(attention_weight, V)

    return output, attention_weight

# Imagine a sequence of 4 words ("Bocchi plays the guitar"), each with a 64-dimension vector
sequence_length = 4
d_k = 64

# Generate random dummy matrices for Q, K, and V
Q = torch.randn(1, sequence_length, d_k)
K = torch.randn(1, sequence_length, d_k)
V = torch.randn(1, sequence_length, d_k)

# Run the attention brain
final_output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Final Output Shape: {final_output.shape}") 
# Output: [1, 4, 64] - We still have 4 words, but their 64 numbers are now context-aware!