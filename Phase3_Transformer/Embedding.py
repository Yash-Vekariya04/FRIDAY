# Here is how we write F.R.I.D.A.Y.'s embedding layer in PyTorch:
import torch
import torch.nn as nn

# Let's say her vocabulary is 10,000 words. 
# We want each word to be represented by a 512-dimension vector.
vocab_size = 10000
emb_size = 512

# Create the Embedding Lookup Table
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)

# Imagine you say the word "suit", and its ID in her dictionary is 402
word_id = torch.tensor([402])

# We pass the ID into the layer to get the mathematical meaning of the word
word_vector = embedding_layer(word_id)

print(f"Word Vector Shape: {word_vector.shape}") 
# Output: [1, 512] - One word, represented by 512 numbers