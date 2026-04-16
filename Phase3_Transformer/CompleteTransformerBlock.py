import torch
import torch.nn as nn
import torch.nn.functional as F

class FridayDecoderBlock(nn.Module):
    def __init__(self, embed_dim, heads, hidden_dim):
        super(FridayDecoderBlock, self).__init__()

        # 1. The Attention Mechnism
        self.attention = nn.MultiheadAttention(embed_dim, heads, batch_first=True)

        # 2. MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # 3. Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 1. Self-Attention with Residual Connection
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output # The unblocked highway

        # 2. MLP
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output

        return x
    

# Lets test the block
# Sequence of 4 words, each with 512 dimension embedding
dummy_sentence = torch.randn(1, 4, 512)

# Initiaze one block in FRIDAY's brain
# We split the 512 dimensions across 8 "heads" (64 dims per head, just like out mutusl math!)
block = FridayDecoderBlock(embed_dim=512, heads=8, hidden_dim=2048)

# Pass the sentece through the block
processed_sentence = block(dummy_sentence)

print(f"Input Shape: {dummy_sentence.shape}")
print(f"Output Shape: {processed_sentence.shape}")


# EXTRA DOUBTS SOLUTION:

# What if i change batch_first=True to False?
'''
If you change batch_first=True to batch_first=False in the nn.MultiheadAttention initialization, the code will fail at runtime with a shape mismatch error during the forward pass.

Why This Happens:

    With batch_first=True (the current setting), nn.MultiheadAttention expects input tensors in the shape (batch_size, seq_len, embed_dim). Your dummy input torch.randn(1, 4, 512) matches this: batch_size=1, seq_len=4, embed_dim=512.
    
    With batch_first=False, it expects (seq_len, batch_size, embed_dim) instead. Your input would need to be (4, 1, 512) to work, but it's still (1, 4, 512), causing a dimension mismatch.
'''

# What have i done in the layer normalization part?

'''
You've implemented **pre-layer normalization** in the transformer block, which is a common and effective approach in modern transformer architectures (like GPT-style decoders). Here's a breakdown of what you've done and why it matters:

### What You've Done
1. **Defined Two LayerNorm Layers**:
   - `self.norm1 = nn.LayerNorm(embed_dim)`: Normalizes the input before the multi-head attention.
   - `self.norm2 = nn.LayerNorm(embed_dim)`: Normalizes the input before the MLP (feed-forward network).

2. **Applied Them in the Forward Pass**:
   - Before attention: `self.attention(self.norm1(x), self.norm1(x), self.norm1(x))`
   - Before MLP: `self.mlp(self.norm2(x))`
   - The normalized outputs are added back to the original `x` via residual connections: `x = x + attn_output` and `x = x + mlp_output`.

### Why This Works
- **Layer Normalization**: `nn.LayerNorm(embed_dim)` normalizes across the embedding dimension (512 in your case) for each token in the sequence independently. It computes the mean and variance of the features for each token, then scales and shifts them. This helps stabilize training by reducing internal covariate shift.
- **Pre-Norm vs. Post-Norm**:
  - **Pre-Norm** (what you have): Normalize *before* the sub-layer (attention or MLP). This is more stable during training, allows for deeper networks, and is used in models like GPT-3, BERT, etc.
  - **Post-Norm** (original Transformer paper): Normalize *after* the sub-layer. It can be less stable for very deep models.
- **Residual Connections**: The `x + output` ensures gradients flow directly through the network, preventing vanishing gradients and enabling deeper architectures.

### Benefits in Your Code
- Improves training stability and convergence.
- Allows the model to learn better representations without exploding/vanishing activations.
- The residual connections ensure the model can "skip" layers if needed, making it easier to train.

'''