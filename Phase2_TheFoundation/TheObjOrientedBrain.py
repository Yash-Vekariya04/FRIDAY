import torch
import torch.nn as nn # Used to define tha layers
import torch.nn.functional as F # Collection of activation functions

# Defining our own custom neural network class
class FridayBrainBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FridayBrainBlock, self).__init__()

        # Layer-1: Matrix of weights connecting input to hidden layer
        self.layer1 = nn.Linear(input_size, hidden_size)

        # Layer-2: Matrix of weights connecting hidden to output layer
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Step-1: Pass the data through first matrix multiplication
        x = self.layer1(x)

        # Step-2: Apply non-linearity
        x = F.gelu(x)

        # Step-3: Pass the data through final matrix to get the new logits
        x = self.layer2(x)

        return x
    
model = FridayBrainBlock(input_size=512, hidden_size=2048, output_size=10000)
print(model)


# Pushing the data through the brain (testcase)
dummy_word_vector = torch.randn(512)

raw_logits = model(dummy_word_vector)

print(f"Initial shape: {dummy_word_vector.shape}")
print(f"Output shape: {raw_logits.shape}")