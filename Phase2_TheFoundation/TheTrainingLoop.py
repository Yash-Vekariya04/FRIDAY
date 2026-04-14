import torch
import torch.nn as nn
import torch.nn.functional as F

# Step-1 : Setup
class FridayBrainBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FridayBrainBlock, self).__init__()
        # Defining Layer1 : connecting input to hidden layer
        self.layer1 = nn.Linear(input_size, hidden_size)
        # Defining Layer2 : connecting weight matrix to output layer
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):

        x = self.layer1(x) # Applying matrix multiplication
        x = F.gelu(x) # Applying non-linearity
        x = self.layer2(x) # # matrix to logits conversion

        return x

model = FridayBrainBlock(input_size=512, hidden_size=2048, output_size=10000)

# --- 2. The Teacher (Loss & Optimizer) ---
# CrossEntropyLoss automatically applies Softmax inside it for numerical stability!
loss_function = nn.CrossEntropyLoss()

# We tell AdamW to optimize the 'parameters' (the weight matrices) of our model
# learning_rate (lr) is the size of the step it takes down the mountain
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# --- 3. The Data ---
# Our dummy input word
x_input = torch.randn(1, 512)
# print(x_input.shape) -- garbage

# The 'correct' answer. Let's say the correct next word is at index 402 in her vocabulary.
# It must be a 1D tensor containing the correct index.
y_correct = torch.tensor([402])

# ==========================================
# --- 4. THE 5-STEP TRAINING LOOP ---
# ==========================================

# 1. forward pass (making a guess)
predictions = model(x_input)

# 2. Calculating loss
loss = loss_function(predictions, y_correct)
print(f"Initial Error (Loss): {loss.item():.4f}")

# Step 3: Zero the Gradients (Crucial PyTorch mechanics)
optimizer.zero_grad()

# Step 4: Backward Pass (The Calculus! Calculate dL/dw for every weight)
loss.backward()

# Step 5: Optimizer Step (Update the weights to be smarter)
optimizer.step()

print("Weights updated successfully. She is now slightly smarter.")


'''
Your Challenge (Lesson 2.3)
There is a massive trap in the 5-step loop above that catches a lot of developers off-guard, and it's a favorite question in technical exams.

Look closely at Step 3: optimizer.zero_grad().

Before we calculate the new calculus derivatives (loss.backward()), we intentionally wipe the old gradients out of memory by setting them to zero.

Based on your understanding of loops and memory from your computing background, what do you think would happen mathematically to the AI if we forgot to include optimizer.zero_grad() in a for loop that runs 100 times?
'''

'''
ANSWER:
In standard Python, if you calculate a variable x = 5, and then later calculate x = 2, the old value is deleted and replaced.

But PyTorch's loss.backward() function behaves differently. When it calculates the gradient (the step size), it adds it to whatever gradient is already sitting in memory. This is called gradient accumulation.

If you forget optimizer.zero_grad(), here is what happens in that 100-step loop:

Loop 1: The AI calculates it needs to take a step of size 2. It takes the step.

Loop 2: The AI calculates a new step of size 3. PyTorch adds this to the old one (2 + 3 = 5). It takes a step of size 5.

Loop 100: By the 100th loop, the AI is adding up the gradients from all 99 previous loops. Instead of taking a careful, calculated step down the mountain, it takes a massive, chaotic leap across the entire loss landscape.

Within a few seconds, the numbers become so catastrophically large that your computer's memory maxes out, and all the weights in your neural network turn into NaN (Not a Number). The AI's brain essentially fries itself.

Wiping the memory clean before every single step ensures she only learns from the current batch of data.
'''