import torch
import torch.nn.functional as F

# Let say our AI use 3D vector as given: [tech_level, danger_level, organixc_level]
suit_vector = torch.tensor([[0.9, 0.8, 0.1]])
iron_vector = torch.tensor([[0.8, 0.9, 0.1]])
apple_vector = torch.tensor([[0.1, 0.1, 0.9]])

# cosine similarity
cosine1 = F.cosine_similarity(suit_vector, iron_vector)
print(f"Similarity between words: {cosine1.item():.4f}")
cosine2 = F.cosine_similarity(iron_vector, apple_vector)
print(f"Similarity between words: {cosine2.item():.4f}")