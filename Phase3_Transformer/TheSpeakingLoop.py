import torch
import torch.nn.functional as F

def generate_text(model, initial_tokens, max_length=20, temperature=0.8):
    # Copy the input so  we can add to it
    current_sequence = initial_tokens.clone()

    for _ in range(max_length):
        # 1. Pass the current sequence to the brain
        # We only care about logits for the very last word in the sequence
        logits = model(current_sequence)[:, -1, :]

        # 2. Apply the temperature
        scaled_logits = logits / temperature

        # 3. Convert the probabilites
        probabilities = F.softmax(scaled_logits, dim=-1)

        # 4. Sample the next word (PyTorch's mulitnomial function does this job based on the probabilites)
        next_word_id = torch.multinomial(probabilities, num_samples=1)

        # 5. Append to the current sequence
        current_sequence = torch.cat([current_sequence, next_word_id], dim=1)

        # Stop if she generated special "END OF SENTENCE" token (ID = 0)
        if next_word_id.item() == 0:
            break

    return current_sequence

# Example usage (Pseudocode since we haven't trained a full model)
# generated_ids = generate_text(model, input_ids)
# print(decode_to_english(generated_ids))