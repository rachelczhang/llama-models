import torch
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np 
from models.llama3.api.tokenizer import Tokenizer

# Load encoded features per prompt
encoded_features_per_prompt = torch.load('encoded_features_per_prompt.pt')

# Convert the encoded features to a matrix where each row is a prompt and each column is a feature
feature_matrix = torch.stack([encoded.max(dim=0)[0] for encoded in encoded_features_per_prompt])

plt.figure(figsize=(12, 8))
sns.heatmap(feature_matrix, cmap='viridis', xticklabels=False)
plt.xlabel('Feature Index')
plt.ylabel('Prompt Index')
plt.title('Feature Activation Heatmap')
plt.savefig('feature_activation_heatmap.png')

# Determine the number of prompts in the feature matrix
n_samples = feature_matrix.shape[0]
# Set the perplexity to be smaller than the number of prompts
perplexity = min(30, n_samples - 1)  

# Apply t-SNE with adjusted perplexity
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
reduced_features = tsne.fit_transform(feature_matrix)

# Plot the reduced features
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of Prompts in Feature Space')
plt.savefig('tsne_visualization.png')

# Plot histograms for a subset of features 
num_features_to_plot = 10
plt.figure(figsize=(15, 10))
for i in range(num_features_to_plot):
    plt.subplot(2, 5, i + 1)
    plt.hist(feature_matrix[:, i], bins=20, color='skyblue')
    plt.title(f'Feature {i} Activation')
    plt.xlabel('Activation Level')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('feature_activation_histograms.png')

# Plot activations for a specific feature over token positions for a specific prompt
feature_index = 5
prompt_index = 2
activations = encoded_features_per_prompt[prompt_index].numpy()[:, feature_index]

plt.figure(figsize=(8, 5))
plt.plot(activations)
plt.xlabel('Token Position')
plt.ylabel('Feature Activation')
plt.title(f'Feature {feature_index} Activation Over Token Positions (Prompt {prompt_index})')
plt.savefig('feature_activation_over_token_positions.png')

tokenized_prompts = torch.load('tokenized_prompts.pt') 
tokenizer = Tokenizer.get_instance()  

feature_index = 1
# Define a threshold for high activation
high_activation_threshold = 0  

# Identify prompts where feature has high activation
high_activation_prompts = []
for idx, encoded in enumerate(encoded_features_per_prompt):
    # encoded is a tensor of shape (seq_len, hidden_size)
    # Extract feature across all tokens
    feature_activations = encoded[:, feature_index].cpu().numpy()
    max_token_activation = torch.max(torch.abs(encoded[:, feature_index])).item()
    print(f"Prompt {idx}, Max Activation for Feature {feature_index}: {max_token_activation}")
    # Print all activations for the prompt for feature
    print(f"All activations for feature {feature_index} in prompt {idx}: {feature_activations}")
    # Check if any token in the prompt has a high activation for feature
    if (feature_activations > high_activation_threshold).any():
        high_activation_prompts.append((idx, feature_activations))

print(f'High activation prompts for feature {feature_index}:', high_activation_prompts)
# Examine tokens associated with high activation
for idx, activations in high_activation_prompts:
    print(f"Analyzing Prompt {idx}:")
    print(f"Undecoded Prompt: {tokenized_prompts[idx]}")
    print(f"Original Prompt: {tokenizer.decode(tokenized_prompts[idx])}")
    # Iterate through the token IDs and match with activations directly
    token_ids = tokenized_prompts[idx]
    if len(token_ids) != len(activations):
        print(f"Warning: Number of tokens ({len(token_ids)}) does not match the number of activation values ({len(activations)}).")
    high_activation_tokens = []
    for i, token_id in enumerate(token_ids):
        token = tokenizer.decode([token_id])
        activation_value = activations[i] if i < len(activations) else None
        if activation_value > 0:
            high_activation_tokens.append((i, token, activation_value))

    print(f"Tokens with high activation for feature {feature_index}:")
    for position, token, activation_value in high_activation_tokens:
        print(f"Token: '{token}' at position {position} with activation value: {activation_value}")
