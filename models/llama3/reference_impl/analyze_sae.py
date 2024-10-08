import torch
import matplotlib.pyplot as plt
from collections import Counter

# Load encoded features per prompt
encoded_features_per_prompt = torch.load('encoded_features_per_prompt.pt')

# Define threshold for feature activation
threshold = 0.1  # Adjust based on your data distribution

feature_counts = Counter()

for encoded in encoded_features_per_prompt:
    # encoded is a tensor of shape (seq_len, hidden_size)
    # Determine which features are activated in this prompt
    # A feature is considered activated if any of its values exceed the threshold

    # Compute the maximum absolute value for each feature across the sequence
    max_abs_values = torch.max(torch.abs(encoded), dim=0)[0]

    # Identify activated features
    activated_features = (max_abs_values > threshold).nonzero(as_tuple=True)[0].tolist()

    # Update feature counts
    for feature_idx in activated_features:
        feature_counts[feature_idx] += 1

# Sort features by index for consistent plotting
sorted_features = sorted(feature_counts.keys())
sorted_counts = [feature_counts[feature] for feature in sorted_features]

# Plot the histogram
plt.figure(figsize=(12, 6))
plt.bar(sorted_features, sorted_counts, color='skyblue')
plt.xlabel('Feature Index')
plt.ylabel('Number of Prompts with Activated Feature')
plt.title('Feature Activation across Prompts')
plt.xticks(sorted_features)
plt.savefig('feature_activation_histogram.png')