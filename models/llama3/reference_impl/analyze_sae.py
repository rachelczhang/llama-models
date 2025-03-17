import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.llama3.reference_impl.model import Transformer
from models.llama3.reference_impl.sparse_autoencoder import SparseAutoencoder
from models.llama3.api.args import ModelArgs
from models.llama3.api.tokenizer import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login
import gc
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import wandb
import torch.nn.functional as F
from models.llama3.reference_impl.run_sae import load_model_and_tokenizer, load_dataset_from_wikipedia, tokenize_data

# def get_activations_and_tokens(model, tokenizer, tokenized_data, batch_size=5):
#     # Collect activations
#     all_activations = []
#     all_tokens = []
    
#     for i in range(0, tokenized_data.shape[0], batch_size):
#         batch_end = min(i + batch_size, tokenized_data.shape[0])
#         batch = tokenized_data[i:batch_end]
        
#         activations_dict = {}
#         def hook_fn(module, input, output):
#             activations_dict['layer_10_w2'] = output.clone().detach()
        
#         layer_to_hook = model.layers[10].feed_forward.w2
#         hook = layer_to_hook.register_forward_hook(hook_fn)
        
#         with torch.no_grad():
#             _ = model(tokens=batch, start_pos=0)
        
#         hook.remove()
        
#         all_activations.append(activations_dict['layer_10_w2'])
#         all_tokens.extend([tokenizer.decode([token.item()]) for token in batch.flatten()])

#     activations = torch.cat(all_activations, dim=0)
#     activations = activations.reshape(-1, activations.shape[-1])
    
#     return activations, all_tokens

def get_activations_and_tokens(model, tokenizer, tokenized_data, batch_size=5):
    # Collect activations and preserve structure
    all_activations = []
    all_token_texts = []
    batch_mapping = []  # To map flattened indices back to (batch_idx, pos_idx)
    
    for i in range(0, tokenized_data.shape[0], batch_size):
        batch_end = min(i + batch_size, tokenized_data.shape[0])
        batch = tokenized_data[i:batch_end]
        
        activations_dict = {}
        def hook_fn(module, input, output):
            activations_dict['layer_10_w2'] = output.clone().detach()
        
        layer_to_hook = model.layers[10].feed_forward.w2
        hook = layer_to_hook.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = model(tokens=batch, start_pos=0)
        
        hook.remove()
        
        # Store activations
        all_activations.append(activations_dict['layer_10_w2'])
        
        # Store token texts but preserve batch structure
        batch_tokens = []
        for seq_idx in range(batch.shape[0]):
            seq_tokens = [tokenizer.decode([token.item()]) for token in batch[seq_idx]]
            batch_tokens.append(seq_tokens)
        all_token_texts.append(batch_tokens)
        
        # Create mapping to reconstruct batch/sequence position
        for batch_idx in range(batch.shape[0]):
            for pos_idx in range(batch.shape[1]):
                flat_idx = len(batch_mapping)
                batch_mapping.append((i + batch_idx, pos_idx))
    
    # Stack activations but preserve structure
    activations = torch.cat(all_activations, dim=0)
    
    # Flatten activations for autoencoder
    flat_activations = activations.reshape(-1, activations.shape[-1])
    
    # Flatten token texts for simple lookup
    flat_tokens = []
    for batch in all_token_texts:
        for seq in batch:
            flat_tokens.extend(seq)
    
    return flat_activations, flat_tokens, tokenized_data, batch_mapping

def calculate_loss_ratio(model, tokenized_data, autoencoder, device, batch_size=5):
    """
    Calculate the loss ratio to evaluate autoencoder performance.
    
    Loss Ratio = (L_zero_ablated - L_approximation) / (L_zero_ablated - L_original)
    """
    losses = {'zero_ablated': 0, 'approximation': 0, 'original': 0}
    num_batches = 0
    
    for i in range(0, tokenized_data.shape[0], batch_size):
        batch_end = min(i + batch_size, tokenized_data.shape[0])
        batch = tokenized_data[i:batch_end]
        
        # Store original activations
        original_activations = {}
        def store_hook(module, input, output):
            original_activations['layer_10_w2'] = output.clone()
        
        # Hook for capturing/modifying activations
        def modify_hook(new_activations):
            def hook(module, input, output):
                return new_activations
            return hook
        
        layer = model.layers[10].feed_forward.w2
        
        # Calculate original loss
        with torch.no_grad():
            store_handle = layer.register_forward_hook(store_hook)
            output = model(tokens=batch, start_pos=0)
            # Get logits from model output
            if isinstance(output, tuple):
                logits = output[0]
            elif hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            # Ensure contiguous memory layout and proper reshaping
            logits = logits[:, :-1, :].contiguous()
            targets = batch[:, 1:].contiguous()
            
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
            losses['original'] += loss.item()
            store_handle.remove()
        
        # Calculate zero-ablated loss
        with torch.no_grad():
            zero_activations = torch.zeros_like(original_activations['layer_10_w2'])
            zero_handle = layer.register_forward_hook(modify_hook(zero_activations))
            output = model(tokens=batch, start_pos=0)
            if isinstance(output, tuple):
                logits = output[0]
            elif hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            logits = logits[:, :-1, :].contiguous()
            targets = batch[:, 1:].contiguous()
            
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
            losses['zero_ablated'] += loss.item()
            zero_handle.remove()
        
        # Calculate approximation loss
        with torch.no_grad():
            flat_activations = original_activations['layer_10_w2'].reshape(-1, original_activations['layer_10_w2'].shape[-1])
            reconstructed, _ = autoencoder(flat_activations)
            approximated_activations = reconstructed.reshape(original_activations['layer_10_w2'].shape)
            
            approx_handle = layer.register_forward_hook(modify_hook(approximated_activations))
            output = model(tokens=batch, start_pos=0)
            if isinstance(output, tuple):
                logits = output[0]
            elif hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            logits = logits[:, :-1, :].contiguous()
            targets = batch[:, 1:].contiguous()
            
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
            losses['approximation'] += loss.item()
            approx_handle.remove()
        
        num_batches += 1
        print(f"Processed batch {num_batches}")
    
    # Average losses
    for key in losses:
        losses[key] /= num_batches
    
    # Calculate loss ratio
    # mlp_contribution = losses['zero_ablated'] - losses['original']
    # autoencoder_contribution = losses['zero_ablated'] - losses['approximation']
    # loss_ratio = (autoencoder_contribution / mlp_contribution) * 100
    mlp_contribution = losses['zero_ablated'] - losses['original']
    autoencoder_contribution = losses['zero_ablated'] - losses['approximation']
    epsilon = 1e-6  # small value to prevent precision issues
    loss_ratio = min((autoencoder_contribution / (mlp_contribution + epsilon)) * 100, 100.0)
    
    print("\nLoss Analysis:")
    print(f"Original Loss: {losses['original']:.4f}")
    print(f"Zero-ablated Loss: {losses['zero_ablated']:.4f}")
    print(f"Approximation Loss: {losses['approximation']:.4f}")
    print(f"MLP's contribution to loss reduction: {mlp_contribution:.4f}")
    print(f"Autoencoder's contribution to loss reduction: {autoencoder_contribution:.4f}")
    print(f"\nLoss Ratio: {loss_ratio:.2f}% of MLP's contribution captured by autoencoder")
    
    # Add these diagnostic prints in the approximation loss section
    print("Activation stats:")
    print(f"Original - mean: {original_activations['layer_10_w2'].mean():.4f}, std: {original_activations['layer_10_w2'].std():.4f}")
    print(f"Reconstructed - mean: {approximated_activations.mean():.4f}, std: {approximated_activations.std():.4f}")
    print(f"Difference - mean: {(original_activations['layer_10_w2'] - approximated_activations).abs().mean():.4f}")

    print("\nDetailed Analysis:")
    print(f"Loss differences:")
    print(f"Zero vs Original: {losses['zero_ablated'] - losses['original']:.6f}")
    print(f"Zero vs Approximation: {losses['zero_ablated'] - losses['approximation']:.6f}")
    print(f"Approximation vs Original: {losses['approximation'] - losses['original']:.6f}")
    
    # Add this before the approximation loss calculation
    print("\nActivation Analysis:")
    flat_orig = original_activations['layer_10_w2'].reshape(-1, original_activations['layer_10_w2'].shape[-1])
    print(f"Original range: [{flat_orig.min():.4f}, {flat_orig.max():.4f}]")
    print(f"Original 5th/95th percentiles: [{torch.quantile(flat_orig.flatten(), 0.05):.4f}, {torch.quantile(flat_orig.flatten(), 0.95):.4f}]")

    # After reconstruction but before loss calculation
    print(f"Reconstruction range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    print(f"Reconstruction 5th/95th percentiles: [{torch.quantile(reconstructed.flatten(), 0.05):.4f}, {torch.quantile(reconstructed.flatten(), 0.95):.4f}]")

    # Compute per-position differences
    diff = (flat_orig - reconstructed).abs()
    print(f"Max absolute difference at any position: {diff.max():.4f}")
    print(f"99th percentile of absolute differences: {torch.quantile(diff.flatten(), 0.99):.4f}")
    return loss_ratio, losses

def plot_activation_frequencies_comparison(transformer_activations, autoencoder, device):
    """
    Creates a histogram comparing neuron activation frequencies between transformer layer
    and sparse autoencoder hidden layer activations.
    
    Args:
        transformer_activations: Flattened tensor of transformer activations (batch_size * seq_len, hidden_dim)
        autoencoder: Trained sparse autoencoder model
        device: Device to run computations on
    """
    # Get transformer activations
    transformer_acts = transformer_activations.cpu().numpy()
    transformer_frequencies = (transformer_acts > 1e-12).mean(axis=0) * 100  # Convert to percentage
    
    # Get autoencoder hidden layer activations
    with torch.no_grad():
        # Forward pass through encoder only to get hidden representations
        _, encoded = autoencoder(transformer_activations)
        autoencoder_acts = encoded.cpu().numpy()
        autoencoder_frequencies = (autoencoder_acts > 1e-12).mean(axis=0) * 100
    
    # Create figure with log-scale x-axis
    plt.figure(figsize=(12, 8))
    
    # Calculate histogram bins uniformly in log space
    min_freq = min(transformer_frequencies.min(), autoencoder_frequencies.min())
    max_freq = max(transformer_frequencies.max(), autoencoder_frequencies.max())
    bins = np.logspace(np.log10(max(min_freq, 1e-4)), np.log10(max_freq), 50)  # Uniform bins in log space
    
    # Plot histograms without density normalization
    plt.hist(transformer_frequencies, bins=bins, alpha=0.6, label='Transformer Neuron Activation Densities',
             color='blue', density=False)
    plt.hist(autoencoder_frequencies, bins=bins, alpha=0.6, label='Autoencoder Feature Activation Densities',
             color='red', density=False)
    
    # Customize plot
    plt.xscale('log')  # Set x-axis to log scale
    plt.xlabel('Activation Frequency')
    plt.ylabel('Number of Neurons')
    plt.title('Activation Densities')
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add summary statistics as text
    trans_stats = f'Transformer Stats:\nMean: {transformer_frequencies.mean():.1f}%\n'
    trans_stats += f'Median: {np.median(transformer_frequencies):.1f}%\n'
    trans_stats += f'Neurons: {len(transformer_frequencies)}'
    
    ae_stats = f'Autoencoder Stats:\nMean: {autoencoder_frequencies.mean():.1f}%\n'
    ae_stats += f'Median: {np.median(autoencoder_frequencies):.1f}%\n'
    ae_stats += f'Neurons: {len(autoencoder_frequencies)}'
    
    # Position stats text boxes
    plt.text(0.02, 0.98, trans_stats, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.75, 0.80, ae_stats, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("activation_frequencies_comparison.png")
    
    # Print additional statistics
    print("\nDetailed Statistics:")
    print("\nTransformer Activations:")
    print(f"Minimum: {transformer_frequencies.min():.1f}%")
    print(f"Maximum: {transformer_frequencies.max():.1f}%")
    print(f"Standard Deviation: {transformer_frequencies.std():.1f}%")
    
    print("\nAutoencoder Activations:")
    print(f"Minimum: {autoencoder_frequencies.min():.1f}%")
    print(f"Maximum: {autoencoder_frequencies.max():.1f}%")
    print(f"Standard Deviation: {autoencoder_frequencies.std():.1f}%")
    
    # Calculate and print percentile information
    percentiles = [10, 25, 50, 75, 90]
    print("\nPercentile Comparison:")
    print(f"{'Percentile':>10} {'Transformer':>12} {'Autoencoder':>12}")
    print("-" * 35)
    for p in percentiles:
        t_perc = np.percentile(transformer_frequencies, p)
        a_perc = np.percentile(autoencoder_frequencies, p)
        print(f"{p:>10}th {t_perc:>12.1f}% {a_perc:>12.1f}%")

    return transformer_frequencies, autoencoder_frequencies

# def analyze_feature_activations(autoencoder, activations, tokens, top_k=8):
#     """Analyze which tokens cause each feature to activate most strongly."""
#     with torch.no_grad():
#         # Get feature activations
#         _, feature_activations = autoencoder(activations)
#         feature_activations = feature_activations.cpu().numpy()
        
#     num_features = feature_activations.shape[1]
#     feature_analysis = []
    
#     for feature_idx in range(num_features):
#         # Get activations for this feature
#         feature_values = feature_activations[:, feature_idx]
        
#         # Get top k activating tokens
#         top_indices = np.argsort(feature_values)[-top_k:][::-1]
#         top_tokens = [tokens[idx] for idx in top_indices]
#         top_activations = [feature_values[idx] for idx in top_indices]
        
#         feature_analysis.append({
#             'feature_id': feature_idx,
#             'top_tokens': top_tokens,
#             'top_activations': top_activations
#         })
    
#     return feature_analysis

def analyze_feature_activations(autoencoder, activations, tokens, tokenized_data, batch_mapping, top_k=8, context_window=5):
    """Analyze which tokens cause each feature to activate most strongly, with context."""
    with torch.no_grad():
        # Get feature activations
        _, feature_activations = autoencoder(activations)
        feature_activations = feature_activations.cpu().numpy()
        
    num_features = feature_activations.shape[1]
    feature_analysis = []
    
    # Create tokenizer for context extraction (adjust this based on your tokenizer)
    def get_context(batch_idx, pos_idx, window=context_window):
        # Get the sequence from tokenized_data
        sequence = tokenized_data[batch_idx].cpu().tolist()
        
        # Extract context
        start_idx = max(0, pos_idx - window)
        end_idx = min(len(sequence), pos_idx + window + 1)
        context_ids = sequence[start_idx:end_idx]
        
        # Convert to tokens
        context_tokens = [tokenizer.decode([token_id]) if isinstance(token_id, (int, float)) else "<unknown>" 
                 for token_id in context_ids]
        target_pos = pos_idx - start_idx
        
        return {
            'context_tokens': context_tokens,
            'target_position': target_pos
        }
    
    for feature_idx in range(num_features):
        # Get activations for this feature
        feature_values = feature_activations[:, feature_idx]
        
        # Get top k activating positions
        top_indices = np.argsort(feature_values)[-top_k:][::-1]
        
        top_tokens = []
        top_contexts = []
        top_positions = []
        top_activations = []
        
        for idx in top_indices:
            # Get the batch and position from mapping
            if idx < len(batch_mapping):
                batch_idx, pos_idx = batch_mapping[idx]
                
                # Get token
                if idx < len(tokens):
                    token = tokens[idx]
                else:
                    token = "<unknown>"
                
                # Get context
                context = get_context(batch_idx, pos_idx, context_window)
                
                top_tokens.append(token)
                top_contexts.append(context)
                top_positions.append((batch_idx, pos_idx))
                top_activations.append(feature_values[idx])
        
        feature_analysis.append({
            'feature_id': feature_idx,
            'top_tokens': top_tokens,
            'top_contexts': top_contexts,
            'top_positions': top_positions,
            'top_activations': top_activations
        })
    
    return feature_analysis

# def create_feature_activation_table(feature_analysis, output_file='feature_analysis.csv'):
#     """Create and save a table of feature activations."""
#     rows = []
#     for feature in feature_analysis:
#         for token, activation in zip(feature['top_tokens'], feature['top_activations']):
#             rows.append({
#                 'Feature ID': feature['feature_id'],
#                 'Token': token,
#                 'Activation': activation
#             })
    
#     df = pd.DataFrame(rows)
#     df.to_csv(output_file, index=False)
#     return df

def create_feature_activation_table(feature_analysis, output_file='feature_analysis.csv'):
    """Create and save a table of feature activations with context."""
    rows = []
    for feature in feature_analysis:
        for i in range(len(feature['top_tokens'])):
            token = feature['top_tokens'][i]
            activation = feature['top_activations'][i]
            context = feature['top_contexts'][i]
            position = feature['top_positions'][i]
            
            # Format the context as a string, highlighting the target token
            context_str = ""
            if context:
                context_tokens = context['context_tokens']
                target_pos = context['target_position']
                
                formatted_context = []
                for j, ctx_token in enumerate(context_tokens):
                    if j == target_pos:
                        formatted_context.append(f"[{ctx_token}]")  # Highlight the target token
                    else:
                        formatted_context.append(ctx_token)
                
                context_str = " ".join(formatted_context)
            
            rows.append({
                'Feature ID': feature['feature_id'],
                'Token': token,
                'Context': context_str,
                'Batch_Idx': position[0],
                'Position_Idx': position[1],
                'Activation': activation
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    return df
    
if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(device)
    print("Model and tokenizer loaded successfully")

    data = load_dataset_from_wikipedia()
    tokenized_data = tokenize_data(data, tokenizer, device)

    # activations, tokens = get_activations_and_tokens(model, tokenizer, tokenized_data)
    
    # Get activations and preserving structure
    activations, tokens, tokenized_data, batch_mapping = get_activations_and_tokens(model, tokenizer, tokenized_data)
    print(f"Collected activations shape: {activations.shape}")

    # Load trained autoencoder
    input_size = activations.shape[1]
    hidden_size = 4096
    autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
    autoencoder.load_state_dict(torch.load('best_llm_sae_stellar-snowball-12.pth', weights_only=True))
    autoencoder.eval()
    print("Loaded trained autoencoder")

    # # Calculate loss ratio
    # loss_ratio, losses = calculate_loss_ratio(model, tokenized_data, autoencoder, device)

    # # Flatten activations
    # activations_tensor_flat = activations.view(-1, activations.shape[-1])

    # # Plot activation frequencies comparison
    # transformer_frequencies, autoencoder_frequencies = plot_activation_frequencies_comparison(activations_tensor_flat, autoencoder, device)

    # # Analyze features
    # feature_analysis = analyze_feature_activations(autoencoder, activations, tokens)
    # print("Analyzed feature activations")

    # Analyze feature activations with context
    feature_analysis = analyze_feature_activations(
        autoencoder, 
        activations, 
        tokens, 
        tokenized_data, 
        batch_mapping,
        top_k=8, 
        context_window=5
    )
    
    # Create and save analysis table
    df = create_feature_activation_table(feature_analysis)
    print(f"Saved feature analysis to feature_analysis.csv")

# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# import numpy as np 
# from models.llama3.api.tokenizer import Tokenizer

# # Load encoded features per prompt
# encoded_features_per_prompt = torch.load('encoded_features_per_prompt.pt')

# # Convert the encoded features to a matrix where each row is a prompt and each column is a feature
# feature_matrix = torch.stack([encoded.max(dim=0)[0] for encoded in encoded_features_per_prompt])

# plt.figure(figsize=(12, 8))
# sns.heatmap(feature_matrix, cmap='viridis', xticklabels=False)
# plt.xlabel('Feature Index')
# plt.ylabel('Prompt Index')
# plt.title('Feature Activation Heatmap')
# plt.savefig('feature_activation_heatmap.png')

# # Determine the number of prompts in the feature matrix
# n_samples = feature_matrix.shape[0]
# # Set the perplexity to be smaller than the number of prompts
# perplexity = min(30, n_samples - 1)
# # Apply t-SNE with adjusted perplexity
# tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
# reduced_features = tsne.fit_transform(feature_matrix)

# # Plot the reduced features
# plt.figure(figsize=(10, 6))
# plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.title('t-SNE Visualization of Prompts in Feature Space')
# plt.savefig('tsne_visualization.png')

# # Plot histograms for a subset of features 
# num_features_to_plot = 10
# plt.figure(figsize=(15, 10))
# for i in range(num_features_to_plot):
#     plt.subplot(2, 5, i + 1)
#     plt.hist(feature_matrix[:, i], bins=20, color='skyblue')
#     plt.title(f'Feature {i} Activation')
#     plt.xlabel('Activation Level')
#     plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('feature_activation_histograms.png')

# # Plot activations for a specific feature over token positions for a specific prompt
# feature_index = 5
# prompt_index = 2
# activations = encoded_features_per_prompt[prompt_index].numpy()[:, feature_index]

# plt.figure(figsize=(8, 5))
# plt.plot(activations)
# plt.xlabel('Token Position')
# plt.ylabel('Feature Activation')
# plt.title(f'Feature {feature_index} Activation Over Token Positions (Prompt {prompt_index})')
# plt.savefig('feature_activation_over_token_positions.png')

# tokenized_prompts = torch.load('tokenized_prompts.pt') 
# tokenizer = Tokenizer.get_instance()  

# feature_index = 1
# # Define a threshold for high activation
# high_activation_threshold = 0  

# # Identify prompts where feature has high activation
# high_activation_prompts = []
# for idx, encoded in enumerate(encoded_features_per_prompt):
#     # encoded is a tensor of shape (seq_len, hidden_size)
#     # Extract feature across all tokens
#     feature_activations = encoded[:, feature_index].cpu().numpy()
#     max_token_activation = torch.max(torch.abs(encoded[:, feature_index])).item()
#     print(f"Prompt {idx}, Max Activation for Feature {feature_index}: {max_token_activation}")
#     # Print all activations for the prompt for feature
#     print(f"All activations for feature {feature_index} in prompt {idx}: {feature_activations}")
#     # Check if any token in the prompt has a high activation for feature
#     if (feature_activations > high_activation_threshold).any():
#         high_activation_prompts.append((idx, feature_activations))

# print(f'High activation prompts for feature {feature_index}:', high_activation_prompts)
# # Examine tokens associated with high activation
# for idx, activations in high_activation_prompts:
#     print(f"Analyzing Prompt {idx}:")
#     print(f"Undecoded Prompt: {tokenized_prompts[idx]}")
#     print(f"Original Prompt: {tokenizer.decode(tokenized_prompts[idx])}")
#     # Iterate through the token IDs and match with activations directly
#     token_ids = tokenized_prompts[idx]
#     if len(token_ids) != len(activations):
#         print(f"Warning: Number of tokens ({len(token_ids)}) does not match the number of activation values ({len(activations)}).")
#     high_activation_tokens = []
#     for i, token_id in enumerate(token_ids):
#         token = tokenizer.decode([token_id])
#         activation_value = activations[i] if i < len(activations) else None
#         if activation_value > 0:
#             high_activation_tokens.append((i, token, activation_value))

#     print(f"Tokens with high activation for feature {feature_index}:")
#     for position, token, activation_value in high_activation_tokens:
#         print(f"Token: '{token}' at position {position} with activation value: {activation_value}")
