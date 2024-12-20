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
import matplotlib.pyplot as plt
import numpy as np 
import wandb
import torch.nn.functional as F

wandb.init(project="test_sae_on_LLM", entity="rczhang")

def load_model_and_tokenizer(device):
    # Load the tokenizer
    tokenizer = Tokenizer.get_instance()

    # Initialize LlaMa model arguments and load model
    model_args = ModelArgs(
        max_batch_size=5,
        max_seq_len=2048,
        dim=2048,
        n_layers=16,
        n_heads=32,
        vocab_size=128256,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        n_kv_heads=8,
        norm_eps=1e-5,
        rope_theta=500000.0,
        use_scaled_rope=True
    )

    model = Transformer(model_args)
    model.to(device)
    model.eval()

    # Load trained model weights
    checkpoint = torch.load('/mnt/home/rzhang/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth', 
                    map_location=device,
                    weights_only=True)
    model.load_state_dict(checkpoint)
    print('model', model)
    print("Model weights loaded successfully.")
    return model, tokenizer

def load_dataset_from_wikipedia():
    num_samples = 200
    max_chars = 3000
    data = []
    
    print("Attempting to load dataset...")
    ds_iterator = iter(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True))
    # ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    # for i, sample in enumerate(ds):
    #     if i >= num_samples:
    #         break
    #     text = sample['text'][:max_chars]  # Truncate to 5000 characters
    #     data.append(text)
    for _ in range(num_samples):
        sample = next(ds_iterator)
        text = sample['text'][:max_chars]  # Truncate to 5000 characters
        data.append(text)

    # stop the iterator process from running in background
    del ds_iterator
    gc.collect()
        
    return data

def tokenize_data(data, tokenizer, device):
    tokenized_data = [tokenizer.encode(d, bos=True, eos=True) for d in data]
    # for i, (data, tokens) in enumerate(zip(data, tokenized_data)):
    #     print(f"data {i}: {data}")
    #     print(f"Tokenized data {i}: {tokens}\n")
    #     break
    max_len = max(len(tokens) for tokens in tokenized_data)
    pad_token_id = tokenizer.pad_id
    padded_data = [tokens + [pad_token_id] * (max_len - len(tokens)) for tokens in tokenized_data]
    tokenized_data = torch.tensor(padded_data).to(device)
    print(f"Padded input shape: {tokenized_data.shape}")
    # print('Padded data tokens: ', padded_data)
    return tokenized_data

def collect_activations(tokenized_data):
    activations_dict = {}
    # Hook function to capture activations
    def hook_fn(module, input, output):
        activations_dict['layer_10_w2'] = output.clone().detach()
    layer_to_hook = model.layers[10].feed_forward.w2
    hook = layer_to_hook.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(
            tokens=tokenized_data,
            start_pos=0
            # collect_activations=True,
            # activations_dict=activations_dict,
            # target_layer_ids=[10], 
        )
    hook.remove()
    # for layer, activations in activations_dict.items():
    #     print(f"Collected activations from {layer}: Shape {activations.shape}")
    print(f"Collected activations shape: {activations_dict['layer_10_w2'].shape}")
    # torch.save(activations_dict, 'activations_dict.pt')
    # Prepare activations tensor
    activations_tensor = activations_dict['layer_10_w2']
    print(f"Activations tensor shape: {activations_tensor.shape}")  # (batch_size, seq_len, dim)

    # Flatten activations for training
    activations_tensor_flat = activations_tensor.view(-1, activations_tensor.shape[-1])
    print(f"Activations tensor shape after flattening: {activations_tensor_flat.shape}")
    
    return activations_tensor_flat

def collect_activations_batched(model, tokenized_data, batch_size=5):
    """
    Collect activations in batches to avoid memory issues.
    
    Args:
        model: The transformer model
        tokenized_data: The tokenized input data
        batch_size: Size of each batch
    """
    all_activations = []
    num_samples = tokenized_data.shape[0]
    
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch = tokenized_data[i:batch_end]
        
        activations_dict = {}
        # Hook function to capture activations
        def hook_fn(module, input, output):
            activations_dict['layer_10_w2'] = output.clone().detach()
        
        layer_to_hook = model.layers[10].feed_forward.w2
        hook = layer_to_hook.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = model(
                tokens=batch,
                start_pos=0
            )
        
        hook.remove()
        
        # Get activations for this batch
        batch_activations = activations_dict['layer_10_w2']
        all_activations.append(batch_activations)
        # print('all activations', all_activations)
        
        print(f"Processed batch {i//batch_size + 1} of {(num_samples + batch_size - 1)//batch_size}")
    
    # Concatenate all batches
    all_activations = torch.cat(all_activations, dim=0)
    activations_tensor_flat = all_activations.view(-1, all_activations.shape[-1])
    print(f"Final activations tensor shape after flattening: {activations_tensor_flat.shape}")
    
    return activations_tensor_flat

def plot_activation_frequencies(activations_tensor_flat):
    """
    Creates a histogram of neuron activation frequencies from transformer layer activations.
    
    Args:
        activations_tensor_flat: Flattened tensor of shape (batch_size * seq_len, hidden_dim)
    """
    # Convert activations to numpy for easier processing
    activations = activations_tensor_flat.cpu().numpy()

    # Calculate the proportion of non-zero activations for each neuron
    # A neuron is considered "active" when its activation is non-zero
    neuron_activities = (activations > 1e-12).mean(axis=0) * 100  # Convert to percentage
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(neuron_activities, bins=50, edgecolor='black')
    plt.xlabel('Activation Frequency (%)')
    plt.ylabel('Number of Neurons')
    plt.title('Distribution of Neuron Activation Frequencies')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add some summary statistics as text
    stats_text = f'Mean: {neuron_activities.mean():.1f}%\n'
    stats_text += f'Median: {np.median(neuron_activities):.1f}%\n'
    stats_text += f'Total Neurons: {len(neuron_activities)}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("activation_frequencies.png")
    
    # Print some additional statistics
    print(f"\nActivation Frequency Statistics:")
    print(f"Minimum: {neuron_activities.min():.1f}%")
    print(f"Maximum: {neuron_activities.max():.1f}%")
    print(f"Standard Deviation: {neuron_activities.std():.1f}%")
    
    # Calculate and print percentile information
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        print(f"{p}th percentile: {np.percentile(neuron_activities, p):.1f}%")

def train_sae(activations_tensor_flat, autoencoder, device):
    # Create dataset and dataloader
    dataset = TensorDataset(activations_tensor_flat)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    sparsity_lambda = 1e-2 
    patience = 300
    best_loss = float('inf')
    epochs_without_improvement = 0

    num_epochs = 100000
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)

            # Forward pass
            reconstructed, encoded = autoencoder(batch)

            # Compute losses
            reconstruction_loss = criterion(reconstructed, batch)
            sparsity_loss = torch.mean(torch.abs(encoded))
            loss = reconstruction_loss + sparsity_lambda * sparsity_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder weights after optimization step
            autoencoder.normalize_decoder_weights()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
        wandb.log({"epoch": epoch+1, "loss": avg_loss, "reconstruction_loss": reconstruction_loss.item(), "sparsity_loss": sparsity_loss.item()})

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(autoencoder.state_dict(), f"best_llm_sae_{wandb.run.name}.pth")  # Save model parameters
            print(f"New best model saved with loss {best_loss:.6f}")
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"Stopping early after {epoch+1} epochs due to no improvement.")
            break

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(device)

    # Load and tokenize data
    print("Loading data")
    data = load_dataset_from_wikipedia()
    print("Tokenizing data")
    tokenized_data = tokenize_data(data, tokenizer, device)
    print("After tokenizing")

    # Collect activations for tokenized data
    activations_tensor_flat = collect_activations_batched(model, tokenized_data)

    # Plot activation frequencies
    plot_activation_frequencies(activations_tensor_flat)

    # Define the sparse autoencoder
    input_size = activations_tensor_flat.shape[1]
    hidden_size = 4096
    autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
    print(f"Initialized Sparse Autoencoder with input size: {input_size}, hidden size: {hidden_size}")

    # Train the sparse autoencoder
    train_sae(activations_tensor_flat, autoencoder, device)

    # # Load the best model weights for loss ratio calculation
    # best_model_path = f"best_llm_sae_{wandb.run.name}.pth"
    # print(f"Loading best model weights from {best_model_path}")
    # # Create a new autoencoder instance for evaluation
    # evaluation_autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
    # evaluation_autoencoder.load_state_dict(torch.load(best_model_path, weights_only=True))
    # evaluation_autoencoder.eval() 

    # # Calculate loss ratio
    # loss_ratio, losses = calculate_loss_ratio(model, tokenized_data, evaluation_autoencoder, device)

    # # Plot activation frequencies comparison
    # transformer_frequencies, autoencoder_frequencies = plot_activation_frequencies_comparison(activations_tensor_flat, evaluation_autoencoder, device)

# # After training, obtain encoded features per prompt
# encoded_features_per_prompt = []

# with torch.no_grad():
#     for i in range(activations_tensor.shape[0]):  # For each prompt
#         activations = activations_tensor[i].to(device)  # (seq_len, dim)
#         encoded = autoencoder.encoder(activations)      # (seq_len, hidden_size)
#         encoded_features_per_prompt.append(encoded.cpu())

# torch.save(encoded_features_per_prompt, 'encoded_features_per_prompt.pt')
# print("Encoded features per prompt saved.")
