import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.llama3.reference_impl.model import Transformer
from models.llama3.reference_impl.sparse_autoencoder import SparseAutoencoder
from models.llama3.api.args import ModelArgs
from models.llama3.api.tokenizer import Tokenizer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Load the tokenizer
tokenizer = Tokenizer.get_instance()

# Initialize model arguments
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

# Load the model
model = Transformer(model_args)
model.to(device)
model.eval()

# Load model weights
checkpoint = torch.load('/mnt/home/rzhang/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth', map_location=device)
model.load_state_dict(checkpoint)
print("Model weights loaded successfully.")

# Prepare prompts
prompts = [
    "What is the capital of France?",
    "Explain quantum mechanics in simple terms.",
    "Write a short story about a dragon.",
    "What are the symptoms of diabetes?",
    "Translate 'hello' to Arabic.",
]

# Tokenize prompts
tokenized_prompts = [tokenizer.encode(prompt, bos=True, eos=True) for prompt in prompts]
for i, (prompt, tokens) in enumerate(zip(prompts, tokenized_prompts)):
    print(f"Prompt {i}: {prompt}")
    print(f"Tokenized Prompt {i}: {tokens}\n")
max_len = max(len(tokens) for tokens in tokenized_prompts)
padded_prompts = [tokens + [0] * (max_len - len(tokens)) for tokens in tokenized_prompts]
input_ids = torch.tensor(padded_prompts).to(device)
print(f"Padded input shape: {input_ids.shape}")

# Dictionary to store activations
activations_dict = {}
with torch.no_grad():
    _ = model(
        tokens=input_ids,
        start_pos=0,
        collect_activations=True,
        activations_dict=activations_dict,
        target_layer_ids=[10],  # Replace with the layer IDs you're interested in
    )
for layer, activations in activations_dict.items():
    print(f"Collected activations from {layer}: Shape {activations.shape}")
torch.save(activations_dict, 'activations_dict.pt')
torch.save(tokenized_prompts, 'tokenized_prompts.pt')

# Prepare activations tensor
activations_tensor = activations_dict['layer_10']
print(f"Activations tensor shape: {activations_tensor.shape}")  # (batch_size, seq_len, dim)

# Flatten activations for training
activations_tensor_flat = activations_tensor.view(-1, activations_tensor.shape[-1])
print(f"Activations tensor shape after flattening: {activations_tensor_flat.shape}")

# Define the sparse autoencoder
input_size = activations_tensor_flat.shape[1]
hidden_size = 128  # Choose the size for the latent space
autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
print(f"Initialized Sparse Autoencoder with input size: {input_size}, hidden size: {hidden_size}")

# Create dataset and dataloader
dataset = TensorDataset(activations_tensor_flat)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
sparsity_lambda = 1e-3  # Adjust sparsity penalty

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
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

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, obtain encoded features per prompt
encoded_features_per_prompt = []

with torch.no_grad():
    for i in range(activations_tensor.shape[0]):  # For each prompt
        activations = activations_tensor[i].to(device)  # (seq_len, dim)
        encoded = autoencoder.encoder(activations)      # (seq_len, hidden_size)
        encoded_features_per_prompt.append(encoded.cpu())

torch.save(encoded_features_per_prompt, 'encoded_features_per_prompt.pt')
print("Encoded features per prompt saved.")
