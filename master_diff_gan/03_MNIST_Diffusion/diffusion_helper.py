import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from diffusion_model import *

from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


device =  'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

def generate_image_with_all_steps(noise_scheduler, net, label, img_shape=(1, 28, 28), device='cuda'):
    # Initialize random noise and label
    x = torch.randn(1, *img_shape).to(device)
    y = torch.tensor([label], device=device)

    # To store intermediate steps
    steps = []

    # Sampling loop
    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            residual = net(x, t, y)
        x = noise_scheduler.step(residual, t, x).prev_sample

        # Normalize current step for visualization and save
        x_clipped = x[0].clip(-1, 1)
        x_normalized = (x_clipped + 1) / 2  # Normalize to [0, 1]
        steps.append(x_normalized.cpu().numpy())  # Save as NumPy array

    # Final normalization
    final_image = steps[-1]  # The final denoised image
    return final_image, steps

def print_detailed_unet_structure(model):
    """
    Print complete structure of the ClassConditionedUnet model including all layers
    
    Parameters:
    -----------
    model : ClassConditionedUnet
        The UNet model to analyze
    """
    def print_block_details(block, indent=""):
        """Helper function to recursively print block details"""
        for name, layer in block.named_children():
            # Print layer name and type
            print(f"{indent}{name}: {layer.__class__.__name__}")
            
            # Print layer details based on type
            if isinstance(layer, nn.Conv2d):
                print(f"{indent}  in_channels: {layer.in_channels}")
                print(f"{indent}  out_channels: {layer.out_channels}")
                print(f"{indent}  kernel_size: {layer.kernel_size}")
                print(f"{indent}  stride: {layer.stride}")
                print(f"{indent}  padding: {layer.padding}")
            elif isinstance(layer, nn.Linear):
                print(f"{indent}  in_features: {layer.in_features}")
                print(f"{indent}  out_features: {layer.out_features}")
            elif isinstance(layer, nn.Embedding):
                print(f"{indent}  num_embeddings: {layer.num_embeddings}")
                print(f"{indent}  embedding_dim: {layer.embedding_dim}")
            
            # Recursively print child modules
            if len(list(layer.children())) > 0:
                print_block_details(layer, indent + "  ")
    
    print("\n=== Complete ClassConditionedUnet Structure ===")
    print("\nEmbedding Layer:")
    print_block_details(model.class_emb, "  ")
    
    print("\nUNet Model:")
    print("Input Processing:")
    print(f"  Sample Size: {model.model.sample_size}")
    print(f"  In Channels: {model.model.in_channels}")
    print(f"  Out Channels: {model.model.out_channels}")
    
    print("\nDown Blocks:")
    for i, block in enumerate(model.model.down_blocks):
        print(f"\nDown Block {i+1}: {type(block).__name__}")
        print_block_details(block, "  ")
    
    print("\nMiddle Block:")
    print_block_details(model.model.mid_block, "  ")
    
    print("\nUp Blocks:")
    for i, block in enumerate(model.model.up_blocks):
        print(f"\nUp Block {i+1}: {type(block).__name__}")
        print_block_details(block, "  ")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def collect_activations(model, save_dir='activations', device='cuda', batch_size=2000, 
                       save_inputs=False, save_outputs=False, 
                       save_first_timestep=False, save_last_timestep=False):
    """
    Collect activations in batches and save them as numbered files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset with specified batch size
    full_dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, 
                                            transform=torchvision.transforms.ToTensor())
    data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    total_samples = len(full_dataset)
    samples_processed = 0
    
    def hook_fn(module, input, output):
        if not hasattr(hook_fn, 'current_activations'):
            hook_fn.current_activations = []
        hook_fn.current_activations.append(output.detach().cpu().numpy())
    
    # Register hooks for middle block attention layers
    hooks = []
    if save_first_timestep or save_last_timestep:
        if hasattr(model.model.mid_block, 'attentions'):
            for attn in model.model.mid_block.attentions:
                hooks.append(attn.to_q.register_forward_hook(hook_fn))
                hooks.append(attn.to_k.register_forward_hook(hook_fn))
                hooks.append(attn.to_v.register_forward_hook(hook_fn))
                hooks.append(attn.to_out[0].register_forward_hook(hook_fn))
        print(f"Registered {len(hooks)} hooks for middle block attention layers")
    
    # Process data
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            current_batch = batch_idx + 1
            start_idx = samples_processed
            end_idx = start_idx + len(x)
            
            # Update progress
            samples_processed += len(x)
            print(f"Processing batch {current_batch}: samples {start_idx}-{end_idx} "
                  f"({samples_processed/total_samples*100:.1f}%)", end='\r')
            
            # Prepare input
            x = x.to(device) * 2 - 1
            y = y.to(device)
            
            # Save inputs and labels if requested
            if save_inputs:
                np.save(os.path.join(save_dir, f'inputs_{end_idx}.npy'), x.cpu().numpy())
                np.save(os.path.join(save_dir, f'labels_{end_idx}.npy'), y.cpu().numpy())
            
            # Process first timestep if requested
            if save_first_timestep:
                hook_fn.current_activations = []
                timesteps = torch.zeros(len(x), device=device, dtype=torch.long)
                _ = model(x, timesteps, y)
                
                if hook_fn.current_activations:
                    timestep_acts = []
                    for act in hook_fn.current_activations:
                        if len(act.shape) > 2:
                            act = act.reshape(act.shape[0], -1)
                        timestep_acts.append(act)
                    batch_activations = np.concatenate(timestep_acts, axis=1)
                    np.save(os.path.join(save_dir, f'activations_first_{end_idx}.npy'), 
                           batch_activations)
            
            # Process last timestep if requested
            if save_last_timestep:
                hook_fn.current_activations = []
                timesteps = torch.full((len(x),), 19, device=device, dtype=torch.long)
                outputs = model(x, timesteps, y)
                
                if hook_fn.current_activations:
                    timestep_acts = []
                    for act in hook_fn.current_activations:
                        if len(act.shape) > 2:
                            act = act.reshape(act.shape[0], -1)
                        timestep_acts.append(act)
                    batch_activations = np.concatenate(timestep_acts, axis=1)
                    np.save(os.path.join(save_dir, f'activations_last_{end_idx}.npy'), 
                           batch_activations)
                
                # Save final output if requested
                if save_outputs:
                    np.save(os.path.join(save_dir, f'outputs_{end_idx}.npy'), 
                           outputs.cpu().numpy())
            
            # Clear GPU memory
            torch.cuda.empty_cache()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print("\nFinished collecting activations!")
    print(f"Data saved to {save_dir}/ in chunks of {batch_size} samples")
    print("Files saved:")
    if save_inputs:
        print("- inputs_XXXX.npy: Input images")
        print("- labels_XXXX.npy: Input labels")
    if save_first_timestep:
        print("- activations_first_XXXX.npy: First timestep activations")
    if save_last_timestep:
        print("- activations_last_XXXX.npy: Last timestep activations")
    if save_outputs:
        print("- outputs_XXXX.npy: Final outputs")
    print("where XXXX represents the cumulative sample count")

def process_new_sample(model, sample, label, device='cuda', save_dir='unet_activations'):
    """
    Process a new sample through the model and return inputs, outputs and activations
    
    Parameters:
    -----------
    model : ClassConditionedUnet
        The trained UNet model
    sample : torch.Tensor
        Input image tensor of shape [1, 28, 28] or [28, 28]
    label : int
        Class label for the input
    device : str, default='cuda'
        Device to run inference on
    save_dir : str, default='unet_activations'
        Directory containing the activation files for comparison
    """
    # Prepare input - ensure correct shape [batch_size, channels, height, width]
    if sample.dim() == 2:  # [28, 28]
        x = sample.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif sample.dim() == 3:  # [1, 28, 28]
        x = sample.unsqueeze(0)  # Add batch dim
    elif sample.dim() == 4:  # [1, 1, 28, 28]
        x = sample
    else:
        raise ValueError(f"Invalid input shape: {sample.shape}")
        
    x = x.to(device) * 2 - 1  # Scale to [-1, 1]
    y = torch.tensor([label], device=device)
    
    # Rest of the function remains the same
    def hook_fn(module, input, output):
        if not hasattr(hook_fn, 'current_activations'):
            hook_fn.current_activations = []
        hook_fn.current_activations.append(output.detach().cpu().numpy())
    
    # Register hooks
    hooks = []
    if hasattr(model.model.mid_block, 'attentions'):
        for attn in model.model.mid_block.attentions:
            hooks.append(attn.to_q.register_forward_hook(hook_fn))
            hooks.append(attn.to_k.register_forward_hook(hook_fn))
            hooks.append(attn.to_v.register_forward_hook(hook_fn))
            hooks.append(attn.to_out[0].register_forward_hook(hook_fn))
    
    # Process sample
    model.eval()
    with torch.no_grad():
        # Store input
        input_data = x.cpu().numpy()
        
        # First timestep (t=0)
        hook_fn.current_activations = []
        timesteps = torch.zeros(1, device=device, dtype=torch.long)
        _ = model(x, timesteps, y)
        
        # Process first timestep activations
        first_timestep_acts = []
        for act in hook_fn.current_activations:
            if len(act.shape) > 2:
                act = act.reshape(act.shape[0], -1)
            first_timestep_acts.append(act)
        first_timestep = np.concatenate(first_timestep_acts, axis=1)
        
        # Last timestep (t=19)
        hook_fn.current_activations = []
        timesteps = torch.full((1,), 19, device=device, dtype=torch.long)
        outputs = model(x, timesteps, y)
        
        # Process last timestep activations
        last_timestep_acts = []
        for act in hook_fn.current_activations:
            if len(act.shape) > 2:
                act = act.reshape(act.shape[0], -1)
            last_timestep_acts.append(act)
        last_timestep = np.concatenate(last_timestep_acts, axis=1)
        
        # Store output
        output = outputs.cpu().numpy()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return {
        'input_data': input_data,
        'first_timestep': first_timestep,
        'last_timestep': last_timestep,
        'output': output
    }

def get_new_sample(model, label, noise_scheduler, device='cuda'):
    """
    Generate a new sample and get all relevant values for a specified digit label
    
    Parameters:
    -----------
    model : ClassConditionedUnet
        The trained UNet model
    label : int
        Desired digit label (0-9)
    noise_scheduler : DDPMScheduler
        The noise scheduler used for diffusion steps
    device : str, default='cuda'
        Device to run inference on
        
    Returns:
    --------
    input_noise : numpy.ndarray
        Initial random noise
    output : numpy.ndarray  
        Final generated image
    first_step : numpy.ndarray
        First timestep activations
    last_step : numpy.ndarray
        Last timestep activations
    """
    # Generate random input
    x = torch.randn(1, 1, 28, 28).to(device)
    y = torch.tensor([label], device=device)
    input_noise = x.clone().cpu().numpy()
    
    def hook_fn(module, input, output):
        if not hasattr(hook_fn, 'current_activations'):
            hook_fn.current_activations = []
        hook_fn.current_activations.append(output.detach().cpu().numpy())
    
    # Register hooks
    hooks = []
    for attn in model.model.mid_block.attentions:
        hooks.append(attn.to_q.register_forward_hook(hook_fn))
        hooks.append(attn.to_k.register_forward_hook(hook_fn))
        hooks.append(attn.to_v.register_forward_hook(hook_fn))
        hooks.append(attn.to_out[0].register_forward_hook(hook_fn))
    
    model.eval()
    with torch.no_grad():
        # First timestep activations
        hook_fn.current_activations = []
        _ = model(x, torch.zeros(1, device=device, dtype=torch.long), y)
        first_step = np.concatenate([act.reshape(1, -1) for act in hook_fn.current_activations], axis=1)
        
        # Run denoising steps
        for t in noise_scheduler.timesteps:
            hook_fn.current_activations = []
            residual = model(x, t, y)
            x = noise_scheduler.step(residual, t, x).prev_sample
        
        # Last timestep activations were captured in final step
        last_step = np.concatenate([act.reshape(1, -1) for act in hook_fn.current_activations], axis=1)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Normalize output to [0, 1] range
    output = x.cpu().numpy()
    output = (output.clip(-1, 1) + 1) / 2
    
    return input_noise, output, first_step, last_step

def knn_search(query, dataset, k=5, metric='euclidean'):
    """
    Perform KNN search for a query point in a dataset
    
    Parameters:
    -----------
    query : numpy.ndarray
        Query point of shape matching dataset samples
    dataset : numpy.ndarray
        Dataset to search in, shape [n_samples, n_features]
    k : int, default=5
        Number of neighbors to find
    metric : str, default='euclidean'
        Distance metric to use ('euclidean' or 'cosine')
        
    Returns:
    --------
    indices : numpy.ndarray
        Indices of k nearest neighbors
    distances : numpy.ndarray
        Distances to k nearest neighbors
    """
    # Ensure inputs are numpy arrays with correct shape
    query = np.asarray(query).flatten()
    dataset = np.asarray(dataset)
    if len(dataset.shape) > 2:
        dataset = dataset.reshape(dataset.shape[0], -1)
    
    # Calculate distances
    if metric == 'euclidean':
        distances = np.sqrt(((dataset - query) ** 2).sum(axis=1))
        #distances = np.abs(dataset - query).sum(axis=1)
    elif metric == 'cosine':
        norm_query = query / np.linalg.norm(query)
        norm_dataset = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
        distances = 1 - (norm_dataset @ norm_query)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Get top-k indices and distances
    top_indices = np.argsort(distances)[:k]
    top_distances = distances[top_indices]
    
    return top_indices, top_distances

def find_optimal_k(query, dataset, k_range=(5, 50), step=5, metric='euclidean', reference=None):
    """
    Find optimal k for KNN search based on blending results
    
    Parameters:
    -----------
    query : numpy.ndarray
        Query point to find neighbors for
    dataset : numpy.ndarray
        Dataset to search in
    k_range : tuple, default=(5, 50)
        Range of k values to test (min_k, max_k)
    step : int, default=5
        Step size for k values
    metric : str, default='euclidean'
        Distance metric to use ('euclidean' or 'cosine')
    reference : numpy.ndarray, optional
        Reference sample to compare blended result against
        
    Returns:
    --------
    tuple
        (best_k, best_indices, best_distances, best_distance)
        - best_k: optimal k value found
        - best_indices: indices for best k
        - best_distances: distances for best k
        - best_distance: distance to reference for best k
    """
    min_k, max_k = k_range
    k_values = range(min_k, max_k + 1, step)
    best_k = min_k
    best_distance = float('inf')
    best_indices = None
    best_distances = None
    
    for k in k_values:
        # Get k nearest neighbors
        indices, distances = knn_search(
            query=query,
            dataset=dataset,
            k=k,
            metric=metric
        )
        
        # Blend samples and get distance
        blended, dist = blend_samples(
            dataset=dataset,
            indices=indices,
            distances=distances,
            reference=reference if reference is not None else query
        )
        
        # Update best if current distance is lower
        if dist < best_distance:
            best_distance = dist
            best_k = k
            best_indices = indices
            best_distances = distances
    
    return best_k, best_indices, best_distances, best_distance


def blend_samples(dataset, indices, distances, reference=None, epsilon=1e-10):
    """
    Blend samples using inverse distance weighting and optionally compare to reference
    
    Parameters:
    -----------
    dataset : numpy.ndarray
        Dataset containing samples to blend, shape [n_samples, ...]
    indices : numpy.ndarray
        Indices of samples to blend
    distances : numpy.ndarray
        Distances to selected samples, matching indices
    reference : numpy.ndarray, optional
        Reference sample to compare blended result against
    epsilon : float, default=1e-10
        Small constant to avoid division by zero
        
    Returns:
    --------
    blended : numpy.ndarray
        Blended sample using inverse distance weighting
    ref_distance : float or None
        L2 distance to reference sample if provided, else None
    """
    # Ensure inputs are numpy arrays
    dataset = np.asarray(dataset)
    distances = np.asarray(distances)
    
    # Calculate inverse distance weights
    weights = 1 / (distances + epsilon)
    weights = weights / np.sum(weights)
    
    # Select samples to blend
    samples = dataset[indices]
    
    # Reshape samples and weights for broadcasting
    samples_flat = samples.reshape(len(samples), -1)  # Flatten all dimensions except first
    weights = weights.reshape(-1, 1)  # Shape: [n_samples, 1]
    
    # Calculate weighted average
    blended_flat = np.sum(samples_flat * weights, axis=0)
    
    # Reshape back to original shape
    blended = blended_flat.reshape(dataset.shape[1:])
    
    # Calculate distance to reference if provided
    ref_distance = None
    if reference is not None:
        reference = np.asarray(reference).flatten()
        ref_distance = np.sqrt(np.sum((blended_flat - reference.flatten()) ** 2))
    
    return blended, ref_distance

def visualize_neighbors(inputs, indices, distances=None, ncols=5, figsize=(15, 6)):
    """
    Visualize the nearest neighbors found by KNN search with their distances
    
    Parameters:
    -----------
    inputs : numpy.ndarray
        Dataset containing all input images, shape [n_samples, channels, height, width]
    indices : numpy.ndarray
        Indices of the nearest neighbors to visualize
    distances : numpy.ndarray, optional
        Distances to the nearest neighbors
    ncols : int, default=5
        Number of images per row
    figsize : tuple, default=(15, 6)
        Figure size (width, height)
    """
    n_samples = len(indices)
    nrows = (n_samples + ncols - 1) // ncols  # Ceiling division
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < n_samples:
            ax.imshow(inputs[indices[idx]][0], cmap='Greys')
            if distances is not None:
                ax.set_title(f'Neighbor {idx+1}\nDist: {distances[idx]:.4f}')
            else:
                ax.set_title(f'Neighbor {idx+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def combine_chunks(save_dir, prefix, total_samples=60000, batch_size=2000, delete_chunks=False):
    """
    Combine chunked files into a single file
    
    Parameters:
    -----------
    save_dir : str
        Directory containing the chunks
    prefix : str
        Prefix of the files to combine (e.g., 'inputs', 'labels', 'activations_first')
    total_samples : int, default=60000
        Total number of samples to process
    batch_size : int, default=2000
        Size of each chunk
    delete_chunks : bool, default=False
        Whether to delete the chunk files after combining
    """
    # Initialize list to store chunks
    chunks = []
    chunk_files = []
    
    # Load all chunks
    print(f"Loading {prefix} chunks...")
    for end_idx in range(batch_size, total_samples + batch_size, batch_size):
        file_path = os.path.join(save_dir, f'{prefix}_{end_idx}.npy')
        if os.path.exists(file_path):
            chunks.append(np.load(file_path))
            chunk_files.append(file_path)
            print(f"Loaded chunk ending at {end_idx}", end='\r')
    
    if not chunks:
        print(f"No chunks found for prefix '{prefix}'")
        return
    
    # Combine chunks
    print(f"\nCombining {len(chunks)} chunks...")
    combined = np.concatenate(chunks, axis=0)
    
    # Save combined file
    output_file = os.path.join(save_dir, f'{prefix}_all.npy')
    print(f"Saving combined file to {output_file}")
    np.save(output_file, combined)
    
    # Delete chunks if requested
    if delete_chunks:
        print("Deleting chunk files...")
        for file_path in chunk_files:
            os.remove(file_path)
    
    print(f"Done! Combined shape: {combined.shape}")
    return combined

def evaluate_random_sample(model, label, device='cuda'):
    """
    Generate and evaluate a random sample for a specified digit label
    
    Parameters:
    -----------
    model : ClassConditionedUnet
        The trained UNet model
    label : int
        Desired digit label (0-9)
    device : str, default='cuda'
        Device to run inference on
        
    Returns:
    --------
    input_tensor : torch.Tensor
        The random input tensor
    input_data : numpy.ndarray
        Processed input as numpy array
    first_timestep : numpy.ndarray
        First timestep (t=0) activations
    last_timestep : numpy.ndarray
        Last timestep (t=19) activations
    output : numpy.ndarray
        Final model output
    """
    # Generate random input
    input_tensor = torch.randn(1, 1, 28, 28).to(device)
    
    # Process the sample
    results = process_new_sample(model, input_tensor, label, device=device)
    
    return (
        input_tensor,
        results['input_data'],
        results['first_timestep'],
        results['last_timestep'],
        results['output']
    )

def load_all_chunks(save_dir, prefix, total_samples=60000, batch_size=2000):
    """Helper function to load and concatenate all chunks"""
    chunks = []
    for end_idx in range(batch_size, total_samples + batch_size, batch_size):
        file_path = os.path.join(save_dir, f'{prefix}_{end_idx}.npy')
        if os.path.exists(file_path):
            chunks.append(np.load(file_path))
    return np.concatenate(chunks, axis=0)

# Load the trained model
def load_trained_model(model_path, device='cuda'):
    # Initialize model and optimizer
    model = ClassConditionedUnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Load the saved state
    checkpoint = torch.load(model_path)
    
    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get training history
    losses = checkpoint['losses']
    last_epoch = checkpoint['epoch']
    
    print(f"Loaded model from epoch {last_epoch}")
    return model, optimizer, losses