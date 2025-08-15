import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn import datasets as sklearn_datasets
import numpy as np
import matplotlib.pyplot as plt

import random

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All seeds set to: {seed}")

def init_weights(model, seed=42):
    """Initialize model weights deterministically"""
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    print(f"Model weights initialized with seed: {seed}")

def generate_rgb_samples(n_samples, device='cpu', seed=42):
    torch.manual_seed(seed)
    return torch.rand(n_samples, 3, device=device)

def rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    max_c = torch.max(torch.max(r, g), b)
    min_c = torch.min(torch.min(r, g), b)
    diff = max_c - min_c
    
    h = torch.zeros_like(max_c)
    s = torch.zeros_like(max_c)
    v = max_c
    
    mask = diff != 0
    s[mask] = diff[mask] / max_c[mask]
    
    r_mask = (max_c == r) & mask
    g_mask = (max_c == g) & mask
    b_mask = (max_c == b) & mask
    
    h[r_mask] = ((g[r_mask] - b[r_mask]) / diff[r_mask]) % 6
    h[g_mask] = ((b[g_mask] - r[g_mask]) / diff[g_mask]) + 2
    h[b_mask] = ((r[b_mask] - g[b_mask]) / diff[b_mask]) + 4
    
    h = h / 6.0
    
    return torch.stack([h, s, v], dim=1)

def predict(model, inputs, device=None):
    """Make predictions with a trained model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)
        
        inputs = inputs.to(device)
        outputs = model(inputs)
        return outputs.cpu()

def get_samples_by_indices(dataloader, indices):
    """Get specific samples from dataloader by indices"""
    all_inputs, all_targets = [], []
    
    for inputs, targets in dataloader:
        all_inputs.append(inputs)
        all_targets.append(targets)
    
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    if isinstance(indices, int):
        indices = [indices]
    
    selected_inputs = all_inputs[indices]
    selected_targets = all_targets[indices]
    
    return selected_inputs, selected_targets

def create_filtered_dataloader(original_dataloader, exclude_indices=None, include_indices=None, batch_size=None, shuffle=False):
    """
    Create a new dataloader excluding or including specific indices from the original dataloader.
    
    Args:
        original_dataloader: The original DataLoader
        exclude_indices: List/tensor of indices to exclude (mutually exclusive with include_indices)
        include_indices: List/tensor of indices to include only (mutually exclusive with exclude_indices)
        batch_size: Batch size for new dataloader (if None, uses original batch size)
        shuffle: Whether to shuffle the new dataloader
    
    Returns:
        DataLoader: New filtered dataloader
        int: Number of samples in the new dataloader
    """
    if exclude_indices is not None and include_indices is not None:
        raise ValueError("Cannot specify both exclude_indices and include_indices")
    
    # Get all data from original dataloader
    all_inputs, all_targets = [], []
    for inputs, targets in original_dataloader:
        all_inputs.append(inputs)
        all_targets.append(targets)
    
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    total_samples = len(all_inputs)
    
    if exclude_indices is not None:
        # Convert to tensor if it's not already
        if isinstance(exclude_indices, list):
            exclude_indices = torch.tensor(exclude_indices)
        elif isinstance(exclude_indices, np.ndarray):
            exclude_indices = torch.from_numpy(exclude_indices)
        
        # Create mask for all indices except excluded ones
        all_indices = torch.arange(total_samples)
        mask = torch.ones(total_samples, dtype=torch.bool)
        mask[exclude_indices] = False
        selected_indices = all_indices[mask]
        
    elif include_indices is not None:
        # Convert to tensor if it's not already
        if isinstance(include_indices, list):
            selected_indices = torch.tensor(include_indices)
        elif isinstance(include_indices, np.ndarray):
            selected_indices = torch.from_numpy(include_indices)
        else:
            selected_indices = include_indices
            
    else:
        # No filtering, return original dataloader recreated
        selected_indices = torch.arange(total_samples)
    
    # Select the filtered data
    filtered_inputs = all_inputs[selected_indices]
    filtered_targets = all_targets[selected_indices]
    
    # Use original batch size if not specified
    if batch_size is None:
        batch_size = original_dataloader.batch_size
    
    # Create new dataloader
    filtered_dataset = TensorDataset(filtered_inputs, filtered_targets)
    filtered_dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return filtered_dataloader, len(filtered_inputs)

def get_activations(model, dataloader, dataset_type='mnist', indices=None, device=None):
    """Capture all layer activations using forward hooks. Returns inputs, activations, and outputs as tensors."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    # Storage for activations
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            # Flatten and store activations
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu().flatten(1)  # Keep batch dim, flatten rest
            return None
        return hook
    
    # Register hooks for Linear and ReLU layers, but exclude the final Linear layer
    hooks = []
    layer_names = []
    
    # Get all layer names first to identify the last Linear layer
    all_linear_layers = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    last_linear_layer = all_linear_layers[-1] if all_linear_layers else None
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.ReLU)):
            # Skip the final Linear layer to avoid duplicating the output
            if isinstance(module, nn.Linear) and name == last_linear_layer:
                continue
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
            layer_names.append(name)
    
    # Get data
    if indices is not None:
        sample_inputs, sample_targets = get_samples_by_indices(dataloader, indices)
        data_inputs = sample_inputs
    else:
        # Get all data from dataloader
        all_inputs, all_targets = [], []
        for inputs, targets in dataloader:
            all_inputs.append(inputs)
            all_targets.append(targets)
        data_inputs = torch.cat(all_inputs, dim=0)
    
    # Forward pass to capture activations
    with torch.no_grad():
        if dataset_type == 'mnist':
            # Flatten MNIST inputs
            flattened_inputs = data_inputs.view(data_inputs.size(0), -1)
            outputs = model(data_inputs.to(device))
        else:
            flattened_inputs = data_inputs
            outputs = model(data_inputs.to(device))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Organize activations in order
    ordered_activations = []
    for name in layer_names:
        if name in activations:
            ordered_activations.append(activations[name])
    
    # Concatenate all activations
    if ordered_activations:
        all_activations = torch.cat(ordered_activations, dim=1)
    else:
        all_activations = torch.empty(data_inputs.size(0), 0)
    
    return flattened_inputs.cpu(), all_activations, outputs.cpu()

def show_mnist_samples(dataloader, indices=None):
    """Show MNIST samples by specific indices"""
    if indices is None:
        indices = list(range(8))
    elif isinstance(indices, int):
        indices = [indices]
    
    inputs, targets = get_samples_by_indices(dataloader, indices)
    
    n_samples = len(indices)
    if n_samples == 1:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(inputs[0].squeeze(), cmap='gray')
        ax.set_title(f'Index: {indices[0]}, Label: {targets[0].item()}')
        ax.axis('off')
    else:
        cols = min(4, n_samples)
        rows = (n_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(n_samples):
            axes[i].imshow(inputs[i].squeeze(), cmap='gray')
            axes[i].set_title(f'Index: {indices[i]}, Label: {targets[i].item()}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_iris_samples(X, y, indices=None):
    """Show Iris samples by specific indices"""
    if indices is None:
        indices = list(range(10))
    elif isinstance(indices, int):
        indices = [indices]
    
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    class_names = ['setosa', 'versicolor', 'virginica']
    
    print("Iris Dataset Samples:")
    print("-" * 80)
    print(f"{'Index':<6} {'Class':<12} {'Sepal_L':<8} {'Sepal_W':<8} {'Petal_L':<8} {'Petal_W':<8}")
    print("-" * 80)
    
    for idx in indices:
        if idx < len(X):
            sample = X[idx]
            label = y[idx].item() if hasattr(y[idx], 'item') else y[idx]
            class_name = class_names[label]
            print(f"{idx:<6} {class_name:<12} {sample[0]:<8.2f} {sample[1]:<8.2f} {sample[2]:<8.2f} {sample[3]:<8.2f}")
        else:
            print(f"Index {idx} out of range (dataset size: {len(X)})")

def show_rgb_hsv_samples(rgb_data, hsv_data, indices=None):
    """Show RGB-HSV conversion samples by specific indices"""
    if indices is None:
        indices = list(range(5))
    elif isinstance(indices, int):
        indices = [indices]
    
    print("RGB to HSV Conversion Samples:")
    print("-" * 60)
    print(f"{'Index':<6} {'RGB':<20} {'HSV':<20}")
    print("-" * 60)
    
    for idx in indices:
        if idx < len(rgb_data):
            rgb = rgb_data[idx]
            hsv = hsv_data[idx]
            rgb_str = f"({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})"
            hsv_str = f"({hsv[0]:.3f}, {hsv[1]:.3f}, {hsv[2]:.3f})"
            print(f"{idx:<6} {rgb_str:<20} {hsv_str:<20}")
        else:
            print(f"Index {idx} out of range (dataset size: {len(rgb_data)})")

def predict_and_show(model, dataloader, dataset_type='mnist', indices=None, device=None):
    """Make predictions and show results for specific indices. Returns multiple variables as tensors."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if indices is None:
        indices = list(range(5))
    elif isinstance(indices, int):
        indices = [indices]
    
    model.eval()
    model.to(device)
    
    sample_inputs, sample_targets = get_samples_by_indices(dataloader, indices)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(sample_inputs.to(device)).cpu()
    
    if dataset_type == 'mnist':
        predicted_classes = torch.argmax(predictions, dim=1)
        probabilities = torch.softmax(predictions, dim=1)
        
        print("MNIST Predictions:")
        print("-" * 50)
        print(f"{'Index':<6} {'True':<6} {'Pred':<6} {'Confidence':<12}")
        print("-" * 50)
        
        for i, idx in enumerate(indices):
            true_label = sample_targets[i].item()
            pred_label = predicted_classes[i].item()
            confidence = probabilities[i][pred_label].item()
            print(f"{idx:<6} {true_label:<6} {pred_label:<6} {confidence:<12.4f}")
        
        return predictions, probabilities, predicted_classes, sample_targets
            
    elif dataset_type == 'iris':
        predicted_classes = torch.argmax(predictions, dim=1)
        probabilities = torch.softmax(predictions, dim=1)
        class_names = ['setosa', 'versicolor', 'virginica']
        
        print("Iris Predictions:")
        print("-" * 60)
        print(f"{'Index':<6} {'True':<12} {'Pred':<12} {'Confidence':<12}")
        print("-" * 60)
        
        for i, idx in enumerate(indices):
            true_label = sample_targets[i].item()
            pred_label = predicted_classes[i].item()
            confidence = probabilities[i][pred_label].item()
            true_class = class_names[true_label]
            pred_class = class_names[pred_label]
            print(f"{idx:<6} {true_class:<12} {pred_class:<12} {confidence:<12.4f}")
        
        return predictions, probabilities, predicted_classes, sample_targets
            
    elif dataset_type == 'rgb_hsv':
        print("RGB to HSV Predictions:")
        print("-" * 90)
        print(f"{'Index':<6} {'Input RGB':<20} {'True HSV':<20} {'Pred HSV':<20}")
        print("-" * 90)
        
        for i, idx in enumerate(indices):
            rgb = sample_inputs[i]
            true_hsv = sample_targets[i]
            pred_hsv = predictions[i]
            
            rgb_str = f"({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})"
            true_str = f"({true_hsv[0]:.3f}, {true_hsv[1]:.3f}, {true_hsv[2]:.3f})"
            pred_str = f"({pred_hsv[0]:.3f}, {pred_hsv[1]:.3f}, {pred_hsv[2]:.3f})"
            
            print(f"{idx:<6} {rgb_str:<20} {true_str:<20} {pred_str:<20}")
        
        return predictions, sample_targets, sample_inputs

def train_model(model, dataloader, criterion, optimizer, num_epochs=100, device=None, verbose=False, seed=42, continue_training=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not continue_training:
        set_seed(seed)
        init_weights(model, seed)
    
    model.to(device)
    model.train()
    losses = []
    
    if verbose:
        print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return losses, model

def find_knn(dataset, single_sample, k=5, metric='l2'):
    """
    Find k nearest neighbors using different distance metrics.
    
    Args:
        dataset (torch.Tensor): Dataset to search in (N x D)
        single_sample (torch.Tensor): Single sample to find neighbors for (1 x D)
        k (int): Number of nearest neighbors to find
        metric (str): Distance metric ('l1', 'l2', 'cosine')
    
    Returns:
        indices (torch.Tensor): Indices of k nearest neighbors (k,)
        distances (torch.Tensor): Distances to k nearest neighbors (k,)
    """
    # Ensure inputs are 2D tensors
    if len(dataset.shape) == 1:
        dataset = dataset.unsqueeze(0)
    if len(single_sample.shape) == 1:
        single_sample = single_sample.unsqueeze(0)
    
    # Remove batch dimension from single_sample for broadcasting
    if single_sample.shape[0] == 1:
        query = single_sample.squeeze(0)  # (D,)
    else:
        query = single_sample[0]  # Take first sample if multiple
    
    if metric == 'l1':
        # L1 (Manhattan) distance
        distances = torch.sum(torch.abs(dataset - query), dim=1)
    
    elif metric == 'l2':
        # L2 (Euclidean) distance
        distances = torch.sqrt(torch.sum((dataset - query) ** 2, dim=1))
    
    elif metric == 'cosine':
        # Cosine distance (1 - cosine similarity)
        # Normalize vectors
        dataset_norm = torch.nn.functional.normalize(dataset, p=2, dim=1)
        query_norm = torch.nn.functional.normalize(query.unsqueeze(0), p=2, dim=1)
        
        # Compute cosine similarity
        cosine_sim = torch.mm(dataset_norm, query_norm.t()).squeeze()
        
        # Convert to cosine distance
        distances = 1 - cosine_sim
    
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from 'l1', 'l2', 'cosine'")
    
    # Find k smallest distances
    k = min(k, len(distances))  # Ensure k doesn't exceed dataset size
    top_k_distances, top_k_indices = torch.topk(distances, k, largest=False)
    
    return top_k_indices, top_k_distances

# # Function to calculate MSE loss for different k and indices
# def calculate_mse_for_k_and_indices(model, network_arch, activation_data, criterion, train_loader, test_loader, list_of_k, list_of_indices):
#     mse_results = {}
#     for k in list_of_k:
#         mse_results[k] = []
#         for index in list_of_indices:
#             # Get activations for the single sample
#             single_inputs, single_activations, _ = get_activations(model, test_loader, dataset_type='rgb_hsv', indices=index)
#
#             # Find k-nearest neighbors
#             indices_knn, _ = find_knn(activation_data, single_activations, k=k, metric='l2')
#
#             # Create filtered dataloader
#             filtered_loader, _ = create_filtered_dataloader(train_loader, exclude_indices=indices_knn, batch_size=64, shuffle=False)
            
#             # Train a new model on the filtered dataset
#             filtered_model = network_arch()
#             criterion = criterion
#             optimizer = optim.Adam(filtered_model.parameters(), lr=0.001)
#             _, trained_filtered_model = train_model(filtered_model, filtered_loader, criterion, optimizer, num_epochs=100, seed=42, verbose=False)
            
#             # Predict and calculate MSE loss
#             predictions, targets, _ = predict_and_show(trained_filtered_model, test_loader, dataset_type='rgb_hsv', indices=index)
#             loss = criterion(predictions, targets).item()
#             mse_results[k].append(loss)
#     return mse_results

# Updated Visualization function
def visualize_mse_results(mse_results, list_of_indices, selected_index=None, average=False, base_mse=None):
       
    plt.figure(figsize=(10, 6))
    
    if selected_index is not None:
        # Visualize for a specific index
        mse_values = [mse_results[k][selected_index] for k in mse_results]
        plt.plot(list(mse_results.keys()), mse_values, marker='o', label=f'Index {list_of_indices[selected_index]}')
    elif average:
        # Visualize averaged results
        mse_values = [sum(mse_results[k]) / len(mse_results[k]) for k in mse_results]
        plt.plot(list(mse_results.keys()), mse_values, marker='o', label='Average MSE')
    else:
        # Visualize all indices
        for i, index in enumerate(list_of_indices):
            mse_values = [mse_results[k][i] for k in mse_results]
            plt.plot(list(mse_results.keys()), mse_values, marker='o', label=f'Index {index}')
    
    if base_mse is not None:
        # Add a straight line for the base model MSE
        plt.axhline(y=base_mse, color='r', linestyle='--', label='Base Model MSE')
    
    plt.xlabel('K Neighbors')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss for Different K Neighbors')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_results(mse_results, list_of_indices, selected_index=None, average=False, base_mse=None, normalize=False, title="Loss Visualization", ylabel="Loss"):
    plt.figure(figsize=(12, 8))

    if average:
        # Prepare data for averages
        averages = {}
        for key, results in mse_results.items():
            averages[key] = [np.mean(results[k]) for k in results.keys()]

        # Normalize if required
        if normalize:
            max_value = max([max(values) for values in averages.values()])
            averages = {key: [val / max_value for val in values] for key, values in averages.items()}

        # Plot averages side by side
        for key, values in averages.items():
            plt.plot(list(mse_results[key].keys()), values, label=f"{key} (Average)", marker="o")

    elif selected_index is not None:
        # Plot for a specific index
        for key, results in mse_results.items():
            plt.plot(list(results.keys()), [results[k][selected_index] for k in results.keys()], label=f"{key} (Index {selected_index})", marker="o")

    else:
        # Plot all indices
        for key, results in mse_results.items():
            for idx, values in enumerate(zip(*results.values())):
                plt.plot(list(results.keys()), values, label=f"{key} (Index {idx})", marker="o")

    if base_mse is not None:
        plt.axhline(y=base_mse, color="r", linestyle="--", label="Base MSE")

    plt.xlabel("K Neighbors")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()