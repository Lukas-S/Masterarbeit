import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def generate_rgb_samples(n_samples, radius=None, invert_mask=False):
    """
    Generate random RGB values between 0 and 1.
    
    Args:
        n_samples: Number of samples to generate
        radius: Sphere radius for masking (None for no masking)
        invert_mask: If True, generate samples outside the sphere
    
    Returns:
        numpy array of shape (n_samples, 3) with RGB values
    """
    if radius is None:
        return np.random.rand(n_samples, 3)
    
    samples = []
    while len(samples) < n_samples:
        # Generate random RGB values
        rgb = np.random.rand(10000, 3)  # Generate in batches for efficiency
        
        # Calculate distances from center (0.5, 0.5, 0.5)
        distances = np.linalg.norm(rgb - 0.5, axis=1)
        
        # Apply mask based on radius and invert_mask
        if invert_mask:
            mask = distances > radius
        else:
            mask = distances <= radius
        
        valid_samples = rgb[mask]
        samples.extend(valid_samples)
    
    return np.array(samples[:n_samples])

def plot_colors_3d(positions, colors, title="Colors in 3D Space"):
    """
    Visualize colors in 3D space with separate position and color inputs.
    
    Args:
        positions: numpy array of shape (n, 3) with 3D positions for each point
        colors: numpy array of shape (n, 3) with RGB color values for each point
        title: Plot title
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points using positions for coordinates and colors for display
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c=colors, s=25, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set axis limits to [0, 1]
    ax.set_xlim(positions.min(), positions.max())
    ax.set_ylim(positions.min(), positions.max())
    ax.set_zlim(positions.min(), positions.max())
    
    plt.show()

def rgb_to_hsv(rgb):
    """
    Convert RGB values to HSV.
    
    Args:
        rgb: numpy array of shape (..., 3) with RGB values between 0 and 1
    
    Returns:
        numpy array of same shape with HSV values (H: 0-1, S: 0-1, V: 0-1)
    """
    rgb = np.asarray(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    diff = max_val - min_val
    
    # Value
    v = max_val
    
    # Saturation
    s = np.where(max_val == 0, 0, diff / max_val)
    
    # Hue
    h = np.zeros_like(max_val)
    mask = diff != 0
    
    r_max = (max_val == r) & mask
    g_max = (max_val == g) & mask
    b_max = (max_val == b) & mask
    
    h[r_max] = (60 * ((g[r_max] - b[r_max]) / diff[r_max]) + 360) % 360
    h[g_max] = (60 * ((b[g_max] - r[g_max]) / diff[g_max]) + 120) % 360
    h[b_max] = (60 * ((r[b_max] - g[b_max]) / diff[b_max]) + 240) % 360
    
    # Normalize hue to 0-1 range
    h = h / 360
    
    return np.stack([h, s, v], axis=-1)


# Prediction function
def predict_hsv(model, rgb_input):
    """
    Predict HSV values from RGB input using the trained model.
    
    Args:
        model: Trained neural network
        rgb_input: RGB values as numpy array or tensor (shape: [n, 3] or [3])
    
    Returns:
        Predicted HSV values as numpy array
    """
    model.eval()
    with torch.no_grad():
        if isinstance(rgb_input, np.ndarray):
            rgb_tensor = torch.FloatTensor(rgb_input)
        else:
            rgb_tensor = rgb_input
        
        # Handle single sample
        if rgb_tensor.dim() == 1:
            rgb_tensor = rgb_tensor.unsqueeze(0)
        
        predictions = model(rgb_tensor)
        return predictions.numpy()
    


def train_model(model, rgb_data, hsv_data, num_epochs=100, batch_size=32, learning_rate=0.001, verbose=True):
    """
    Train the RGB to HSV neural network model.
    
    Args:
        model: Neural network model to train
        rgb_data: RGB training data (numpy array)
        hsv_data: HSV target data (numpy array)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        verbose: Whether to print training progress
    
    Returns:
        losses: List of training losses per epoch
        trained_model: The trained model
    """
    # Prepare data
    X_train = torch.FloatTensor(rgb_data)
    y_train = torch.FloatTensor(hsv_data)
    
    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_rgb, batch_hsv in dataloader:
            # Forward pass
            predictions = model(batch_rgb)
            loss = criterion(predictions, batch_hsv)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    if verbose:
        print("Training completed!")
    
    return losses, model


def calculate_prediction_errors(model, rgb_colors, error_metric='mse'):
    """
    Calculate prediction errors for custom RGB color arrays.
    
    Args:
        model: Trained neural network
        rgb_colors: RGB values as numpy array (shape: [n, 3])
        error_metric: Type of error to calculate ('mse', 'mae', 'per_channel')
    
    Returns:
        numpy array with errors for each input color
    """
    # Get predictions and ground truth
    predicted_hsv = predict_hsv(model, rgb_colors)
    actual_hsv = rgb_to_hsv(rgb_colors)
    
    if error_metric == 'mse':
        # Mean Squared Error for each sample
        errors = np.mean((predicted_hsv - actual_hsv) ** 2, axis=1)
    elif error_metric == 'mae':
        # Mean Absolute Error for each sample
        errors = np.mean(np.abs(predicted_hsv - actual_hsv), axis=1)
    elif error_metric == 'per_channel':
        # Error per channel (H, S, V) for each sample
        errors = np.abs(predicted_hsv - actual_hsv)
    else:
        raise ValueError("error_metric must be 'mse', 'mae', or 'per_channel'")
    
    return errors, predicted_hsv, actual_hsv

def plot_boxplots(overall_errors, names=['Group 1', 'Group 2']):
    """
    Plot two boxplots side by side.
    
    Args:
        overall_errors: 2D array with 2 columns of error values
        names: List of names for each boxplot
    """
    plt.figure(figsize=(8, 6))
    plt.boxplot([overall_errors[:, 0], overall_errors[:, 1]], labels=names)
    plt.ylabel('Error')
    plt.title('Error Comparison')
    plt.show()

def plot_two_points_3d(pos1, pos2, color1, color2, show_boundaries=True, show_line=True, title="Two Points in 3D Space"):
    """
    Visualize two specific points in 3D space with their colors.
    
    Args:
        pos1: Position of first point [x, y, z]
        pos2: Position of second point [x, y, z]
        color1: RGB color of first point [r, g, b]
        color2: RGB color of second point [r, g, b]
        show_boundaries: If True, show the 0-1 cube boundaries
        show_line: If True, draw line between the two points
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the two points
    ax.scatter(*pos1, c=[color1], s=100, alpha=0.9, edgecolors='black', linewidth=1)
    ax.scatter(*pos2, c=[color2], s=100, alpha=0.9, edgecolors='black', linewidth=1)
    
    # Draw line between points if requested
    if show_line:
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                'k--', alpha=0.7, linewidth=2)
        
        # Calculate and display distance
        distance = np.linalg.norm(np.array(pos2) - np.array(pos1))
        mid_point = [(pos1[i] + pos2[i]) / 2 for i in range(3)]
        ax.text(mid_point[0], mid_point[1], mid_point[2], f'd={distance:.3f}', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Show 0-1 cube boundaries if requested
    if show_boundaries:
        # Draw cube edges
        for i in [0, 1]:
            for j in [0, 1]:
                ax.plot([i, i], [j, j], [0, 1], 'gray', alpha=0.3, linewidth=1)
                ax.plot([i, i], [0, 1], [j, j], 'gray', alpha=0.3, linewidth=1)
                ax.plot([0, 1], [i, i], [j, j], 'gray', alpha=0.3, linewidth=1)
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set appropriate limits
    all_coords = np.array([pos1, pos2])
    if show_boundaries:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
    else:
        margin = 0.1
        ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
        ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
        ax.set_zlim(all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin)
    
    plt.show()

def capture_activations(model, data):
    """
    Capture neuron activations from hidden layers during forward pass.
    
    Args:
        model: Trained neural network
        data: Input data (numpy array or tensor)
    
    Returns:
        activations: Numpy array with shape (n_samples, total_hidden_neurons)
        Concatenates all hidden layer activations (excludes output layer)
    """
    # Convert data to tensor if needed
    if isinstance(data, np.ndarray):
        data_tensor = torch.FloatTensor(data)
    else:
        data_tensor = data
    
    # List to store activations
    all_activations = []
    
    # Define hook function
    def hook_fn(module, input, output):
        all_activations.append(output.detach().numpy())
    
    # Register hooks only on hidden Linear layers (exclude output layer)
    hooks = []
    layer_count = 0
    total_linear_layers = sum(1 for module in model.modules() if isinstance(module, (nn.Linear, nn.ReLU)))
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.ReLU)):
            layer_count += 1
            # Skip the last linear layer (output layer)
            if layer_count < total_linear_layers:
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
    
    # Forward pass without gradients
    model.eval()
    with torch.no_grad():
        _ = model(data_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate all hidden layer activations
    if all_activations:
        return np.concatenate(all_activations, axis=1)
    else:
        return np.array([])

def plot_line_graphs(lines, colors=None, labels=None, title="Line Graph", xlabel="Neuron Index", ylabel="Activation Value", smooth=False, smooth_window=5):
    """
    Plot up to 4 line graphs with customizable colors and optional smoothing.
    
    Args:
        lines: List of arrays to plot (max 4 lines), each with 256 values
        colors: List of colors for each line (default: ['blue', 'red', 'green', 'orange'])
        labels: List of labels for each line (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        smooth: If True, apply smoothing to lines
        smooth_window: Window size for smoothing (default: 5)
    """
    if len(lines) > 4:
        raise ValueError("Maximum 4 lines allowed")
    
    # Default colors
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange']
    
    # Default labels
    if labels is None:
        labels = [f'Line {i+1}' for i in range(len(lines))]
    
    plt.figure(figsize=(12, 5))
    
    # Plot each line
    for i, line in enumerate(lines):
        if smooth:
            # Apply smoothing using moving average
            smoothed_line = np.convolve(line.astype(float), np.ones(smooth_window)/smooth_window, mode='same')
            plt.plot(smoothed_line, color=colors[i], label=labels[i], linewidth=2)
        else:
            plt.plot(line, color=colors[i], label=labels[i], linewidth=2)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def check_activation_bounds(activation, min_vals, max_vals):
    within = (activation >= min_vals) & (activation <= max_vals)
    return {
        'within_bounds': within,
        'all_within': np.all(within),
        'num_within': np.sum(within),
        'exceeding_indices': np.where(~within)[0]
    }

def plot_activation_spectrum(activations, rgb_colors, show_minmax=True, alpha=0.01, colored_spectrum=False, title="Activation Spectrum"):
    """
    Plot all activation arrays with low alpha and color-coded min/max points.
    
    Args:
        activations: Array of shape (1000, 256) with all activation values
        rgb_colors: Array of shape (1000, 3) with corresponding RGB colors
        show_minmax: If True, overlay min/max lines with color-coded points
        alpha: Alpha value for individual activation lines
        colored_spectrum: If True, color each line with its RGB color; if False, use grey
        title: Plot title
    """
    plt.figure(figsize=(15, 8))
    
    # Plot all activation arrays with low alpha
    for i in range(activations.shape[0]):
        if colored_spectrum:
            plt.plot(activations[i], color=rgb_colors[i], alpha=alpha, linewidth=0.5)
        else:
            plt.plot(activations[i], color='gray', alpha=alpha, linewidth=0.5)
    
    if show_minmax:
        # Calculate min and max values and their corresponding colors
        min_vals = np.min(activations, axis=0)
        max_vals = np.max(activations, axis=0)
        
        # Find which sample contributed to each min/max value
        min_indices = np.argmin(activations, axis=0)  # Shape: (256,)
        max_indices = np.argmax(activations, axis=0)  # Shape: (256,)
        
        # Get colors for min/max points
        min_colors = rgb_colors[min_indices]  # Shape: (256, 3)
        max_colors = rgb_colors[max_indices]  # Shape: (256, 3)
        
        # Plot min line with colored points
        plt.plot(min_vals, 'k-', linewidth=2, label='Min Values', alpha=0.7)
        for i in range(len(min_vals)):
            plt.scatter(i, min_vals[i], c=[min_colors[i]], s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Plot max line with colored points
        plt.plot(max_vals, 'k-', linewidth=2, label='Max Values', alpha=0.7)
        for i in range(len(max_vals)):
            plt.scatter(i, max_vals[i], c=[max_colors[i]], s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        plt.legend()
    
    plt.xlabel('Neuron Index')
    plt.ylabel('Activation Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def find_knn(sample, dataset, k=5, metric='l2'):
    """
    Find K nearest neighbors for a sample in a dataset.
    
    Args:
        sample: Single sample array (shape: [n_features])
        dataset: Dataset to search in (shape: [n_samples, n_features])
        k: Number of nearest neighbors to return
        metric: Distance metric ('l1' for Manhattan, 'l2' for Euclidean)
    
    Returns:
        indices: Indices of k nearest neighbors
        distances: Distances to k nearest neighbors
    """
    # Ensure sample is 2D for sklearn
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    # Set up metric parameter for sklearn
    sklearn_metric = 'manhattan' if metric == 'l1' else 'euclidean'
    
    # Create and fit KNN
    knn = NearestNeighbors(n_neighbors=k, metric=sklearn_metric)
    knn.fit(dataset)
    
    # Find neighbors
    distances, indices = knn.kneighbors(sample)
    
    return indices[0], distances[0]  # Return 1D arrays

def inverse_distance_weighted_activations(activations, indices, distances, epsilon=1e-8):
    """
    Calculate inverse distance weighted average of activations.
    
    Args:
        activations: Full activation dataset (shape: [n_samples, n_neurons])
        indices: Indices of selected samples
        distances: Distances to selected samples
        epsilon: Small value to avoid division by zero
    
    Returns:
        weighted_activation: Inverse distance weighted activation values
        weights: The calculated weights for each sample
    """
    # Get activations for selected indices
    selected_activations = activations[indices]  # Shape: [k, n_neurons]
    
    # Calculate inverse distance weights
    weights = 1.0 / (distances + epsilon)  # Add epsilon to avoid division by zero
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    
    # Apply weights
    weighted_activation = np.average(selected_activations, axis=0, weights=weights)
    
    return weighted_activation, weights

def knn_blending_analysis(target_sample, dataset, activations, max_k=100, metric='l2'):
    """
    Analyze KNN blending by tracking distance changes as k increases.
    
    Args:
        target_sample: Target sample to blend towards (shape: [n_features])
        dataset: Dataset to search in (shape: [n_samples, n_features])
        activations: Activation dataset (shape: [n_samples, n_neurons])
        max_k: Maximum number of neighbors to use (default: 100)
        metric: Distance metric ('l1' or 'l2')
    
    Returns:
        distances_to_target: Array of distances between blended and target for each k
        blended_activations: List of blended activation arrays for each k
        neighbor_indices: Indices of all neighbors (up to max_k)
        neighbor_distances: Distances to all neighbors (up to max_k)
    """
    # Find all neighbors up to max_k
    neighbor_indices, neighbor_distances = find_knn(target_sample, dataset, k=max_k, metric=metric)
    
    # Get target activation (assuming it's in the dataset at the closest neighbor)
    target_activation = activations[neighbor_indices[0]]
    
    distances_to_target = []
    blended_activations = []
    
    # Calculate blending for k = 1 to max_k
    for k in range(1, max_k + 1):
        # Use first k neighbors
        current_indices = neighbor_indices[:k]
        current_distances = neighbor_distances[:k]
        
        # Calculate inverse distance weighted activation
        weighted_activation, _ = inverse_distance_weighted_activations(
            activations, current_indices, current_distances
        )
        
        # Calculate distance between blended and target activation
        if metric == 'l1':
            distance_to_target = np.sum(np.abs(weighted_activation - target_activation))
        else:  # l2
            distance_to_target = np.sqrt(np.sum((weighted_activation - target_activation) ** 2))
        
        distances_to_target.append(distance_to_target)
        blended_activations.append(weighted_activation)
    
    return np.array(distances_to_target), blended_activations, neighbor_indices, neighbor_distances

def plot_color_sequence(rgb_colors, indices, distances=None, title="Color Sequence"):
    """
    Display colors in order based on provided indices.
    
    Args:
        rgb_colors: Full RGB color dataset (shape: [n_samples, 3])
        indices: Indices of colors to display in order
        distances: Optional distances to display as labels
        title: Plot title
    """
    # Get colors for the specified indices
    selected_colors = rgb_colors[indices]
    
    fig, ax = plt.subplots(1, 1, figsize=(max(len(indices), 8), 2))
    
    # Create color patches
    for i, (idx, color) in enumerate(zip(indices, selected_colors)):
        # Create rectangle for each color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Add index label at bottom
        ax.text(i + 0.5, -0.15, f'{idx}', ha='center', va='top', fontsize=8)
        
        # Add distance label at top if provided
        if distances is not None:
            ax.text(i + 0.5, 1.15, f'{distances[i]:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Set axis limits and labels
    ax.set_xlim(0, len(indices))
    ax.set_ylim(-0.3, 1.3 if distances is not None else 1.1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('Order')
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    # Add labels
    if distances is not None:
        ax.text(-0.5, 1.15, 'Distance:', ha='right', va='bottom', fontsize=8, weight='bold')
    ax.text(-0.5, -0.15, 'Index:', ha='right', va='top', fontsize=8, weight='bold')
    
    plt.tight_layout()
    plt.show()

def calc_weighted_distance(distances, indices, activations):
    weights = 1/distances
    norm_weights = weights / np.sum(weights)
    weighted_activations = activations[indices] * norm_weights[:, np.newaxis]
    weighted_activations = weighted_activations.sum(axis=0)
    return weighted_activations