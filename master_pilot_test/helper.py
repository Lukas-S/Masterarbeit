import torch
import torch.nn as nn
import numpy as np

import network
import matplotlib.pyplot as plt

from network import *
from mpl_toolkits.mplot3d import Axes3D
from colorsys import rgb_to_hsv, hsv_to_rgb
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(17)
#np.random.seed(17)


def generate_rgb_samples(n_samples):
    """Generate random RGB samples between 0 and 1"""
    rgb_samples = np.random.uniform(0, 1, (n_samples, 3))
    return rgb_samples

def rgb_to_hsv_batch(rgb_samples):
    """Convert batch of RGB samples to HSV"""
    hsv_samples = np.array([rgb_to_hsv(*rgb) for rgb in rgb_samples])
    return hsv_samples

def prepare_data(n_samples):
    """Prepare data for PyTorch model"""
    rgb_data = generate_rgb_samples(n_samples)
    hsv_data = rgb_to_hsv_batch(rgb_data)
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(rgb_data)
    y = torch.FloatTensor(hsv_data)
    
    return X, y

def visualize_samples(rgb_samples):
    """Visualize RGB samples in 3D space"""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(
        rgb_samples[:, 0],
        rgb_samples[:, 1],
        rgb_samples[:, 2],
        c=rgb_samples,
        marker='o',
        s = 5
    )
    
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title('RGB Color Space Samples')
    plt.show()
    
def visualize_hsv_samples(hsv_samples):
    """
    Visualize HSV samples in 3D space, with points colored by their RGB equivalents
    
    Args:
        hsv_samples: numpy array of HSV values, shape (n_samples, 3)
    """
    # Convert HSV to RGB for coloring
    rgb_colors = np.array([hsv_to_rgb(*hsv) for hsv in hsv_samples])
    
    # Create 3D plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(
        hsv_samples[:, 0],  # Hue
        hsv_samples[:, 1],  # Saturation
        hsv_samples[:, 2],  # Value
        c=rgb_colors,       # Color points by their RGB values
        marker='o',
        s=5
    )
    
    # Set labels and title
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')
    ax.set_title('HSV Color Space Samples')
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    plt.show()
    
def visualize_model_structure(model):
    """
    Visualize the structure of the neural network model with detailed layer information
    """
    def count_parameters(layer):
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)
    
    print("\nNeural Network Structure")
    print("=" * 85)
    print(f"{'#':<3} {'Layer Type':<20} {'Input Shape':<15} {'Output Shape':<15} {'Parameters':<12} {'Activation':<10}")
    print("-" * 85)
    
    # Input layer
    print(f"{'0':<3} {'Input':<20} {'(-, 3)':<15} {'(-, 3)':<15} {'0':<12} {'None':<10}")
    
    # Track the current shape
    current_shape = 3
    
    # Iterate through model layers
    for i, layer in enumerate(model.model):
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
            params = count_parameters(layer)
            print(f"{i+1:<3} {'Linear':<20} {'(-, '+str(in_features)+')':<15} "
                  f"{'(-, '+str(out_features)+')':<15} {str(params):<12} {'None':<10}")
            current_shape = out_features
        elif isinstance(layer, nn.ReLU):
            print(f"{i+1:<3} {'ReLU':<20} {'(-, '+str(current_shape)+')':<15} "
                  f"{'(-, '+str(current_shape)+')':<15} {'0':<12} {'ReLU':<10}")
    
    print("=" * 85)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Final output shape: (batch_size, 3) [HSV values]")

    
def test_random_color(model):
    """Test the model with a random RGB color"""
    # Generate random RGB color
    rgb_test = torch.FloatTensor(np.random.uniform(0, 1, (1, 3)))
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        hsv_pred = model(rgb_test)
    
    # Convert to numpy for visualization
    rgb_np = rgb_test.numpy()[0]
    hsv_np = hsv_pred.numpy()[0]
    
    return rgb_np, hsv_np

def visualize_conversion(rgb, hsv):
    """Visualize input RGB and output HSV side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 1.5))
    
    # Plot RGB color
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, color=rgb))
    ax1.set_title(f'Input RGB: ({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})')
    ax1.axis('equal')
    
    # Create color from HSV
    hsv_color = plt.cm.hsv(hsv[0])  # Use hue for visualization
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, color=hsv_color))
    ax2.set_title(f'Output HSV: ({hsv[0]:.2f}, {hsv[1]:.2f}, {hsv[2]:.2f})')
    ax2.axis('equal')
    
    plt.show()
    
def train_model(model, train_loader, num_epochs, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_rgb, batch_hsv in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_rgb)
            loss = criterion(outputs, batch_hsv)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Test and visualize
def test_model(model, rgb_input=None):
    """Test the model with either a random RGB color or a specific input
    
    Args:
        model: The trained RGB to HSV model
        rgb_input: Optional list/array of 3 RGB values between 0 and 1.
                  If None, generates random values
    """
    if rgb_input is None:
        # Generate random RGB color
        rgb_test = torch.FloatTensor(np.random.uniform(0, 1, (1, 3)))
    else:
        # Use provided RGB values
        rgb_input = np.array(rgb_input).reshape(1, 3)
        rgb_test = torch.FloatTensor(rgb_input)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        hsv_pred = model(rgb_test)
    
    # Convert to numpy for visualization
    rgb_np = rgb_test.numpy()[0]
    hsv_np = hsv_pred.numpy()[0]
    
    # Visualize the conversion
    visualize_conversion(rgb_np, hsv_np)
    
    # Print true HSV values for comparison
    true_hsv = rgb_to_hsv(*rgb_np)
    print(f"Pred HSV values: ({hsv_np[0]:.2f}, {hsv_np[1]:.2f}, {hsv_np[2]:.2f})")
    print(f"True HSV values: ({true_hsv[0]:.2f}, {true_hsv[1]:.2f}, {true_hsv[2]:.2f})")
    
    return rgb_np, hsv_np
    
    
def collect_activations(model, train_loader):
    """Collect all activations from the network during one forward pass"""
    
    # Lists to store inputs, activations, and outputs
    all_inputs = []
    all_activations = []
    all_outputs = []
    
    # Dictionary to store intermediate activations for each forward pass
    activation_dict = {}
    
    # Hook function to capture activations
    def hook_fn(module, input, output):
        activation_dict[len(activation_dict)] = output.detach().cpu().numpy()
    
    # Register hooks for all layers except the final one
    hooks = []
    for layer in list(model.model)[:-1]:  # Exclude the final layer
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # Set model to eval mode
    model.eval()
    
    # Forward pass through all samples
    with torch.no_grad():
        for batch_rgb, _ in train_loader:
            # Store inputs
            all_inputs.append(batch_rgb.numpy())
            
            # Forward pass
            outputs = model(batch_rgb)
            
            # Store outputs
            all_outputs.append(outputs.numpy())
            
            # Store activations for this batch
            batch_activations = []
            for i in range(len(activation_dict)):
                batch_activations.append(activation_dict[i])
            all_activations.append(np.concatenate([act.reshape(act.shape[0], -1) for act in batch_activations], axis=1))
            
            # Clear activation dictionary for next batch
            activation_dict.clear()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate all batches
    inputs = np.concatenate(all_inputs, axis=0)
    activations = np.concatenate(all_activations, axis=0)
    outputs = np.concatenate(all_outputs, axis=0)
    
    return inputs, activations, outputs

def visualize_pca_distributions(inputs, activations, outputs):
    """
    Reduce dimensionality of inputs, activations and outputs using PCA
    and visualize their distributions in 2D
    """
    from sklearn.decomposition import PCA
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Process and plot each array
    for data, ax, title in zip(
        [inputs, activations, outputs],
        [ax1, ax2, ax3],
        ['Input Distribution', 'Hidden Layer Activations', 'Output Distribution']
    ):
        # Apply PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        
        # Create scatter plot
        scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5, s=5)
        ax.set_title(title)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
def visualize_pca_distributions_with_colors(inputs, activations, outputs):
    """
    Reduce dimensionality using PCA and visualize distributions in 2D with actual colors
    
    Args:
        inputs: RGB input values
        activations: Hidden layer activation patterns
        outputs: HSV output values
    """
    from sklearn.decomposition import PCA
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Process and plot each array
    data_arrays = [inputs, activations, outputs]
    axes = [ax1, ax2, ax3]
    titles = ['Input Distribution (RGB)', 'Hidden Layer Activations', 'Output Distribution (HSV)']
    
    for data, ax, title in zip(data_arrays, axes, titles):
        # Apply PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        
        if title.startswith('Input'):
            # Ensure RGB values are within [0,1]
            rgb_colors = np.clip(inputs, 0, 1)
            scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                               c=rgb_colors, alpha=0.5, s=5)
        elif title.startswith('Output'):
            # Convert HSV to RGB and ensure values are within [0,1]
            rgb_colors = np.array([hsv_to_rgb(h % 1.0, min(1, max(0, s)), min(1, max(0, v))) 
                                 for h, s, v in outputs])
            scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                               c=rgb_colors, alpha=0.5, s=5)
        else:
            # Use PCA values for coloring activation space
            scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                               c=data_2d[:, 0], cmap='viridis', 
                               alpha=0.5, s=5)
            
        ax.set_title(title)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
def visualize_activations_3d(inputs, activations, outputs):
    """
    Reduce activations to 3D using PCA and visualize them in 3D space with colors from RGB inputs
    
    Args:
        inputs: RGB input values
        activations: Hidden layer activation patterns
        outputs: HSV output values (not used, kept for consistency)
    """
    from sklearn.decomposition import PCA
    
    # Apply PCA to reduce activations to 3 dimensions
    pca = PCA(n_components=3)
    activations_3d = pca.fit_transform(activations)
    
    # Create 3D plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with RGB colors
    scatter = ax.scatter(
        activations_3d[:, 0],
        activations_3d[:, 1],
        activations_3d[:, 2],
        c=inputs,  # Use RGB values for colors
        marker='o',
        s=1,
        alpha=0.6
    )
    
    # Set labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
    ax.set_title('Activation Space (3D PCA) colored by RGB inputs')
    
    # Add total explained variance in the title
    total_var = sum(pca.explained_variance_ratio_)
    plt.suptitle(f'Total explained variance: {total_var:.2%}')
    
    plt.show()

def visualize_new_sample_pca(base_inputs, base_activations, base_outputs, 
                           new_input, new_activations, new_output):
    """
    Visualize where a new sample falls in the PCA space of the base distributions
    """
    from sklearn.decomposition import PCA
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Process and plot each array pair
    for base_data, new_data, ax, title in zip(
        [base_inputs, base_activations, base_outputs],
        [new_input, new_activations, new_output],
        [ax1, ax2, ax3],
        ['Input Distribution', 'Hidden Layer Activations', 'Output Distribution']
    ):
        # Fit PCA on base data
        pca = PCA(n_components=2)
        base_2d = pca.fit_transform(base_data)
        
        # Transform new sample using the same PCA
        new_2d = pca.transform(new_data)
        
        # Plot base distribution
        scatter = ax.scatter(base_2d[:, 0], base_2d[:, 1], alpha=0.5, s=5, c='blue')
        
        # Plot new sample as red dot
        ax.scatter(new_2d[:, 0], new_2d[:, 1], c='red', s=50, label='New Sample')
        
        ax.set_title(title)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
def get_new_sample_data(model):
    """
    Generate a new random color sample and collect its activations
    """
    # Generate new random RGB color
    new_rgb = torch.FloatTensor(np.random.uniform(0, 1, (1, 3)))
    
    # Dictionary to store activations
    activation_dict = {}
    
    # Hook function
    def hook_fn(module, input, output):
        activation_dict[len(activation_dict)] = output.detach().cpu().numpy()
    
    # Register hooks for all layers except the final one
    hooks = []
    for layer in list(model.model)[:-1]:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # Get model prediction and activations
    model.eval()
    with torch.no_grad():
        new_output = model(new_rgb)
    
    # Concatenate activations
    new_activations = np.concatenate([act.reshape(act.shape[0], -1) 
                                    for act in list(activation_dict.values())], axis=1)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return new_rgb.numpy(), new_activations, new_output.numpy()

def evaluate_center_point(model, center_color):
    """
    Evaluate model performance on center point used for filtering
    
    Args:
        model: Trained RGB to HSV model
        center_color: RGB values of center point used for filtering
        
    Returns:
        dict: Evaluation metrics
    """
    # Convert to tensor and add batch dimension
    center_tensor = torch.FloatTensor(center_color).unsqueeze(0)
    
    # Get true HSV values
    true_hsv = rgb_to_hsv(*center_color)
    true_hsv_tensor = torch.FloatTensor(true_hsv).unsqueeze(0)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        pred_hsv = model(center_tensor)
    
    # Calculate L2 distance and MSE
    l2_distance = torch.norm(pred_hsv - true_hsv_tensor)
    mse = nn.MSELoss()(pred_hsv, true_hsv_tensor)
    
    print("\nCenter Point Evaluation:")
    print(f"RGB input:      {center_color}")
    print(f"True HSV:       {true_hsv}")
    print(f"Predicted HSV:  {pred_hsv.numpy()[0]}")
    print(f"L2 Distance:    {l2_distance.item():.6f}")
    print(f"MSE Loss:       {mse.item():.6f}")
    
    return {
        'l2_distance': l2_distance.item(),
        'mse_loss': mse.item()
    }

def filter_rgb_data(rgb_data, center_color, keep_percentage):
    """
    Filter out RGB data points based on distance from center color, keeping specified percentage
    
    Args:
        rgb_data: numpy array of RGB values, shape (n_samples, 3)
        center_color: RGB color to center the filtering around, shape (3,)
        keep_percentage: Percentage of samples to keep (0-100)
        
    Returns:
        filtered_rgb: Filtered RGB data with samples kept
        mask: Boolean mask indicating which samples were kept (True) or removed (False)
    """
    # Convert inputs to numpy arrays if they aren't already
    rgb_data = np.array(rgb_data)
    center_color = np.array(center_color)
    
    # Calculate Euclidean distances from center color to all samples
    distances = np.sqrt(np.sum((rgb_data - center_color) ** 2, axis=1))
    
    # Sort distances and find threshold for desired percentage
    sorted_distances = np.sort(distances)
    n_keep = int(len(rgb_data) * (keep_percentage / 100))
    threshold = sorted_distances[n_keep - 1]
    
    # Create mask for samples to keep
    mask = distances > threshold
    
    # Apply mask to get filtered data
    filtered_rgb = rgb_data[mask]
    
    # Print statistics
    kept_count = len(filtered_rgb)
    total_count = len(rgb_data)
    actual_percentage = (kept_count / total_count) * 100
    
    print(f"Kept {kept_count} samples ({actual_percentage:.1f}%) "
          f"outside radius {threshold:.3f} of {center_color}")
    
    return filtered_rgb, mask, threshold

def evaluate_with_decreasing_percentage(n_samples=1000, batch_size=64, 
                                     start_percentage=30, step=5, n_epochs=100,
                                     n_test_samples=10):
    """
    Train models with decreasing percentages of data and evaluate performance
    on multiple random test samples
    """
    results = []
    test_colors = [np.random.uniform(0, 1, 3) for _ in range(n_test_samples)]
    print(f"Using {n_test_samples} random test colors:")
    for i, color in enumerate(test_colors):
        print(f"Color {i+1}: {color}")
    
    current_percentage = start_percentage
    while current_percentage > 0:
        print(f"\n{'='*50}")
        print(f"Training with {current_percentage}% data")
        print(f"{'='*50}")
        
        percentage_metrics = {
            'percentage': current_percentage,
            'l2_distances': []
        }
        
        # Train and evaluate for each test color
        for i, center_color in enumerate(test_colors):
            print(f"\nTraining for test color {i+1}: {center_color}")
            
            # Generate and filter data for current test color
            rgb_data = generate_rgb_samples(n_samples)
            filtered_rgb, mask, threshold = filter_rgb_data(rgb_data, center_color, current_percentage)
            hsv_data = rgb_to_hsv_batch(filtered_rgb)
            
            # Create and train model
            model = RGBtoHSV()
            dataset = RGBHSVDataset(filtered_rgb, hsv_data)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            train_model(model, train_loader, num_epochs=n_epochs)
            
            # Evaluate on center color
            eval_metrics = evaluate_center_point(model, center_color)
            percentage_metrics['l2_distances'].append(eval_metrics['l2_distance'])
        
        results.append(percentage_metrics)
        current_percentage -= step
    
    # Plot results
    percentages = [r['percentage'] for r in results]
    all_l2_distances = np.array([r['l2_distances'] for r in results])
    mean_l2_distances = np.mean(all_l2_distances, axis=1)
    
    plt.figure(figsize=(12, 5))
    
    # Individual curves
    plt.subplot(121)
    for i in range(n_test_samples):
        plt.plot(percentages[::-1], all_l2_distances[::-1, i], 
                alpha=0.5, label=f'Sample {i+1}')
    plt.xlabel('Percentage of Data Used')
    plt.ylabel('L2 Distance')
    plt.title('Individual L2 Distances')
    plt.grid(True)
    plt.legend()
    
    # Average curve
    plt.subplot(122)
    plt.plot(percentages[::-1], mean_l2_distances[::-1], 'b-o')
    plt.xlabel('Percentage of Data Used')
    plt.ylabel('Average L2 Distance')
    plt.title('Average L2 Distance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

def find_closest_samples(base_inputs, base_activations, base_outputs,
                        new_input, new_activations, new_output):
    """
    Find the closest samples from the base distributions to a new sample
    using L2 distance for inputs, activations, and outputs
    
    Args:
        base_inputs: Array of all training input samples
        base_activations: Array of all training activation patterns
        base_outputs: Array of all training output samples
        new_input: Single new input sample
        new_activations: Single new activation pattern
        new_output: Single new output sample
        
    Returns:
        tuple: (closest_input_idx, closest_activation_idx, closest_output_idx),
               (input_distance, activation_distance, output_distance)
    """
    # Reshape new samples to match base dimensions
    new_input = new_input.reshape(1, -1)
    new_activations = new_activations.reshape(1, -1)
    new_output = new_output.reshape(1, -1)
    
    # Calculate L2 distances for each distribution
    input_distances = np.sqrt(np.sum((base_inputs - new_input) ** 2, axis=1))
    activation_distances = np.sqrt(np.sum((base_activations - new_activations) ** 2, axis=1))
    output_distances = np.sqrt(np.sum((base_outputs - new_output) ** 2, axis=1))
    
    # Find indices of closest samples
    closest_input_idx = np.argmin(input_distances)
    closest_activation_idx = np.argmin(activation_distances)
    closest_output_idx = np.argmin(output_distances)
    
    # Get minimum distances
    min_distances = (
        input_distances[closest_input_idx],
        activation_distances[closest_activation_idx],
        output_distances[closest_output_idx]
    )
    
    # Print results
    print("\nClosest samples found:")
    print(f"Input space - Distance: {min_distances[0]:.4f}")
    print(f"RGB values - New: {new_input[0]}, Closest: {base_inputs[closest_input_idx]}")
    
    print(f"\nActivation space - Distance: {min_distances[1]:.4f}")
    
    print(f"\nOutput space - Distance: {min_distances[2]:.4f}")
    print(f"HSV values - New: {new_output[0]}, Closest: {base_outputs[closest_output_idx]}")
    
    return (closest_input_idx, closest_activation_idx, closest_output_idx), min_distances

def find_k_nearest_neighbors(base_inputs, base_activations, base_outputs,
                           new_input, new_activations, new_output, k=15, verbose=True):
    """
    Find k nearest neighbors from the base distributions
    
    Args:
        base_inputs, base_activations, base_outputs: Base distribution arrays
        new_input, new_activations, new_output: New sample data
        k: Number of nearest neighbors to find (default=15)
        
    Returns:
        tuple: (input_indices, activation_indices, output_indices),
               (input_distances, activation_distances, output_distances)
    """
    # Reshape new samples
    new_input = new_input.reshape(1, -1)
    new_activations = new_activations.reshape(1, -1)
    new_output = new_output.reshape(1, -1)
    
    # Calculate L2 distances
    input_distances = np.sqrt(np.sum((base_inputs - new_input) ** 2, axis=1))
    activation_distances = np.sqrt(np.sum((base_activations - new_activations) ** 2, axis=1))
    output_distances = np.sqrt(np.sum((base_outputs - new_output) ** 2, axis=1))
    
    # Get k nearest indices and distances
    input_idx = np.argsort(input_distances)[:k]
    activation_idx = np.argsort(activation_distances)[:k]
    output_idx = np.argsort(output_distances)[:k]
    
    nearest_distances = (
        input_distances[input_idx],
        activation_distances[activation_idx],
        output_distances[output_idx]
    )
    
    if verbose:
        print(f"\nFound {k} nearest neighbors:")
        print(f"Input space - Distance range: {nearest_distances[0].min():.4f} to {nearest_distances[0].max():.4f}")
        print(f"Activation space - Distance range: {nearest_distances[1].min():.4f} to {nearest_distances[1].max():.4f}")
        print(f"Output space - Distance range: {nearest_distances[2].min():.4f} to {nearest_distances[2].max():.4f}")
    
    return (input_idx, activation_idx, output_idx), nearest_distances

def blend_nearest_neighbors(base_inputs, base_activations, base_outputs, indices, distances, verbose=True):
    """
    Create weighted blend of nearest neighbors in input, activation, and output spaces
    
    Args:
        base_inputs: Base RGB values
        base_activations: Base activation patterns
        base_outputs: Base HSV values
        indices: Tuple of indices for input, activation, output neighbors
        distances: Tuple of distances for input, activation, output neighbors
        
    Returns:
        dict: Blended values for each space (input_rgb, activations, output_hsv)
    """
    def compute_weights(distances):
        # Convert distances to weights using inverse distance weighting
        weights = 1 / (distances + 1e-6)  # Add small epsilon to avoid division by zero
        return weights / weights.sum()  # Normalize weights to sum to 1
    
    # Compute weighted averages for each space
    input_weights = compute_weights(distances[0])
    activation_weights = compute_weights(distances[1])
    output_weights = compute_weights(distances[2])
    
    # Blend RGB values (inputs)
    blended_rgb = np.average(base_inputs[indices[0]], weights=input_weights, axis=0)
    
    # Blend activations
    blended_activations = np.average(base_activations[indices[1]], weights=activation_weights, axis=0)
    
    # Blend HSV values (outputs)
    blended_hsv = np.average(base_outputs[indices[2]], weights=output_weights, axis=0)
    
    if verbose:
        print("\nBlended values:")
        print(f"RGB (input space): {blended_rgb}")
        print(f"Activation dimensions: {blended_activations.shape}")
        print(f"HSV (output space): {blended_hsv}")
    
    return {
        'blended_rgb': blended_rgb,
        'blended_activations': blended_activations,
        'blended_hsv': blended_hsv
    }

def compare_blended_values(blended_rgb, blended_activations, blended_hsv, 
                         true_rgb, true_activations, true_hsv, verbose=True):
    """
    Compare blended values with the true values from the original sample
    
    Args:
        blended_rgb: Blended RGB values from nearest neighbors
        blended_activations: Blended activation patterns from nearest neighbors
        blended_hsv: Blended HSV values from nearest neighbors
        true_rgb: Original input RGB values
        true_activations: Original activation patterns
        true_hsv: Original output HSV values
        
    Returns:
        dict: Distances between blended and true values
    """
    # Reshape true values if needed
    true_rgb = true_rgb.reshape(-1)
    true_hsv = true_hsv.reshape(-1)
    
    # Calculate L2 distances
    rgb_distance = np.sqrt(np.sum((blended_rgb - true_rgb) ** 2))
    hsv_distance = np.sqrt(np.sum((blended_hsv - true_hsv) ** 2))
    activation_distance = np.sqrt(np.sum((blended_activations - true_activations) ** 2))
    
    if verbose:
        print("\nComparing blended values with true values:")
        print(f"RGB space:")
        print(f"  Blended RGB: {blended_rgb}")
        print(f"  True RGB:    {true_rgb}")
        print(f"\nActivation patterns:")
        print(f"  Blended shape: {blended_activations.shape}")
        print(f"  True shape:    {true_activations.shape}")
        print(f"\nHSV space:")
        print(f"  Blended HSV: {blended_hsv}")
        print(f"  True HSV:    {true_hsv}")
        print(f"\nL2 Distances:")
        print(f"RGB space (blended vs true):        {rgb_distance:.4f}")
        print(f"Activation space (blended vs true): {activation_distance:.4f}")
        print(f"HSV space (blended vs true):        {hsv_distance:.4f}")
    
    return {
        'rgb_distance': rgb_distance,
        'hsv_distance': hsv_distance,
        'activation_distance': activation_distance
    }
    
def evaluate_k_range(model, base_inputs, base_activations, base_outputs, n_samples=50, max_k=50):
    """
    Evaluate different k values for nearest neighbor blending
    
    Args:
        model: Trained model
        base_inputs, base_activations, base_outputs: Training data
        n_samples: Number of test samples to evaluate
        max_k: Maximum number of neighbors to try
    
    Returns:
        dict: Average distances for each k value and space
    """
    # Storage for results
    results = {
        'k_values': list(range(1, max_k + 1)),
        'rgb_distances': np.zeros(max_k),
        'activation_distances': np.zeros(max_k),
        'hsv_distances': np.zeros(max_k)
    }
    
    # Generate test samples
    for sample in range(n_samples):
        test = get_new_sample_data(model)
        for k in range(1, max_k + 1):
            indices, distances = find_k_nearest_neighbors(
                base_inputs, base_activations, base_outputs,
                test[0], test[1], test[2], k=k, verbose=False
            )
            
            blended = blend_nearest_neighbors(
                base_inputs, base_activations, base_outputs,
                indices, distances, verbose=False
            )
            
            comparison = compare_blended_values(
                blended['blended_rgb'],
                blended['blended_activations'],
                blended['blended_hsv'],
                test[0], test[1], test[2], verbose=False
            )
            
            # Accumulate distances
            results['rgb_distances'][k-1] += comparison['rgb_distance']
            results['activation_distances'][k-1] += comparison['activation_distance']
            results['hsv_distances'][k-1] += comparison['hsv_distance']
            
        print(f"Processed sample {sample+1}/{n_samples}")
    
    # Average distances
    results['rgb_distances'] /= n_samples
    results['activation_distances'] /= n_samples
    results['hsv_distances'] /= n_samples
    
    return results

def plot_k_evaluation(results):
    """
    Plot the results of k-value evaluation
    
    Args:
        results: Dictionary containing evaluation results
    """
    plt.figure(figsize=(12, 6))
    
    # Plot distances for each space
    plt.plot(results['k_values'], results['rgb_distances'], 
             label='RGB Space', marker='o')
    plt.plot(results['k_values'], results['activation_distances'], 
             label='Activation Space', marker='s')
    plt.plot(results['k_values'], results['hsv_distances'], 
             label='HSV Space', marker='^')
    
    # Find and mark best k values
    best_k_rgb = results['k_values'][np.argmin(results['rgb_distances'])]
    best_k_act = results['k_values'][np.argmin(results['activation_distances'])]
    best_k_hsv = results['k_values'][np.argmin(results['hsv_distances'])]
    
    plt.axvline(x=best_k_rgb, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(x=best_k_act, color='orange', linestyle=':', alpha=0.5)
    plt.axvline(x=best_k_hsv, color='green', linestyle=':', alpha=0.5)
    
    plt.title('Average L2 Distance vs Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Average L2 Distance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text annotations for best k values
    plt.text(0.02, 0.98, f'Best k values:\nRGB: {best_k_rgb}\nActivation: {best_k_act}\nHSV: {best_k_hsv}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return best_k_rgb, best_k_act, best_k_hsv

def evaluate_training_sizes(training_sizes=[500, 2000, 5000], n_epochs=100, batch_size=64, 
                          n_test_samples=100, max_k=30):
    """
    Evaluate how different training set sizes affect the model and kNN performance
    
    Args:
        training_sizes: List of training set sizes to evaluate
        n_epochs: Number of epochs for training each model
        batch_size: Batch size for training
        n_test_samples: Number of test samples for kNN evaluation
        max_k: Maximum number of neighbors to try
        
    Returns:
        dict: Results for each training size
    """
    summary = {}
    
    for n_samples in training_sizes:
        print(f"\n{'='*50}")
        print(f"Training with {n_samples} samples")
        print(f"{'='*50}")
        
        # 1. Prepare training data
        rgb_data = generate_rgb_samples(n_samples)
        hsv_data = rgb_to_hsv_batch(rgb_data)
        dataset = RGBHSVDataset(rgb_data, hsv_data)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 2. Create and train model
        model = RGBtoHSV()
        train_model(model, train_loader, num_epochs=n_epochs)
        
        # 3. Collect activations
        inputs, activations, outputs = collect_activations(model, train_loader)
        
        # 4. Evaluate kNN performance
        knn_results = evaluate_k_range(model, inputs, activations, outputs, 
                                     n_samples=n_test_samples, max_k=max_k)
        
        # 5. Find best k values
        best_k_rgb = np.argmin(knn_results['rgb_distances']) + 1
        best_k_act = np.argmin(knn_results['activation_distances']) + 1
        best_k_hsv = np.argmin(knn_results['hsv_distances']) + 1
        
        # 6. Store results
        summary[n_samples] = {
            'best_k_values': {
                'rgb': best_k_rgb,
                'activation': best_k_act,
                'hsv': best_k_hsv
            },
            'min_distances': {
                'rgb': knn_results['rgb_distances'][best_k_rgb-1],
                'activation': knn_results['activation_distances'][best_k_act-1],
                'hsv': knn_results['hsv_distances'][best_k_hsv-1]
            }
        }
        
        # 7. Print summary for this training size
        print(f"\nResults for {n_samples} training samples:")
        print(f"Best k values:")
        print(f"  RGB space: {best_k_rgb} (distance: {summary[n_samples]['min_distances']['rgb']:.4f})")
        print(f"  Activation space: {best_k_act} (distance: {summary[n_samples]['min_distances']['activation']:.4f})")
        print(f"  HSV space: {best_k_hsv} (distance: {summary[n_samples]['min_distances']['hsv']:.4f})")
    
    return summary

def evaluate_and_compare_training_sizes(training_sizes=[500, 2000, 5000], n_epochs=100, 
                                      batch_size=64, n_compare_samples=100, k_neighbors=15):
    """
    Train models with different dataset sizes and compare closest vs blended samples
    
    Args:
        training_sizes: List of training set sizes to evaluate
        n_epochs: Number of epochs for training
        batch_size: Batch size for training
        n_compare_samples: Number of samples to use for comparing methods
        k_neighbors: Number of neighbors to use for blending
    """
    # Storage for results
    results = {size: {'closest': {'rgb': [], 'activation': [], 'hsv': []},
                     'blended': {'rgb': [], 'activation': [], 'hsv': []}} 
              for size in training_sizes}
    
    for n_samples in training_sizes:
        print(f"\n{'='*50}")
        print(f"Training with {n_samples} samples")
        print(f"{'='*50}")
        
        # 1. Prepare training data
        rgb_data = generate_rgb_samples(n_samples)
        hsv_data = rgb_to_hsv_batch(rgb_data)
        dataset = RGBHSVDataset(rgb_data, hsv_data)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 2. Train model
        model = RGBtoHSV()
        train_model(model, train_loader, num_epochs=n_epochs)
        
        # 3. Collect activations
        inputs, activations, outputs = collect_activations(model, train_loader)
        
        # 4. Compare methods over multiple samples
        for i in range(n_compare_samples):
            # Get new test sample
            test = get_new_sample_data(model)
            
            # Find closest samples - FIXED unpacking
            closest_indices, distances = find_closest_samples(
                inputs, activations, outputs,
                test[0], test[1], test[2]
            )
            
            # Store closest distances
            results[n_samples]['closest']['rgb'].append(distances[0])
            results[n_samples]['closest']['activation'].append(distances[1])
            results[n_samples]['closest']['hsv'].append(distances[2])
            
            # Find and blend k nearest neighbors
            indices, distances = find_k_nearest_neighbors(
                inputs, activations, outputs,
                test[0], test[1], test[2],
                k=k_neighbors, verbose=False
            )
            
            blended = blend_nearest_neighbors(
                inputs, activations, outputs,
                indices, distances, verbose=False
            )
            
            comparison = compare_blended_values(
                blended['blended_rgb'],
                blended['blended_activations'],
                blended['blended_hsv'],
                test[0], test[1], test[2],
                verbose=False
            )
            
            # Store blended distances
            results[n_samples]['blended']['rgb'].append(comparison['rgb_distance'])
            results[n_samples]['blended']['activation'].append(comparison['activation_distance'])
            results[n_samples]['blended']['hsv'].append(comparison['hsv_distance'])
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{n_compare_samples} samples")
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot settings
    spaces = ['RGB', 'Activation', 'HSV']
    axes = [ax1, ax2, ax3]
    data_keys = ['rgb', 'activation', 'hsv']
    
    for ax, space, key in zip(axes, spaces, data_keys):
        positions = []
        closest_data = []
        blended_data = []
        
        for i, size in enumerate(training_sizes):
            base_pos = i * 3
            positions.extend([base_pos + 0.5, base_pos + 1.5])
            closest_data.append(results[size]['closest'][key])
            blended_data.append(results[size]['blended'][key])
        
        # Create boxplots
        bp = ax.boxplot(closest_data + blended_data,
                       positions=positions,
                       widths=0.6)
        
        # Customize plot
        ax.set_title(f'{space} Space')
        ax.set_ylabel('L2 Distance')
        ax.set_xticks([i * 3 + 1 for i in range(len(training_sizes))])
        ax.set_xticklabels(training_sizes)
        ax.set_xlabel('Training Size')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Different colors for closest and blended
        plt.setp(bp['boxes'][::2], color='blue', alpha=0.6)
        plt.setp(bp['boxes'][1::2], color='green', alpha=0.6)
    
    # Add legend
    fig.legend(['Closest', 'Blended'], 
              loc='upper center', 
              bbox_to_anchor=(0.5, 1.05),
              ncol=2)
    
    plt.tight_layout()
    plt.show()
    
    return results

def collect_single_color_activations(model, rgb_color):
    """
    Collect activations from the network for a single RGB color input
    
    Args:
        model: The trained neural network model
        rgb_color: RGB color values as numpy array or list [r, g, b]
        
    Returns:
        tuple: (input_rgb, activations, output_hsv)
            - input_rgb: The input RGB values
            - activations: Hidden layer activation patterns
            - output_hsv: The output HSV values
    """
    # Convert input to tensor and add batch dimension
    rgb_tensor = torch.FloatTensor(rgb_color).unsqueeze(0)
    
    # Dictionary to store activations
    activation_dict = {}
    
    # Hook function to capture activations
    def hook_fn(module, input, output):
        activation_dict[len(activation_dict)] = output.detach().cpu().numpy()
    
    # Register hooks for all layers except the final one
    hooks = []
    for layer in list(model.model)[:-1]:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # Get model prediction and activations
    model.eval()
    with torch.no_grad():
        output = model(rgb_tensor)
    
    # Concatenate activations
    activations = np.concatenate([act.reshape(act.shape[0], -1) 
                                for act in list(activation_dict.values())], axis=1)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return (rgb_tensor.numpy(), activations, output.numpy())

def find_k_nearest_neighbors(base_inputs, base_activations, base_outputs,
                           new_input, new_activations, new_output, k=15, verbose=True):
    """
    Find k nearest neighbors from the base distributions
    
    Args:
        base_inputs, base_activations, base_outputs: Base distribution arrays
        new_input, new_activations, new_output: New sample data
        k: Number of nearest neighbors to find (default=15)
        
    Returns:
        dict: Dictionary containing nearest neighbors and their distances for each space:
            {
                'input': {'samples': array, 'distances': array},
                'activation': {'samples': array, 'distances': array},
                'output': {'samples': array, 'distances': array}
            }
    """
    # Reshape new samples
    new_input = new_input.reshape(1, -1)
    new_activations = new_activations.reshape(1, -1)
    new_output = new_output.reshape(1, -1)
    
    # Calculate L2 distances
    input_distances = np.sqrt(np.sum((base_inputs - new_input) ** 2, axis=1))
    activation_distances = np.sqrt(np.sum((base_activations - new_activations) ** 2, axis=1))
    output_distances = np.sqrt(np.sum((base_outputs - new_output) ** 2, axis=1))
    
    # Get k nearest indices and sort distances
    input_idx = np.argsort(input_distances)[:k]
    activation_idx = np.argsort(activation_distances)[:k]
    output_idx = np.argsort(output_distances)[:k]
    
    # Create results dictionary
    results = {
        'input': {
            'samples': base_inputs[input_idx],
            'distances': input_distances[input_idx]
        },
        'activation': {
            'samples': base_activations[activation_idx],
            'distances': activation_distances[activation_idx]
        },
        'output': {
            'samples': base_outputs[output_idx],
            'distances': output_distances[output_idx]
        }
    }
    
    if verbose:
        print(f"\nFound {k} nearest neighbors:")
        for space in ['input', 'activation', 'output']:
            dists = results[space]['distances']
            print(f"{space.capitalize()} space - Distance range: "
                  f"{dists.min():.4f} to {dists.max():.4f}")
    
    return results

def find_k_nearest_neighbors_with_indices(base_inputs, base_activations, base_outputs,
                                        new_input, new_activations, new_output, k=15, verbose=True):
    """
    Find k nearest neighbors from the base distributions and return indices
    
    Args:
        base_inputs, base_activations, base_outputs: Base distribution arrays
        new_input, new_activations, new_output: New sample data
        k: Number of nearest neighbors to find (default=15)
        verbose: Whether to print distance information
        
    Returns:
        dict: Dictionary containing nearest neighbors, distances, and indices for each space:
            {
                'input': {'samples': array, 'distances': array, 'indices': array},
                'activation': {'samples': array, 'distances': array, 'indices': array},
                'output': {'samples': array, 'distances': array, 'indices': array}
            }
    """
    # Reshape new samples
    new_input = new_input.reshape(1, -1)
    new_activations = new_activations.reshape(1, -1)
    new_output = new_output.reshape(1, -1)
    
    # Calculate L2 distances
    input_distances = np.sqrt(np.sum((base_inputs - new_input) ** 2, axis=1))
    activation_distances = np.sqrt(np.sum((base_activations - new_activations) ** 2, axis=1))
    output_distances = np.sqrt(np.sum((base_outputs - new_output) ** 2, axis=1))
    
    # Get k nearest indices and sort distances
    input_idx = np.argsort(input_distances)[:k]
    activation_idx = np.argsort(activation_distances)[:k]
    output_idx = np.argsort(output_distances)[:k]
    
    # Create results dictionary
    results = {
        'input': {
            'samples': base_inputs[input_idx],
            'distances': input_distances[input_idx],
            'indices': input_idx
        },
        'activation': {
            'samples': base_activations[activation_idx],
            'distances': activation_distances[activation_idx],
            'indices': activation_idx
        },
        'output': {
            'samples': base_outputs[output_idx],
            'distances': output_distances[output_idx],
            'indices': output_idx
        }
    }
    
    if verbose:
        print(f"\nFound {k} nearest neighbors:")
        for space in ['input', 'activation', 'output']:
            dists = results[space]['distances']
            print(f"{space.capitalize()} space - Distance range: "
                  f"{dists.min():.4f} to {dists.max():.4f}")
    
    return results

def find_neighbors_by_neurons(base_activations, target_activations, n_neurons=64, neuron_start=0, k=10):
    """
    Find k nearest neighbors using only specified activation neurons
    
    Args:
        base_activations: Base activation patterns array
        target_activations: Target activation pattern
        n_neurons: Number of neurons to consider (default=64)
        neuron_start: Starting index for neuron selection (default=0)
        k: Number of nearest neighbors to find (default=10)
        
    Returns:
        dict: Dictionary with nearest neighbors data:
            {
                'samples': array of k nearest activation patterns,
                'distances': array of k distances,
                'indices': array of k indices
            }
    """
    # Reshape target activations
    target = np.array(target_activations).reshape(1, -1)
    
    # Select neuron subset
    neuron_end = min(neuron_start + n_neurons, base_activations.shape[1])
    selected_base = base_activations[:, neuron_start:neuron_end]
    selected_target = target[:, neuron_start:neuron_end]
    
    # Calculate distances using selected neurons
    distances = np.sqrt(np.sum((selected_base - selected_target) ** 2, axis=1))
    
    # Get k nearest indices and sort distances
    nearest_idx = np.argsort(distances)[:k]
    nearest_distances = distances[nearest_idx]
    
    return {
        'samples': base_activations[nearest_idx],
        'distances': nearest_distances,
        'indices': nearest_idx
    }

def blend_nearest_neighbors(space_type, neighbors, distances, target_value):
    """
    Blend nearest neighbors based on their distances and compare with target value
    
    Args:
        space_type: String indicating the space type ('input', 'activation', 'output')
        neighbors: Array of nearest neighbor samples from specified space
        distances: Array of corresponding distances
        target_value: The value to compare against (input RGB, activation, or output HSV)
        
    Returns:
        dict: Results containing blended value, weights, and comparison metrics
    """
    # Convert distances to weights (inverse distance weighting)
    weights = 1 / (distances + 1e-6)  # Add small epsilon to avoid division by zero
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    
    # Compute weighted average
    blended_value = np.sum(neighbors * weights[:, np.newaxis], axis=0)
    
    # Calculate L2 distance between blended result and target
    difference = np.sqrt(np.sum((blended_value - target_value) ** 2))
    
    # Prepare results
    results = {
        'space_type': space_type,
        'blended_value': blended_value,
        'weights': weights,
        'difference': difference
    }
    
    # Print results
    print(f"\nBlending Results for {space_type.capitalize()} Space:")
    print(f"Target value: {target_value.flatten()}")
    print(f"Blended value: {blended_value}")
    print(f"L2 difference: {difference:.4f}")
    
    return blended_value

def calculate_distance(value1, value2):
    """
    Calculate L2 distance between two values (can be RGB, activations, or HSV)
    
    Args:
        value1: First value (numpy array)
        value2: Second value (numpy array)
        
    Returns:
        float: L2 distance between the values
    """
    # Ensure inputs are numpy arrays and flatten
    value1 = np.array(value1).reshape(1, -1)
    value2 = np.array(value2).reshape(1, -1)
    
    # Calculate L2 distance
    distance = np.sqrt(np.sum((value1 - value2) ** 2))
    
    return distance

# def compare_values_side_by_side(value1, value2, titles=('Value 1', 'Value 2'), color_space='rgb', decimals=2):
#     """
#     Visualize two values side by side with formatted display
    
#     Args:
#         value1: First value (RGB or HSV array/list)
#         value2: Second value (RGB or HSV array/list)
#         titles: Tuple of titles for the two values
#         color_space: 'rgb' or 'hsv' - determines color conversion for display
#         decimals: Number of decimal places to show in titles
#     """
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 1.5))
    
#     # Format values for display
#     formatted_value1 = np.round(value1, decimals)
#     formatted_value2 = np.round(value2, decimals)
    
#     # Convert HSV to RGB for display if needed
#     if color_space.lower() == 'hsv':
#         display_color1 = hsv_to_rgb(*value1)
#         display_color2 = hsv_to_rgb(*value2)
#         space_name = 'HSV'
#     else:
#         display_color1 = value1
#         display_color2 = value2
#         space_name = 'RGB'
    
#     # Plot first value
#     ax1.add_patch(plt.Rectangle((0, 0), 1, 1, color=display_color1))
#     ax1.set_title(f'{titles[0]} ({space_name}): {formatted_value1}')
#     ax1.axis('equal')
    
#     # Plot second value
#     ax2.add_patch(plt.Rectangle((0, 0), 1, 1, color=display_color2))
#     ax2.set_title(f'{titles[1]} ({space_name}): {formatted_value2}')
#     ax2.axis('equal')
    
#     plt.tight_layout()
#     plt.show()
    
def compare_values_side_by_side(value1, value2, titles=('Value 1', 'Value 2'), color_space='rgb', decimals=2):
    """
    Visualize two values side by side with formatted display
    
    Args:
        value1: First value (RGB or HSV array/list)
        value2: Second value (RGB or HSV array/list)
        titles: Tuple of titles for the two values
        color_space: 'rgb' or 'hsv' - determines color conversion for display
        decimals: Number of decimal places to show in titles
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 1.5))
    
    # Format values for display
    formatted_value1 = np.round(value1, decimals)
    formatted_value2 = np.round(value2, decimals)
    
    # Convert HSV to RGB for display if needed
    if color_space.lower() == 'hsv':
        display_color1 = np.clip(hsv_to_rgb(*value1), 0, 1)
        display_color2 = np.clip(hsv_to_rgb(*value2), 0, 1)
        space_name = 'HSV'
    else:
        display_color1 = np.clip(value1, 0, 1)
        display_color2 = np.clip(value2, 0, 1)
        space_name = 'RGB'
    
    # Plot first value
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, color=display_color1))
    ax1.set_title(f'{titles[0]} ({space_name}): {formatted_value1}')
    ax1.axis('equal')
    
    # Plot second value
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, color=display_color2))
    ax2.set_title(f'{titles[1]} ({space_name}): {formatted_value2}')
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
def blend_activation_neighbors_values(neuron_results, base_inputs, base_outputs):
    """
    Blend the RGB inputs and HSV outputs of the k-nearest activation neighbors
    
    Args:
        neuron_results: Dictionary from find_neighbors_by_neurons containing:
            - indices: indices of nearest neighbors
            - distances: distances to nearest neighbors
        base_inputs: Original RGB inputs array
        base_outputs: Original HSV outputs array
        
    Returns:
        tuple: (blended_rgb, blended_hsv)
            - blended_rgb: Weighted average of RGB values from activation neighbors
            - blended_hsv: Weighted average of HSV values from activation neighbors
    """
    # Get indices and distances
    indices = neuron_results['indices']
    distances = neuron_results['distances']
    
    # Get corresponding RGB and HSV values using indices
    rgb_values = base_inputs[indices]
    
    # Calculate weights based on distances
    weights = 1 / (distances + 1e-6)
    weights = weights / np.sum(weights)
    
    # Compute weighted averages using the same weights
    blended_rgb = np.sum(rgb_values * weights[:, np.newaxis], axis=0)
    blended_hsv = rgb_to_hsv(blended_rgb[0], blended_rgb[1], blended_rgb[2])
    
    return blended_rgb, blended_hsv

def analyze_activation_patterns(base_activations, target_activations, base_inputs, base_outputs, 
                             start_neurons=10, max_neurons=640, k=10):
    """
    Analyze how different numbers of neurons affect nearest neighbor finding and blending
    
    Args:
        base_activations: Base activation patterns array
        target_activations: Target activation pattern
        base_inputs: Original RGB inputs array
        base_outputs: Original HSV outputs array
        start_neurons: Number of neurons to start with (default=10)
        max_neurons: Maximum number of neurons to analyze (default=640)
        k: Number of nearest neighbors to find (default=10)
        
    Returns:
        dict: Results containing for each neuron count:
            {
                'n_neurons': list of neuron counts,
                'distances': list of L2 distances between target and blended result,
                'rgb_values': list of blended RGB values,
                'hsv_values': list of blended HSV values
            }
    """
    results = {
        'n_neurons': [],
        'distances': [],
        'rgb_values': [],
        'hsv_values': []
    }
    
    for n in range(start_neurons, max_neurons + 1):
        # Find nearest neighbors using current number of neurons
        neighbors = find_neighbors_by_neurons(
            base_activations,
            target_activations,
            n_neurons=n,
            neuron_start=0,
            k=k
        )
        
        # Blend values using found neighbors
        blended_rgb, blended_hsv = blend_activation_neighbors_values(
            neighbors,
            base_inputs,
            base_outputs
        )
        
        # Calculate distance using only the considered neurons
        target_subset = target_activations[0, :n]
        blended_subset = base_activations[neighbors['indices']][:, :n]
        weighted_blend = np.sum(blended_subset * neighbors['distances'][:, np.newaxis], axis=0)
        weighted_blend = weighted_blend / np.sum(neighbors['distances'])
        distance = np.sqrt(np.sum((target_subset - weighted_blend) ** 2))
        
        # Store results
        results['n_neurons'].append(n)
        results['distances'].append(distance)
        results['rgb_values'].append(blended_rgb)
        results['hsv_values'].append(blended_hsv)
    
    return results

def plot_neuron_progression(results):
    """
    Plot the results of the neuron progression analysis
    
    Args:
        results: Dictionary containing the analysis results
    """
    plt.figure(figsize=(15, 8))
    
    # Plot activation-based results
    plt.plot(results['neurons'], results['act_vs_input'], 
             'b-', label='Activation Blend → Input', alpha=0.7)
    plt.plot(results['neurons'], results['act_vs_output'], 
             'r-', label='Activation Blend → Output', alpha=0.7)
    
    # Plot baselines
    plt.axhline(y=results['input_baseline'], color='b', linestyle='--', 
                label='Input Blend Baseline', alpha=0.5)
    plt.axhline(y=results['output_baseline'], color='r', linestyle='--', 
                label='Output Blend Baseline', alpha=0.5)
    
    plt.xlabel('Number of Neurons Used')
    plt.ylabel('Mean L2 Distance')
    plt.title('Activation-Based Blending Performance vs. Neuron Count')
    plt.legend()
    plt.grid(True)
    
    # Add summary text
    min_act_input = min(results['act_vs_input'])
    min_act_output = min(results['act_vs_output'])
    best_n_input = results['neurons'][results['act_vs_input'].index(min_act_input)]
    best_n_output = results['neurons'][results['act_vs_output'].index(min_act_output)]
    
    plt.figtext(0.02, 0.02, 
                f'Best Results:\n'
                f'Input: {min_act_input:.4f} at {best_n_input} neurons\n'
                f'Output: {min_act_output:.4f} at {best_n_output} neurons',
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_activation_analysis(results):
    """
    Plot the results of the activation pattern analysis
    
    Args:
        results: Dictionary containing analysis results
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['n_neurons'], results['distances'], '-o')
    plt.xlabel('Number of Neurons Used')
    plt.ylabel('L2 Distance')
    plt.title('Activation Pattern Analysis')
    plt.grid(True)
    plt.show()
    
def analyze_blending_statistics(model, num_samples=100, k=10, n_neurons=64, verbose=True, train_loader=None):
    """
    Analyze blending results across multiple random samples
    
    Args:
        model: The trained neural network model
        num_samples: Number of random samples to analyze (default=100)
        k: Number of nearest neighbors to use (default=10)
        n_neurons: Number of neurons to use for activation-based blending (default=64)
        verbose: Whether to print progress updates (default=True)
        
    Returns:
        dict: Statistics containing distances for different blending approaches:
            {
                'input_to_output': list of distances from input-based blends to original outputs
                'output_to_input': list of distances from output-based blends to original inputs
                'act_to_input': list of distances from activation-based blends to original inputs
                'act_to_output': list of distances from activation-based blends to original outputs
            }
    """
    # Initialize results dictionary
    results = {
        'input_to_output': [],
        'output_to_input': [],
        'act_to_input': [],
        'act_to_output': []
    }
    
    # Get base distributions
    inputs, activations, outputs = collect_activations(model, train_loader)
    
    for i in range(num_samples):
        # Generate random color and get its values
        rgb_color = np.random.uniform(0, 1, 3)
        new_input, new_activations, new_output = collect_single_color_activations(model, rgb_color)
        
        # Get knn results for all spaces
        knn_results = find_k_nearest_neighbors(
            inputs, activations, outputs,
            new_input, new_activations, new_output,
            k=k,
            verbose=False
        )
        
        # Get blends based on input space
        input_blend = blend_nearest_neighbors(
            'input',
            knn_results['input']['samples'],
            knn_results['input']['distances'],
            new_input
        )
        input_blend_to_hsv = rgb_to_hsv(input_blend[0], input_blend[1], input_blend[2])
        
        # Get blends based on output space
        output_blend = blend_nearest_neighbors(
            'output',
            knn_results['output']['samples'],
            knn_results['output']['distances'],
            new_output
        )
        output_blend_to_rgb = hsv_to_rgb(output_blend[0], output_blend[1], output_blend[2])
        
        # Get blends based on activation space
        act_results = find_neighbors_by_neurons(
            activations,
            new_activations,
            n_neurons=n_neurons,
            k=k
        )
        act_blended_rgb, act_blended_hsv = blend_activation_neighbors_values(
            act_results,
            inputs,
            outputs
        )
        
        # Calculate and store distances
        results['input_to_output'].append(calculate_distance(input_blend_to_hsv, new_output[0]))
        results['output_to_input'].append(calculate_distance(output_blend_to_rgb, new_input[0]))
        results['act_to_input'].append(calculate_distance(act_blended_rgb, new_input[0]))
        results['act_to_output'].append(calculate_distance(act_blended_hsv, new_output[0]))
        
        if verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_samples} samples")
    
    return results

def plot_blending_statistics(stats):
    """
    Create a boxplot visualization of the blending statistics with summary table
    
    Args:
        stats: Dictionary containing the distance statistics
    """
    # Create figure with subplot grid
    fig = plt.figure(figsize=(15, 8))
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
    
    # Create boxplot in top subplot
    ax1 = plt.subplot(gs[0, :])
    
    # Prepare data and labels
    data = [
        stats['input_to_output'],
        stats['output_to_input'],
        stats['act_to_input'],
        stats['act_to_output']
    ]
    
    labels = [
        'Input Blend → Output',
        'Output Blend → Input',
        'Activation Blend → Input',
        'Activation Blend → Output'
    ]
    
    # Create boxplot
    bp = ax1.boxplot(data, labels=labels)
    
    # Customize plot
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylabel('L2 Distance')
    ax1.set_title('Blending Approach Comparison')
    ax1.grid(True, axis='y')
    
    # Create summary table in bottom subplot
    ax2 = plt.subplot(gs[1, :])
    ax2.axis('off')  # Hide axes
    
    # Prepare table data
    table_data = []
    metrics = ['Mean', 'Median', 'Std', 'Min', 'Max']
    
    for distances in data:
        stats_row = [
            f"{np.mean(distances):.4f}",
            f"{np.median(distances):.4f}",
            f"{np.std(distances):.4f}",
            f"{np.min(distances):.4f}",
            f"{np.max(distances):.4f}"
        ]
        table_data.append(stats_row)
    
    # Create table
    table = ax2.table(
        cellText=table_data,
        rowLabels=labels,
        colLabels=metrics,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nDetailed Blending Statistics:")
    for label, distances in zip(labels, data):
        print(f"\n{label}:")
        print(f"Mean: {np.mean(distances):.4f}")
        print(f"Median: {np.median(distances):.4f}")
        print(f"Std: {np.std(distances):.4f}")
        print(f"Min: {np.min(distances):.4f}")
        print(f"Max: {np.max(distances):.4f}")
        
def analyze_neuron_progression(model, num_samples=50, k=10, max_neurons=640, step=10, train_loader=None):
    """
    Analyze how increasing neuron count affects blending performance compared to input/output methods
    
    Args:
        model: The trained neural network model
        num_samples: Number of random samples to analyze per neuron count (default=50)
        k: Number of nearest neighbors to use (default=10)
        max_neurons: Maximum number of neurons to analyze (default=640)
        step: Step size for neuron count progression (default=10)
        
    Returns:
        dict: Results containing statistics for each neuron count:
            {
                'neurons': list of neuron counts,
                'act_vs_input': list of mean distances between activation blend and input,
                'act_vs_output': list of mean distances between activation blend and output,
                'input_baseline': mean distance for input-based blending,
                'output_baseline': mean distance for output-based blending
            }
    """
    results = {
        'neurons': [],
        'act_vs_input': [],
        'act_vs_output': [],
        'input_baseline': None,
        'output_baseline': None
    }
    
    # Get base distributions
    inputs, activations, outputs = collect_activations(model, train_loader)
    
    # Calculate baselines first (only need to do this once)
    baseline_stats = analyze_blending_statistics(model, num_samples=num_samples, k=k, n_neurons=64, verbose=False)
    results['input_baseline'] = np.mean(baseline_stats['output_to_input'])
    results['output_baseline'] = np.mean(baseline_stats['input_to_output'])
    
    # Analyze each neuron count
    for n in range(1, max_neurons + 1, step):
        print(f"\nAnalyzing with {n} neurons...")
        stats = analyze_blending_statistics(model, num_samples=num_samples, k=k, n_neurons=n)
        
        results['neurons'].append(n)
        results['act_vs_input'].append(np.mean(stats['act_to_input']))
        results['act_vs_output'].append(np.mean(stats['act_to_output']))
        
    return results

def analyze_knn_approaches(model, space='input', num_samples=100, fixed_k=[5,10,15,20], 
                         adaptive_range=(5,30), step=1, verbose=False, train_loader=None):
    """
    Analyze fixed and adaptive KNN performance for a specific space
    
    Args:
        model: Trained neural network model
        space: Which space to analyze ('input', 'activation', or 'output')
        num_samples: Number of random samples to test
        fixed_k: List of fixed k values to test
        adaptive_range: (min_k, max_k) for adaptive approach
        step: Step size for adaptive k range
        verbose: Whether to print progress
        
    Returns:
        dict: Results containing distances for each approach
    """
    results = {
        'fixed': {k: [] for k in fixed_k},
        'adaptive': []
    }
    
    # Get base distributions
    inputs, activations, outputs = collect_activations(model, train_loader)
    
    # Create mapping for variable names (handle plural/singular mismatch)
    var_mapping = {
        'input': 'new_input',
        'activation': 'new_activations',  # Note plural form
        'output': 'new_output'
    }
    
    for i in range(num_samples):
        if verbose and i % 10 == 0:
            print(f"\rProcessing sample {i+1}/{num_samples}", end='')
            
        # Generate random sample
        rgb_color = np.random.uniform(0, 1, 3)
        new_input, new_activations, new_output = collect_single_color_activations(model, rgb_color)
        
        # Test fixed k values
        for k in fixed_k:
            knn_results = find_k_nearest_neighbors(
                inputs, activations, outputs,
                new_input, new_activations, new_output,
                k=k, verbose=False
            )
            
            # Get blend for specified space
            blend = blend_nearest_neighbors(
                space,
                knn_results[space]['samples'],
                knn_results[space]['distances'],
                locals()[var_mapping[space]]  # Use mapping instead of direct f-string
            )
            
            # Calculate distance
            target = locals()[var_mapping[space]][0]  # Use mapping here too
            distance = calculate_distance(blend, target)
            results['fixed'][k].append(distance)
        
        # Test adaptive approach
        adaptive_results = find_adaptive_neighbors(
            inputs, activations, outputs,
            new_input, new_activations, new_output,
            k_range=adaptive_range,
            step=step,
            verbose=False
        )
        
        results['adaptive'].append(adaptive_results[space]['best_distance'])
    
    if verbose:
        print("\nAnalysis complete!")
    
    return results

def plot_knn_comparison(results, space):
    """
    Create boxplot comparing fixed and adaptive KNN approaches
    """
    plt.figure(figsize=(10, 6))
    
    # Prepare data for boxplot
    data = [results['fixed'][k] for k in sorted(results['fixed'].keys())]
    data.append(results['adaptive'])
    
    # Create labels
    labels = [f'k={k}' for k in sorted(results['fixed'].keys())]
    labels.append('Adaptive')
    
    # Create boxplot
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Customize colors
    colors = ['lightblue'] * len(results['fixed']) + ['lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title(f'KNN Performance Comparison in {space.capitalize()} Space')
    plt.ylabel('L2 Distance')
    plt.grid(True, axis='y')
    
    # Add summary statistics
    stats_text = "Summary Statistics:\n"
    for i, label in enumerate(labels):
        mean = np.mean(data[i])
        std = np.std(data[i])
        stats_text += f"{label}: {mean:.4f} ± {std:.4f}\n"
    
    plt.figtext(1.02, 0.5, stats_text, fontsize=9, va='center')
    plt.tight_layout()
    plt.show()

def find_adaptive_neighbors(base_inputs, base_activations, base_outputs,
                          new_input, new_activations, new_output,
                          k_range=(5, 30), step=1, verbose=True) :
    """
    Find optimal k for nearest neighbors by testing a range of k values and blending
    
    Args:
        base_inputs: Base RGB input samples array
        base_activations: Base activation patterns array
        base_outputs: Base HSV output samples array
        new_input: New RGB input sample
        new_activations: New activation pattern
        new_output: New HSV output sample
        k_range: Tuple of (min_k, max_k) to test (default=(5,30))
        step: Step size for k values (default=1)
        verbose: Whether to print progress (default=True)
        
    Returns:
        dict: Results for best k containing:
            {
                'input': {
                    'best_k': optimal k value,
                    'best_blend': blended RGB values,
                    'best_distance': smallest L2 distance,
                    'samples': nearest neighbor samples,
                    'distances': distances to neighbors
                },
                'activation': {...},  # Same structure as input
                'output': {...}       # Same structure as input
            }
    """
    # Initialize results with worst possible values
    results = {
        'input': {'best_distance': float('inf')},
        'activation': {'best_distance': float('inf')},
        'output': {'best_distance': float('inf')}
    }
    
    # Try different k values
    for k in range(k_range[0], k_range[1] + 1, step):
        if verbose:
            print(f"\rTesting k={k}", end='')
            
        # Get knn results for current k
        knn_results = find_k_nearest_neighbors(
            base_inputs, base_activations, base_outputs,
            new_input, new_activations, new_output,
            k=k, verbose=False
        )
        
        # Test input space
        input_blend = blend_nearest_neighbors(
            'input',
            knn_results['input']['samples'],
            knn_results['input']['distances'],
            new_input
        )
        input_distance = calculate_distance(input_blend, new_input[0])
        
        if input_distance < results['input']['best_distance']:
            results['input'] = {
                'best_k': k,
                'best_blend': input_blend,
                'best_distance': input_distance,
                'samples': knn_results['input']['samples'],
                'distances': knn_results['input']['distances']
            }
            
        # Test activation space
        act_blend = blend_nearest_neighbors(
            'activation',
            knn_results['activation']['samples'],
            knn_results['activation']['distances'],
            new_activations
        )
        act_distance = calculate_distance(act_blend, new_activations[0])
        
        if act_distance < results['activation']['best_distance']:
            results['activation'] = {
                'best_k': k,
                'best_blend': act_blend,
                'best_distance': act_distance,
                'samples': knn_results['activation']['samples'],
                'distances': knn_results['activation']['distances']
            }
            
        # Test output space
        output_blend = blend_nearest_neighbors(
            'output',
            knn_results['output']['samples'],
            knn_results['output']['distances'],
            new_output
        )
        output_distance = calculate_distance(output_blend, new_output[0])
        
        if output_distance < results['output']['best_distance']:
            results['output'] = {
                'best_k': k,
                'best_blend': output_blend,
                'best_distance': output_distance,
                'samples': knn_results['output']['samples'],
                'distances': knn_results['output']['distances']
            }
    
    if verbose:
        print("\n\nBest results found:")
        for space in ['input', 'activation', 'output']:
            print(f"\n{space.capitalize()} space:")
            print(f"Best k: {results[space]['best_k']}")
            print(f"Best distance: {results[space]['best_distance']:.4f}")
    
    return results

def analyze_neuron_based_knn(model, neuron_range=(0,128), num_samples=100, 
                           fixed_k=[5,10,15,20], adaptive_range=(5,30), 
                           step=1, verbose=False, train_loader=None):
    """
    Analyze fixed and adaptive KNN performance using specific activation neurons
    
    Args:
        model: Trained neural network model
        neuron_range: Tuple of (start_neuron, end_neuron) to use
        num_samples: Number of random samples to test
        fixed_k: List of fixed k values to test
        adaptive_range: (min_k, max_k) for adaptive approach
        step: Step size for adaptive k range
        verbose: Whether to print progress
        
    Returns:
        dict: Results containing distances for each approach
    """
    results = {
        'fixed': {k: [] for k in fixed_k},
        'adaptive': []
    }
    
    # Get base distributions
    inputs, activations, outputs = collect_activations(model, train_loader)
    
    # Select neuron range
    start_neuron, end_neuron = neuron_range
    activations = activations[:, start_neuron:end_neuron]
    
    for i in range(num_samples):
        if verbose and i % 10 == 0:
            print(f"\rProcessing sample {i+1}/{num_samples}", end='')
            
        # Generate random sample
        rgb_color = np.random.uniform(0, 1, 3)
        new_input, new_activations, new_output = collect_single_color_activations(model, rgb_color)
        
        # Select same neuron range for new sample
        new_activations = new_activations[:, start_neuron:end_neuron]
        
        # Test fixed k values
        for k in fixed_k:
            # Find neighbors using selected neurons
            neighbors = find_neighbors_by_neurons(
                activations,
                new_activations,
                n_neurons=end_neuron-start_neuron,
                k=k
            )
            
            # Blend values
            blended_rgb, blended_hsv = blend_activation_neighbors_values(
                neighbors,
                inputs,
                outputs
            )
            
            # Calculate distance (using activation space)
            distance = calculate_distance(blended_rgb, new_input[0])
            results['fixed'][k].append(distance)
        
        # Test adaptive approach
        best_distance = float('inf')
        for k in range(adaptive_range[0], adaptive_range[1] + 1, step):
            neighbors = find_neighbors_by_neurons(
                activations,
                new_activations,
                n_neurons=end_neuron-start_neuron,
                k=k
            )
            
            blended_rgb, _ = blend_activation_neighbors_values(
                neighbors,
                inputs,
                outputs
            )
            
            distance = calculate_distance(blended_rgb, new_input[0])
            best_distance = min(best_distance, distance)
            
        results['adaptive'].append(best_distance)
    
    if verbose:
        print("\nAnalysis complete!")
    
    return results