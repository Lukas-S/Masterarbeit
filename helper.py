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

def showcase_iris_neighbors(X_train, y_train, knn_indices, knn_distances=None, X_test=None, y_test=None, query_index=None, feature_names=None):
    """Text-based showcase for an Iris query (from test set) and its K nearest neighbors (from train set).

    Args:
        X_train, y_train: train arrays/tensors (N_train, 4) and labels
        knn_indices: iterable of neighbor indices (indices into the train set)
        knn_distances: optional iterable of distances corresponding to knn_indices
        X_test, y_test: optional test arrays/tensors (N_test, 4) and labels used to display the query
        query_index: index into X_test (if X_test provided) or into X_train if no X_test given
        feature_names: optional list of 4 feature names
    """
    if feature_names is None:
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Prefer using test data for the query when available
    if X_test is not None and query_index is not None:
        Xq = X_test
        yq = y_test
        query_source = 'test'
    else:
        Xq = X_train
        yq = y_train
        query_source = 'train'

    # Convert to CPU numpy for pretty printing
    X_train_np = X_train.detach().cpu().numpy() if hasattr(X_train, 'detach') else np.asarray(X_train)
    y_train_np = y_train.detach().cpu().numpy() if hasattr(y_train, 'detach') else np.asarray(y_train)

    Xq_np = Xq.detach().cpu().numpy() if hasattr(Xq, 'detach') else np.asarray(Xq)
    yq_np = yq.detach().cpu().numpy() if hasattr(yq, 'detach') else np.asarray(yq)

    # Prepare printable arrays
    class_names = ['setosa', 'versicolor', 'virginica']

    # Ensure knn indices and distances are lists
    knn_list = knn_indices.tolist() if hasattr(knn_indices, 'tolist') else list(knn_indices)
    dist_list = None
    if knn_distances is not None:
        dist_list = knn_distances.tolist() if hasattr(knn_distances, 'tolist') else list(knn_distances)

    # Header layout: Role, Index, Class, Distance, features
    col_role = "Role"
    col_idx = "Index"
    col_class = "Class"
    cols_feat = feature_names

    # Column widths
    w_role = 8
    w_idx = 6
    w_class = 12
    w_dist = 10
    w_feat = 12

    # Print header
    header_parts = [f"{col_role:<{w_role}}", f"{col_idx:<{w_idx}}", f"{col_class:<{w_class}}", f"{'Distance':<{w_dist}}"]
    header_parts += [f"{fn:<{w_feat}}" for fn in cols_feat]
    header = " ".join(header_parts)
    sep = "-" * len(header)

    print("Iris K-NN Showcase (Query from test set, neighbors from train set)")
    print(sep)
    print(header)
    print(sep)

    # Print query row first (if provided)
    if query_index is not None:
        q = Xq_np[int(query_index)]
        q_label = int(yq_np[int(query_index)])
        # Query has no distance
        row_parts = [f"{'Query':<{w_role}}", f"{int(query_index):<{w_idx}}", f"{class_names[q_label]:<{w_class}}", f"{'':<{w_dist}}"]
        row_parts += [f"{q[i]:<{w_feat}.2f}" for i in range(len(cols_feat))]
        print(" ".join(row_parts))
        print(sep)

    # Print neighbors aligned under same columns
    for i, idx in enumerate(knn_list):
        idx_i = int(idx)
        label_i = int(y_train_np[idx_i])
        dist_i = float(dist_list[i]) if (dist_list is not None and i < len(dist_list)) else None
        dist_str = f"{dist_i:.4f}" if dist_i is not None else ""
        feats = X_train_np[idx_i]
        row_parts = [f"{'Neighbor':<{w_role}}", f"{idx_i:<{w_idx}}", f"{class_names[label_i]:<{w_class}}", f"{dist_str:<{w_dist}}"]
        row_parts += [f"{feats[j]:<{w_feat}.2f}" for j in range(len(cols_feat))]
        print(" ".join(row_parts))
    print(sep)


def showcase_mnist_neighbors(train_dataloader, knn_indices, knn_distances=None, test_dataloader=None, query_index=None, n_cols=8, cmap='gray'):
    """Show MNIST query and neighbors as images in a single visualization.

    Args:
        dataloader: DataLoader containing MNIST data
        knn_indices: iterable of neighbor indices
        knn_distances: optional iterable of distances matching knn_indices
        query_index: optional int index of the query sample (will be shown first)
        n_cols: columns in grid
    """
    # Build display: query from test_dataloader (if provided) then neighbors from train_dataloader
    display_imgs = []
    display_labels = []
    titles = []

    if test_dataloader is not None and query_index is not None:
        q_imgs, q_labels = get_samples_by_indices(test_dataloader, query_index)
        # q_imgs may be single sample
        display_imgs.append(q_imgs[0])
        display_labels.append(q_labels[0])
        titles.append(f"Query idx={query_index}")

    # Fetch neighbor images from train set
    neigh_imgs, neigh_labels = get_samples_by_indices(train_dataloader, knn_indices)
    for i in range(len(knn_indices)):
        display_imgs.append(neigh_imgs[i])
        display_labels.append(neigh_labels[i])
        if knn_distances is not None and i < len(knn_distances):
            titles.append(f"N{i+1} idx={int(knn_indices[i])}\n dist={float(knn_distances[i]):.4f}")
        else:
            titles.append(f"N{i+1} idx={int(knn_indices[i])}")

    imgs = torch.stack(display_imgs) if isinstance(display_imgs[0], torch.Tensor) else np.stack(display_imgs)
    labels = torch.tensor(display_labels) if isinstance(display_labels[0], (int, np.integer)) else np.array(display_labels)

    n = len(display_imgs)
    cols = min(n_cols, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            img = imgs[i]
            ax.imshow(img.squeeze(), cmap=cmap)
            title = titles[i]
            # append true label
            title += f"\nlabel={int(labels[i])}" if labels is not None else ''
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def showcase_rgb_neighbors(rgb_train, knn_indices=None, knn_distances=None, rgb_test=None, query_index=None, hsv_train=None, hsv_test=None, square_size=40, n_cols=8):
    """Visualize RGB query (from test set) and neighbor colors (from train set) as small colored squares.

    Backwards compatible: if rgb_test is None, query_index will be taken from rgb_train.
    """
    # Build display: query from test set if provided, neighbors from train
    display_colors = []
    titles = []

    if rgb_test is not None and query_index is not None:
        rgb_test_np = rgb_test.detach().cpu().numpy() if hasattr(rgb_test, 'detach') else np.asarray(rgb_test)
        display_colors.append(rgb_test_np[int(query_index)])
        titles.append(f"Query idx={query_index}")
    elif query_index is not None:
        # fall back to train set for query
        rgb_train_np_all = rgb_train.detach().cpu().numpy() if hasattr(rgb_train, 'detach') else np.asarray(rgb_train)
        display_colors.append(rgb_train_np_all[int(query_index)])
        titles.append(f"Query idx={query_index}")

    # neighbors from train
    rgb_train_np = rgb_train.detach().cpu().numpy() if hasattr(rgb_train, 'detach') else np.asarray(rgb_train)
    hsv_train_np = hsv_train.detach().cpu().numpy() if (hsv_train is not None and hasattr(hsv_train, 'detach')) else (np.asarray(hsv_train) if hsv_train is not None else None)

    if knn_indices is not None:
        for i, idx in enumerate(knn_indices):
            display_colors.append(rgb_train_np[int(idx)])
            if knn_distances is not None and i < len(knn_distances):
                titles.append(f"N{i+1} idx={int(idx)}\n dist={float(knn_distances[i]):.4f}")
            else:
                titles.append(f"N{i+1} idx={int(idx)}")

    if len(display_colors) == 0:
        print("No indices to display.")
        return

    n = len(display_colors)
    cols = min(n_cols, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2., rows*2))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i < n:
            color = np.clip(display_colors[i].reshape(1,1,3), 0, 1)
            square = np.ones((square_size, square_size, 3), dtype=float) * color
            ax.imshow(square)
            ax.set_title(titles[i], fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
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

# def train_model(model, dataloader, criterion, optimizer, num_epochs=100, device=None, verbose=False, seed=42, continue_training=False):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     if not continue_training:
#         set_seed(seed)
#         init_weights(model, seed)
    
#     model.to(device)
#     model.train()
#     losses = []
    
#     if verbose:
#         print(f"Training on device: {device}")
    
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
        
#         avg_loss = epoch_loss / len(dataloader)
#         losses.append(avg_loss)
        
#         if verbose and (epoch + 1) % 20 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
#     return losses, model

def train_model(model, dataloader, criterion, optimizer, num_epochs=100, device=None, verbose=False, seed=42, continue_training=False, test_loader=None, return_val=False):
    """
    Train the model.

    Backwards compatible:
      - By default returns (losses, model) as before.

    New functionality:
      - If test_loader is provided, computes validation loss (and accuracy for classification)
        at the end of each epoch and stores them in val_losses / val_accuracies.
      - If return_val=True and test_loader is provided, returns (losses, model, val_losses, val_accuracies).

    Args:
        test_loader: optional DataLoader used to evaluate after each epoch.
        return_val: if True and test_loader provided, include val lists in return.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not continue_training:
        set_seed(seed)
        init_weights(model, seed)

    model.to(device)
    model.train()
    losses = []

    val_losses = [] if test_loader is not None else None
    val_accuracies = [] if test_loader is not None else None

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

        # Evaluate on test_loader after each epoch if provided
        if test_loader is not None:
            val_loss, val_acc = evaluate_model(model, test_loader, criterion, device=device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

        if verbose and (epoch + 1) % 20 == 0:
            if test_loader is not None:
                acc_str = f", Val Loss: {val_losses[-1]:.6f}"
                if val_accuracies[-1] is not None:
                    acc_str += f", Val Acc: {val_accuracies[-1]:.4f}"
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}{acc_str}')
            else:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

    # Return results while keeping backward compatibility
    if return_val and test_loader is not None:
        return losses, model, val_losses, val_accuracies
    return losses, model

def evaluate_model(model, dataloader, criterion, device=None):
    """
    Evaluate model on dataloader. Returns (loss, accuracy_or_None).

    Accuracy is computed only when criterion is CrossEntropyLoss (classification),
    otherwise returns None for accuracy (e.g., MSE regression).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Compute accuracy for classification (CrossEntropy)
            if isinstance(criterion, nn.CrossEntropyLoss):
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (correct / total_samples) if (total_samples > 0 and isinstance(criterion, nn.CrossEntropyLoss)) else None

    model.train()
    return avg_loss, accuracy

def print_training_history(losses, val_losses=None, val_accuracies=None, show_last_n=10):
    """
    Print a concise summary of training (and optional validation) losses/accuracies.

    Args:
      losses: list of training losses per epoch
      val_losses: optional list of validation losses per epoch
      val_accuracies: optional list of validation accuracies per epoch (or None entries)
      show_last_n: number of last epochs to print
    """
    n = len(losses)
    start = max(0, n - show_last_n)
    print("Training Losses (last {} epochs):".format(min(show_last_n, n)))
    for i, l in enumerate(losses[start:], start=start+1):
        line = f"Epoch {i:3d}: Train Loss = {l:.6f}"
        if val_losses is not None:
            vl = val_losses[i-1] if i-1 < len(val_losses) else None
            if vl is not None:
                line += f", Val Loss = {vl:.6f}"
        if val_accuracies is not None:
            va = val_accuracies[i-1] if i-1 < len(val_accuracies) else None
            if va is not None:
                line += f", Val Acc = {va:.4f}"
        print(line)

def plot_training_history(losses, val_losses=None, val_accuracies=None, show_last_n=None, title="Training / Validation Loss", figsize=(12,8), markersize=3):
    """
    Plot training loss and optional validation loss (line plots).

    Args:
      losses: list of training losses per epoch
      val_losses: optional list of validation losses per epoch
      val_accuracies: optional list of validation accuracies (not plotted here)
      show_last_n: if set, only plot the last N epochs
    """

    if show_last_n is not None:
        losses_plot = losses[-show_last_n:]
        if val_losses is not None:
            val_losses_plot = val_losses[-show_last_n:]
        else:
            val_losses_plot = None
        start_epoch = max(1, len(losses) - len(losses_plot) + 1)
        xs = list(range(start_epoch, start_epoch + len(losses_plot)))
    else:
        losses_plot = losses
        val_losses_plot = val_losses
        xs = list(range(1, len(losses_plot) + 1))

    plt.figure(figsize=figsize)
    plt.plot(xs, losses_plot, label="Train Loss", marker="o", markersize=markersize)
    if val_losses_plot is not None:
        plt.plot(xs, val_losses_plot, label="Val Loss", marker="o", markersize=markersize)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

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
       
    plt.figure(figsize=(8, 4))
    
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

# def visualize_results(mse_results, list_of_indices, selected_index=None, average=False, base_mse=None, normalize=False, title="Loss Visualization", ylabel="Loss"):
#     plt.figure(figsize=(12, 8))

#     if average:
#         # Prepare data for averages
#         averages = {}
#         for key, results in mse_results.items():
#             averages[key] = [np.mean(results[k]) for k in results.keys()]

#         # Normalize if required
#         if normalize:
#             max_value = max([max(values) for values in averages.values()])
#             averages = {key: [val / max_value for val in values] for key, values in averages.items()}

#         # Plot averages side by side
#         for key, values in averages.items():
#             plt.plot(list(mse_results[key].keys()), values, label=f"{key} (Average)", marker="o")

#     elif selected_index is not None:
#         # Plot for a specific index
#         for key, results in mse_results.items():
#             plt.plot(list(results.keys()), [results[k][selected_index] for k in results.keys()], label=f"{key} (Index {selected_index})", marker="o")

#     else:
#         # Plot all indices
#         for key, results in mse_results.items():
#             for idx, values in enumerate(zip(*results.values())):
#                 plt.plot(list(results.keys()), values, label=f"{key} (Index {idx})", marker="o")

#     if base_mse is not None:
#         plt.axhline(y=base_mse, color="r", linestyle="--", label="Base MSE")

#     plt.xlabel("K Neighbors")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def visualize_results(mse_results, list_of_indices, selected_index=None, average=False, base_mse=None, normalize=False, title="Loss Visualization", ylabel="Loss", max_points=None, max_knn=None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 4))

    def keys_for(results):
        # Ensure ks are sorted numeric K values
        ks = sorted(list(results.keys()))
        # If a maximum K value is supplied, only keep ks <= max_knn
        if max_knn is not None:
            ks = [k for k in ks if k <= max_knn]
        # If a maximum count is supplied, trim to that many points
        if max_points is not None:
            ks = ks[:max_points]
        return ks

    if average:
        averages = {}
        for key, results in mse_results.items():
            ks = keys_for(results)
            if len(ks) == 0:
                print(f"No k values selected for {key} (check max_knn/max_points).")
                averages[key] = []
                continue
            averages[key] = [np.mean(results[k]) for k in ks]

        if normalize and any(len(v) > 0 for v in averages.values()):
            max_value = max([max(values) for values in averages.values() if len(values) > 0])
            if max_value != 0:
                averages = {key: [val / max_value for val in values] for key, values in averages.items()}

        for key, values in averages.items():
            ks = keys_for(mse_results[key])
            if len(ks) == 0:
                continue
            plt.plot(ks, values, label=f"{key} (Average)", marker="o")

    elif selected_index is not None:
        for key, results in mse_results.items():
            ks = keys_for(results)
            if len(ks) == 0:
                continue
            try:
                plt.plot(ks, [results[k][selected_index] for k in ks], label=f"{key} (Index {selected_index})", marker="o")
            except IndexError:
                print(f"Selected index {selected_index} out of range for some k in {key}.")

    else:
        for key, results in mse_results.items():
            ks = keys_for(results)
            if len(ks) == 0:
                continue
            values_matrix = [results[k] for k in ks]
            for idx, values in enumerate(zip(*values_matrix)):
                plt.plot(ks, list(values), label=f"{key} (Index {idx})", marker="o")

    if base_mse is not None:
        plt.axhline(y=base_mse, color="r", linestyle="--", label="Base MSE")

    plt.xlabel("K Neighbors")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def train_multiple_inits_and_plot(network_fn,
                                  train_loader,
                                  criterion,
                                  optimizer_fn,
                                  test_loader=None,
                                  seeds=None,
                                  num_epochs=150,
                                  device=None,
                                  plot_val=True,
                                  figsize=(12,4),
                                  markersize=3,
                                  alpha=0.25,
                                  colors=None,
                                  title_prefix="Training Loss",
                                  color_by_seed=True,
                                  show_seed_legend=True,
                                  seed_colors=None,
                                  loss_view="train"):
    """
    Train the same network architecture multiple times with different initialization seeds,
    capture training (and optional validation) losses and plot runs (thin dotted) + average (bold).

    New args:
      color_by_seed: assign a distinct color per seed for individual-run lines (default True)
      show_seed_legend: add each seed's line to the legend (default True)
      seed_colors: optional dict {seed: color} or list of colors (len == len(seeds))
      loss_view: 'train' or 'val' - which loss to visualize (only one shown). Default 'train'.

    Returns:
      dict with train_runs, val_runs, avg_train, avg_val, seeds
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if loss_view not in ("train", "val"):
        raise ValueError("loss_view must be 'train' or 'val'")

    if seeds is None:
        seeds = [42 + i for i in range(5)]

    if colors is None:
        colors = {"train": "C0", "val": "C1"}

    # Build per-seed color mapping if requested
    seed_color_map = {}
    if color_by_seed:
        if isinstance(seed_colors, dict):
            for s in seeds:
                seed_color_map[s] = seed_colors.get(s, None)
        elif isinstance(seed_colors, (list, tuple)) and len(seed_colors) >= len(seeds):
            for s, c in zip(seeds, seed_colors):
                seed_color_map[s] = c
        else:
            cmap = plt.get_cmap('tab10' if len(seeds) <= 10 else 'tab20')
            for i, s in enumerate(seeds):
                seed_color_map[s] = cmap(i % cmap.N)

    train_runs = []
    val_runs = [] if test_loader is not None else None

    for run_i, s in enumerate(seeds):
        model = network_fn()
        optimizer = optimizer_fn(model)

        if test_loader is not None:
            losses, trained_model, val_losses, val_accuracies = train_model(
                model, train_loader, criterion, optimizer,
                num_epochs=num_epochs, seed=s, verbose=False,
                continue_training=False, test_loader=test_loader, return_val=True
            )
            val_runs.append(val_losses)
        else:
            losses, trained_model = train_model(
                model, train_loader, criterion, optimizer,
                num_epochs=num_epochs, seed=s, verbose=False,
                continue_training=False, test_loader=None, return_val=False
            )

        train_runs.append(losses)

    # Align lengths
    min_epochs = min(len(r) for r in train_runs)
    train_runs = [r[:min_epochs] for r in train_runs]
    if val_runs is not None:
        min_val_epochs = min(len(r) for r in val_runs)
        val_runs = [r[:min_val_epochs] for r in val_runs]
    else:
        min_val_epochs = None

    # Compute averages
    avg_train = list(np.mean(np.vstack(train_runs), axis=0))
    avg_val = list(np.mean(np.vstack(val_runs), axis=0)) if val_runs is not None else None

    # Ensure requested loss_view is available
    if loss_view == "val" and val_runs is None:
        raise ValueError("loss_view='val' requested but no test_loader was provided (no val_runs available).")

    # Plotting (only one of train or val, never both)
    if loss_view == "train":
        xs = np.arange(1, min_epochs + 1)
        plt.figure(figsize=figsize)
        for run_i, (r, s) in enumerate(zip(train_runs, seeds)):
            line_color = seed_color_map.get(s) if color_by_seed else colors.get("train", "C0")
            label = f"seed {s}" if show_seed_legend and color_by_seed else None
            plt.plot(xs, r, linestyle=':', color=line_color, alpha=alpha, marker='.', markersize=markersize, label=label)
        plt.plot(xs, avg_train, linestyle='-', color=colors.get("train", "C0"), linewidth=2.5, label='Avg Train Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title(f"{title_prefix} - Train (n_seeds={len(seeds)}, epochs={min_epochs})")
    else:  # loss_view == "val"
        xs = np.arange(1, min_val_epochs + 1)
        plt.figure(figsize=figsize)
        for run_i, (r, s) in enumerate(zip(val_runs, seeds)):
            line_color = seed_color_map.get(s) if color_by_seed else colors.get("val", "C1")
            label = f"seed {s} (val)" if (show_seed_legend and color_by_seed) else None
            plt.plot(xs, r, linestyle=':', color=line_color, alpha=alpha, marker='.', markersize=markersize, label=label)
        plt.plot(xs, avg_val, linestyle='-', color=colors.get("val", "C1"), linewidth=2.5, label='Avg Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.title(f"{title_prefix} - Val (n_seeds={len(seeds)}, epochs={min_val_epochs})")

    plt.grid(True)
    # Manage legend: avoid duplicate entries if many seeds shown
    if show_seed_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        seen = set()
        new_h, new_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                new_h.append(h); new_l.append(l); seen.add(l)
        plt.legend(new_h, new_l, fontsize='small')
    else:
        plt.legend(fontsize='small')

    plt.tight_layout()
    plt.show()

    return {
        "train_runs": train_runs,
        "val_runs": val_runs,
        "avg_train": avg_train,
        "avg_val": avg_val,
        "seeds": seeds
    }
    
def plot_knn_distance_stats(knn_distances,
                            subset='closest',
                            k=None,
                            max_k=None,
                            figsize=(8,4),
                            show_min=True,
                            show_max=True,
                            show_mean=True,
                            fill_between=True,
                            title=None,
                            xlabel="Neighbor index (1 = nearest)",
                            ylabel="Distance",
                            marker='o',
                            color_mean='C0',
                            color_min='C1',
                            color_max='C2'):
    """
    Plot statistics (mean / min / max) of KNN distances for a chosen subset.

    Args:
      knn_distances: dict with keys like 'closest_distances','last_distances','random_distances'
                     where each maps k -> list_of_samples_of_length_k
      subset: one of 'closest','last','random' (selects the dict key `<subset>_distances`)
      k: integer. If provided, use that exact k (must exist in data). If None, selected key is chosen by max_k or largest available.
      max_k: if k is None, pick the largest available k <= max_k; if max_k is None pick largest available k.
      figsize, show_min/max/mean, fill_between, title, labels/colors: plotting options.

    Returns:
      stats: dict with keys 'k', 'mean', 'min', 'max', each a list of length k
    """
    import numpy as np
    import matplotlib.pyplot as plt

    key = f"{subset}_distances"
    if key not in knn_distances:
        raise KeyError(f"knn_distances does not contain key '{key}'")

    available_ks = sorted([int(x) for x in knn_distances[key].keys()])
    if len(available_ks) == 0:
        raise ValueError(f"No k entries found under '{key}'")

    # select k to use
    if k is not None:
        if k not in available_ks:
            raise ValueError(f"Requested k={k} not available for '{key}'. Available: {available_ks}")
        k_sel = k
    else:
        if max_k is None:
            k_sel = available_ks[-1]
        else:
            # pick largest available k <= max_k
            cand = [kk for kk in available_ks if kk <= int(max_k)]
            if not cand:
                raise ValueError(f"No available k <= max_k ({max_k}). Available: {available_ks}")
            k_sel = cand[-1]

    # Gather sample distance lists for selected k and normalize/truncate to exact length k_sel
    raw_lists = knn_distances[key].get(k_sel, [])
    if len(raw_lists) == 0:
        raise ValueError(f"No distance lists found for k={k_sel} in '{key}'")

    # keep only samples with at least k_sel entries, trim extras
    filtered = []
    for lst in raw_lists:
        try:
            arr = np.asarray(lst, dtype=float)
        except Exception:
            continue
        if arr.size >= k_sel:
            filtered.append(arr[:k_sel])
    if len(filtered) == 0:
        raise ValueError(f"No valid distance arrays of length >= {k_sel} for '{key}'")

    data = np.vstack(filtered)  # shape (n_samples, k_sel)

    mean_vec = np.mean(data, axis=0)
    min_vec = np.min(data, axis=0)
    max_vec = np.max(data, axis=0)

    xs = np.arange(1, k_sel + 1)

    plt.figure(figsize=figsize)
    if show_mean:
        plt.plot(xs, mean_vec, label='mean', color=color_mean, marker=marker, linewidth=2, markersize=2)
    if show_min:
        plt.plot(xs, min_vec, label='min', color=color_min, linestyle='--', marker=None)
    if show_max:
        plt.plot(xs, max_vec, label='max', color=color_max, linestyle='--', marker=None)
    if fill_between and show_min and show_max:
        plt.fill_between(xs, min_vec, max_vec, color=color_mean, alpha=0.08)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title or f"KNN distances ({subset}) — k={k_sel} — n_samples={data.shape[0]}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"k": k_sel, "mean": mean_vec.tolist(), "min": min_vec.tolist(), "max": max_vec.tolist()}