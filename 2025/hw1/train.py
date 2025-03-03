"""
Student Name: Zitong Yang

CS375 / Psych 279 Homework 1

You have to finish the following tasks:
- Implement the AlexNet model definition
- Implement the forward pass of the AlexNet model
- Implement the missing parts of the training loop
- Implement the evaluation function to compute top-1 accuracy and loss
- Implement the function to plot the Conv1 kernels
- Implement the function to plot the kernel responses for sine gratings

"""


import os
import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from typing import Tuple, List

sns.set_theme(style="whitegrid")


#######################################################################################
### MODEL CLASS
#######################################################################################

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            # FC6
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            # FC7
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            # FC8
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through convolutional layers
        x = self.features(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Pass through fully connected layers
        x = self.classifier(x)
        
        return x


#######################################################################################
### EVALUATION FUNCTIONS
#######################################################################################

def evaluate_accuracy_and_loss(
        model: torch.nn.Module, 
        data_loader: torch.utils.data.DataLoader, 
        device: torch.device
    ) -> Tuple[float, float]:
    """
    Computes the top-1 accuracy of the model on the entire data_loader.

    Parameters:
        - model: torch.nn.Module, the model to evaluate
        - data_loader: torch.utils.data.DataLoader, the data loader to evaluate
        - device: torch.device, the device to run the evaluation on

    Returns:
        - accuracy: float, the top-1 accuracy of the model on the data_loader
        - loss: float, the average loss of the model on the data_loader
    """
    model.eval()  # Set model to evaluation mode
    criterion = nn.CrossEntropyLoss()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient calculations
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update metrics
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    # Calculate final metrics
    accuracy = (total_correct / total_samples) * 100.0  # Convert to percentage
    avg_loss = running_loss / total_samples

    return accuracy, avg_loss

def plot_conv1_kernels(model: torch.nn.Module, epoch: int):
    """
    Creates a figure with a grid of the Conv1 kernels at the given epoch.

    Parameters:
        - model: torch.nn.Module, the AlexNet model
        - epoch: int, the current training epoch
    """
    # Get the weights from the first conv layer
    conv1_weights = model.features[0].weight.data.cpu()

    # Normalize the weights for better visualization
    min_val = conv1_weights.min()
    max_val = conv1_weights.max()
    conv1_weights = (conv1_weights - min_val) / (max_val - min_val)

    # Create a grid of the kernels
    kernel_grid = torchvision.utils.make_grid(conv1_weights, nrow=8, padding=2, normalize=False)

    # Convert to numpy for matplotlib
    kernel_grid = kernel_grid.numpy().transpose((1, 2, 0))

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(kernel_grid)
    ax.axis('off')
    ax.set_title(f'Conv1 Kernels (Epoch {epoch})')

    # Save the figure
    fig.savefig(os.path.join('out', f'conv1_kernels_epoch_{epoch}.png'))
    plt.close(fig)

def compute_circular_variance(angles_deg, responses):
    """
    Computes circular variance for a set of angles (in degrees) and responses.
    
    The angles of the stimuli are in the range [0,180] degrees
    meaning 0 ~ 180 are effectively the same orientation. We handle this by 
    doubling the angles before computing the circular variance.
      
    Circular Variance (CV) in [0,1]:
      CV = 1 - (R / sum(r)), 
      R = length_of_resultant_vector = sqrt( (Σ r_i cos θ_i)^2 + (Σ r_i sin θ_i)^2 )

    The returned CV is clamped in [0,1].
    """
    # 1) Convert angles to a NumPy array
    angles_deg = np.array(angles_deg, dtype=float)

    # 2) Double the angles since we are measuring direcion not orientation
    #    selectivity. This effectively treats 0 and 180 as the same.
    angles_deg = 2.0 * angles_deg

    # 3) Convert degrees to radians
    angles_rad = np.radians(angles_deg)

    # 4) ReLU the responses (clip negative to zero)
    r = np.array(responses, dtype=float)
    r = np.maximum(r, 0.0)  # ReLU

    sum_r = r.sum()
    if sum_r == 0:
        # If the total response is 0, define CV = 0 (or handle as you see fit)
        return 0.0

    # 5) Compute vector sums
    sum_rcos = np.sum(r * np.cos(angles_rad))
    sum_rsin = np.sum(r * np.sin(angles_rad))
    resultant = math.sqrt(sum_rcos**2 + sum_rsin**2)

    # 6) Normalize by the sum of responses
    R = resultant / sum_r

    # 7) Compute circular variance: 1 - R
    cv = 1.0 - R

    # 8) Clamp to [0,1] to guard against floating-point imprecision
    cv = max(0.0, min(cv, 1.0))

    return cv

def plot_sine_grating_responses_for_filters(
    model: torch.nn.Module,
    epoch: int,
    device: torch.device
) -> List[float]:
    """
    1) Loads images from sine_grating_images/.
    2) For each image, parse degree (deg) and spatial frequency (sf) from the filename.
    3) Forward pass each image through only the first conv layer of AlexNet.
    4) Grabs the kernel response at the central spatial location (27, 27).
    5) For each kernel, produce a figure with:
       - Subplot 1: response vs. orientation (deg)
       - Subplot 2: response vs. spatial frequency (sf)
       - Subplot 3: the kernel weights visualization
       - Include circular variance in the figure title
    6) Save all kernel figures to out/kernel_responses_{epoch}/.
    7) Return a list of circular variances for each kernel.

    Parameters:
        - model: torch.nn.Module, the AlexNet model
        - epoch: int, the current training epoch
        - device: torch.device, the device to run the evaluation on

    Returns:
        - circular_variances: List[float], the circular variance for each kernel
    """

    # Where to save the plots for this epoch
    out_dir = f"out/imagenet_kernel_responses_{epoch:02d}"
    os.makedirs(out_dir, exist_ok=True)

    # We'll use a simple transform to match the input size expected by AlexNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    # We'll measure the responses of the first conv layer
    conv1 = model.features[0]

    # Gather all sine grating image files
    image_dir = "sine_grating_images"
    if not os.path.isdir(image_dir):
        print(f"Warning: Directory '{image_dir}' does not exist.")
        return
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    # Prepare a list for each kernel to store (deg, sf, response)
    responses_per_kernel = [[] for _ in range(conv1.out_channels)]

    model.eval()
    with torch.no_grad():
        for fname in image_files:
            # Example: fullfield_0.0deg_5.5sf_1.3phase_bw.jpg
            match = re.match(r"fullfield_(?P<deg>[\d\.]+)deg_(?P<sf>[\d\.]+)sf_.*\.jpg", fname)
            if match is None:
                # Skip files that don't match the naming convention
                continue
            deg = float(match.group("deg"))
            sf  = float(match.group("sf"))

            # Load and transform the image
            path = os.path.join(image_dir, fname)
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

            # Forward pass only through the first conv layer
            out = conv1(img_tensor)  # shape: [1, out_channels, H, W]

            # Grab the response at the central spatial location (27, 27)
            out_mean = out[0, :, 27, 27]

            # Store the response in responses_per_kernel
            for k in range(conv1.out_channels):
                responses_per_kernel[k].append((deg, sf, out_mean[k].item()))

    circular_variances = []

    # Now plot, for each kernel, the response vs deg, the response vs sf, and the kernel itself
    for k in range(conv1.out_channels):
        # If this kernel received no data (unlikely, but just a safety check), skip
        if len(responses_per_kernel[k]) == 0:
            continue

        # Sort responses by degree and spatial frequency
        responses = responses_per_kernel[k]
        responses.sort()  # Sort by degree (first element of tuple)
        
        # Extract separate lists for plotting
        degrees = [r[0] for r in responses]
        sfs = [r[1] for r in responses]
        resp_values = [r[2] for r in responses]

        # Compute circular variance
        cv = compute_circular_variance(degrees, resp_values)
        circular_variances.append(cv)

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Response vs Orientation
        unique_degrees = sorted(list(set(degrees)))
        degree_responses = {deg: [] for deg in unique_degrees}
        for d, _, r in responses:
            degree_responses[d].append(r)
        mean_responses = [np.mean(degree_responses[d]) for d in unique_degrees]
        ax1.plot(unique_degrees, mean_responses, 'b-')
        ax1.set_xlabel('Orientation (degrees)')
        ax1.set_ylabel('Response')
        ax1.set_title('Response vs Orientation')

        # Plot 2: Response vs Spatial Frequency
        unique_sfs = sorted(list(set(sfs)))
        sf_responses = {sf: [] for sf in unique_sfs}
        for _, sf, r in responses:
            sf_responses[sf].append(r)
        mean_sf_responses = [np.mean(sf_responses[sf]) for sf in unique_sfs]
        ax2.plot(unique_sfs, mean_sf_responses, 'r-')
        ax2.set_xlabel('Spatial Frequency')
        ax2.set_ylabel('Response')
        ax2.set_title('Response vs Spatial Frequency')

        # Plot 3: Kernel Visualization
        kernel_weights = conv1.weight[k].detach().cpu()
        kernel_grid = torchvision.utils.make_grid(kernel_weights.unsqueeze(1), normalize=True)
        ax3.imshow(kernel_grid.permute(1, 2, 0))
        ax3.axis('off')
        ax3.set_title('Kernel Weights')

        # Set overall title with circular variance
        fig.suptitle(f'Kernel {k} (CV: {cv:.3f})')
        plt.tight_layout()
        
        # Save figure
        fig.savefig(os.path.join(out_dir, f'kernel_{k}_responses.png'))
        plt.close(fig)

    # Plot histogram of circular variances for all kernels at the given epoch
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(circular_variances, bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel("Circular Variance")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Circular Variance Distribution (Epoch {epoch})")
    fig.savefig(os.path.join(out_dir, "circular_variance_histogram.png"))
    plt.close(fig)

    return circular_variances



#######################################################################################
### TRAINING LOOP
#######################################################################################

def get_lr_for_epoch(epoch: int) -> float:
    """
    Returns the learning rate for the given epoch based on our schedule:
    - 1 <= epoch <= 15: 0.01
    - 16 <= epoch <= 25: 0.001
    - 26 <= epoch <= 30: 0.0001

    Parameters:
        - epoch: int, the current epoch number

    Returns:
        - lr: float, the learning rate for this epoch
    """
    if epoch <= 15:
        return 0.01
    elif epoch <= 25:
        return 0.001
    else:
        return 0.0001

def main():
    # ---------------------------
    # 1. Configure Parameters
    # ---------------------------
    os.makedirs("out", exist_ok=True)

    # Hyperparameters
    total_epochs = 30
    batch_size = 256
    num_workers = 128
    momentum = 0.9
    weight_decay = 0.0
    seed = 1110

    # Select device
    if torch.backends.mps.is_available():  # Check for Apple Silicon (MPS)
        device = torch.device("mps")  # Metal Performance Shaders (MPS) on macOS
    elif torch.cuda.is_available():  # Check for CUDA availability
        device = torch.device("cuda")
    else:  # Fallback to CPU
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    # Data directory (ImageNet structure assumed)
    data_dir = 'imagenet-mini' ### TODO: Set the path to the ImageNet dataset
    
    # ---------------------------
    # 2. Data Preparation
    # ---------------------------
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = torchvision.datasets.ImageNet(
        root=data_dir, 
        split='train', 
        transform=train_transforms
    )
    test_dataset = torchvision.datasets.ImageNet(
        root=data_dir, 
        split='val', 
        transform=val_transforms
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # ---------------------------
    # 3. Model Setup
    # ---------------------------
    model = AlexNet(num_classes=1000).to(device)

    optimizer = optim.SGD(
        model.parameters(), 
        lr=get_lr_for_epoch(1),  # Start with initial LR for epoch=1
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Lists to store losses and accuracies
    train_losses = []
    test_losses = []
    test_acc_history = []
    circular_variances = []

    # Check if there's an existing checkpoint we can load
    checkpoint_path = "out/imagenet.pt"
    start_epoch = 1
    if os.path.isfile(checkpoint_path):
        print(f"Found checkpoint {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load state dicts
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load epoch info and histories
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        test_acc_history = checkpoint['test_acc_history']
        circular_variances = checkpoint['circular_variances']

        print(f"Resuming from epoch {start_epoch}...")
    else:
        print("No checkpoint found. Starting from scratch.")

        # Compute top-1 accuracy on the entire validation set
        test_accuracy, test_loss = evaluate_accuracy_and_loss(model, test_loader, device)
        test_acc_history.append(test_accuracy)

        # # Compute circular variance of initial kernel responses
        # curr_circular_variances = plot_sine_grating_responses_for_filters(model, 0, device)
        # circular_variances.append(curr_circular_variances)

        # # Compute and save the initial kernel filters
        # plot_conv1_kernels(model, 0)


    # ---------------------------
    # 4. Training Loop
    # ---------------------------
    for epoch in range(start_epoch, total_epochs + 1):
        # Set the correct LR for this epoch
        lr = get_lr_for_epoch(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        running_train_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} (Train)"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate training loss
            running_train_loss += loss.item() * images.size(0)
        
        # Calculate average train loss for the epoch
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Compute top-1 accuracy on the entire validation set
        test_accuracy, test_loss = evaluate_accuracy_and_loss(model, test_loader, device)
        test_losses.append(test_loss)
        test_acc_history.append(test_accuracy)

        # Print stats
        print(f"[Epoch {epoch}/{total_epochs}] "
              f"LR: {lr} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_accuracy:.2f}%")

        # ---------------------------
        # 5. Save Metrics And Visualizations
        # ---------------------------

        # # After each epoch, we can call the function to save kernel responses and grab
        # # the list of circular variances for each kernel
        # curr_circular_variances = plot_sine_grating_responses_for_filters(model, epoch, device)
        # circular_variances.append(curr_circular_variances)

        # # We can also plot the kernel filters
        # plot_conv1_kernels(model, epoch)

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # # Loss plot
        # ax1.plot(range(1, epoch + 1), train_losses, label='Train Loss')
        # ax1.plot(range(1, epoch + 1), test_losses, label='Test Loss')
        # ax1.set_xlabel('Epoch')
        # ax1.set_ylabel('Loss')
        # ax1.set_title('Train & Test Loss')
        # ax1.legend()

        # # Accuracy plot
        # ax2.plot(range(0, epoch + 1), test_acc_history, label='Test Accuracy')
        # ax2.set_xlabel('Epoch')
        # ax2.set_ylabel('Accuracy (%)')
        # ax2.set_title('Test Accuracy')
        # ax2.legend()

        # # Plot the circular variances of each kernel as a line plot over epochs
        # # all on the same plot axis (ax3)
        # for k in range(len(curr_circular_variances)):
        #     ax3.plot(range(0, epoch + 1), [cv[k] for cv in circular_variances], label=f"Kernel {k}")

        # plt.tight_layout()
        # plt.savefig("out/imagenet_training_metrics.png")
        # plt.close(fig)


        # Save the model checkpoint with epoch/loss/accuracy
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_acc_history': test_acc_history,
            'circular_variances': circular_variances,
        }
        torch.save(checkpoint_dict, "out/imagenet.pt")  # Overwrite/update main checkpoint

    print("Training complete. Kernels, training metrics, and sine-grating responses have been saved to the 'out/' folder.")


if __name__ == "__main__":
    main()
