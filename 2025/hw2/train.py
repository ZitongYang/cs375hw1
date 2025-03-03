"""
Modified AlexNet for Barcode Classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import seaborn as sns
from tqdm import tqdm
from typing import Tuple, List
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

sns.set_theme(style="whitegrid")

#######################################################################################
### BARCODE DATASET CLASS
#######################################################################################

class BarcodeDataset(Dataset):
    """
    A Dataset that yields (image, label) pairs for 32-bit unsigned integers.
    Each integer is converted to a 224x224x3 "barcode" image with 32 vertical
    stripes (each stripe is 7 pixels wide and 224 pixels tall).
    
    - 'split': either 'train' or 'val'
    - 'length': how many samples per epoch
    - 'val_numbers': a list of unique integers used ONLY for validation
    - 'seed': to control reproducibility (only used for training)
    """
    def __init__(self, split: str, length: int, val_numbers: list, seed: int = 1110):
        super().__init__()
        assert split in ['train', 'val']
        self.split = split
        self.length = length
        self.val_numbers = val_numbers
        
        # For reproducible random sampling
        self.rng = random.Random(seed if split == 'train' else seed + 9999)

        # Convert val_numbers to a set for quick membership checks
        self.val_set = set(val_numbers)
        self.max_uint32 = 2**32  # 0 to 2^32-1

        # For val split, we'll store val_numbers as a list
        if self.split == 'val':
            self.val_numbers_list = list(self.val_numbers)
            self.rng.shuffle(self.val_numbers_list)

    def __len__(self):
        # We define the 'epoch size' as self.length
        return self.length

    def __getitem__(self, index):
        if self.split == 'train':
            # Randomly pick a number not in the validation set
            while True:
                candidate = self.rng.randrange(0, self.max_uint32)
                if candidate not in self.val_set:
                    number = candidate
                    break
        else:
            # Validation: pick from val_numbers_list (cycling if needed)
            number = self.val_numbers_list[index % len(self.val_numbers_list)]

        # Build the (image, label) pair
        img_tensor = self.make_barcode_data(number)      # shape: [3, 224, 224]
        label_tensor = self.int_to_32bit_label(number)     # shape: [32]
        return img_tensor, label_tensor

    @staticmethod
    def make_barcode_data(number: int) -> torch.Tensor:
        """
        Given an unsigned 32-bit integer, produce a 224x224x3 "barcode" image.
        - 32 vertical stripes, each 7 pixels wide => total width = 224
        - Stripe i is black if bit i=1, white if bit i=0.
        """
        # Convert to a 32-bit binary array (bit 31 = LSB, bit 0 = MSB)
        bits = [(number >> i) & 1 for i in range(32)]
        bits.reverse()  # Now bits[0] is the MSB, bits[31] is the LSB

        # Create blank white image [H=224, W=224, C=3]
        img = np.ones((224, 224, 3), dtype=np.float32)  # 1.0 = white

        # Fill stripes for bits=1 with black (0.0)
        for i, bit in enumerate(bits):
            col_start = i * 7
            col_end = col_start + 7
            if bit == 1:
                img[:, col_start:col_end, :] = 0.0

        # Convert to torch tensor, shape [3, 224, 224]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor

    @staticmethod
    def int_to_32bit_label(number: int) -> torch.Tensor:
        """
        Convert the integer into a 32-bit binary label (0/1) (float32).
        bits[0] = MSB, bits[31] = LSB.
        """
        bits = [(number >> i) & 1 for i in range(32)]
        bits.reverse()
        return torch.tensor(bits, dtype=torch.float32)


#######################################################################################
### MODEL CLASS
#######################################################################################

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 32, dropout: float = 0.5) -> None:
        # Changed num_classes from 1000 to 32 for barcode bits
        super().__init__()
        
        # Feature extraction layers - kept the same as original AlexNet
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

        # Classifier layers - last layer outputs 32 values (one per bit)
        self.classifier = nn.Sequential(
            # FC6
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            # FC7
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            # FC8 - Modified to output 32 values (one per bit)
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
    Computes the bit accuracy and loss of the model on the barcode validation set.

    Parameters:
        - model: torch.nn.Module, the model to evaluate
        - data_loader: torch.utils.data.DataLoader, the data loader to evaluate
        - device: torch.device, the device to run the evaluation on

    Returns:
        - accuracy: float, the bit-wise accuracy of the model on the data_loader
        - loss: float, the average loss of the model on the data_loader
    """
    model.eval()  # Set model to evaluation mode
    criterion = nn.BCEWithLogitsLoss()  # Use BCE loss for binary classification
    total_correct_bits = 0
    total_bits = 0
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient calculations
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Get predictions (>0 for binary classification with logits)
            predicted = (outputs > 0).float()
            
            # Update metrics
            total_bits += labels.numel()  # Total number of bits (batch_size * 32)
            total_correct_bits += (predicted == labels).sum().item()

    # Calculate final metrics
    accuracy = (total_correct_bits / total_bits) * 100.0  # Convert to percentage
    avg_loss = running_loss / (total_bits / 32)  # Divide by number of samples

    return accuracy, avg_loss


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
    num_workers = 4  # Reduced from original since this is a synthetic dataset
    momentum = 0.9
    weight_decay = 0.0
    seed = 1110

    # Barcode dataset parameters
    train_samples_per_epoch = 50000  # Number of training samples per epoch
    val_samples_per_epoch = 5000     # Number of validation samples per epoch
    val_numbers = [random.randrange(0, 2**32) for _ in range(1000)]  # Reserve 1000 numbers for validation
    
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
    
    # ---------------------------
    # 2. Data Preparation
    # ---------------------------
    # Create Barcode datasets
    train_dataset = BarcodeDataset(
        split='train',
        length=train_samples_per_epoch,
        val_numbers=val_numbers,
        seed=seed
    )
    
    val_dataset = BarcodeDataset(
        split='val',
        length=val_samples_per_epoch,
        val_numbers=val_numbers,
        seed=seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # ---------------------------
    # 3. Model Setup
    # ---------------------------
    model = AlexNet(num_classes=32).to(device)  # 32 bits output

    optimizer = optim.SGD(
        model.parameters(), 
        lr=get_lr_for_epoch(1),  # Start with initial LR for epoch=1
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Use Binary Cross Entropy with Logits Loss for binary classification
    criterion = nn.BCEWithLogitsLoss()

    # Lists to store losses and accuracies
    train_losses = []
    val_losses = []
    val_acc_history = []

    # Check if there's an existing checkpoint we can load
    checkpoint_path = "out/barcode.pt"  # Changed from "imagenet.pt" to "barcode.pt"
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
        val_losses = checkpoint['val_losses']
        val_acc_history = checkpoint['val_acc_history']

        print(f"Resuming from epoch {start_epoch}...")
    else:
        print("No checkpoint found. Starting from scratch.")

        # Compute accuracy on the validation set
        val_accuracy, val_loss = evaluate_accuracy_and_loss(model, val_loader, device)
        val_acc_history.append(val_accuracy)
        val_losses.append(val_loss)

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

        # Compute accuracy on the validation set
        val_accuracy, val_loss = evaluate_accuracy_and_loss(model, val_loader, device)
        val_losses.append(val_loss)
        val_acc_history.append(val_accuracy)

        # Print stats
        print(f"[Epoch {epoch}/{total_epochs}] "
              f"LR: {lr} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Bit Accuracy: {val_accuracy:.2f}%")

        # ---------------------------
        # 5. Save Metrics And Visualizations
        # ---------------------------

        # Save the model checkpoint with epoch/loss/accuracy
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_acc_history': val_acc_history,
        }
        torch.save(checkpoint_dict, "out/barcode.pt")  # Changed from "imagenet.pt" to "barcode.pt"

    print("Training complete.")


if __name__ == "__main__":
    main()
