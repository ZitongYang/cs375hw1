# AlexNet Training Report

## Code Implementation

I implemented AlexNet following the original architecture specifications with some modern adaptations. The key components include:

1. **Feature Extraction Layers**:
   - 5 convolutional layers with ReLU activations
   - 3 max pooling layers after Conv1, Conv2, and Conv5
   - First layer (Conv1): 11x11 kernels, 64 filters, stride 4
   
2. **Classifier Layers**:
   - 3 fully connected layers (4096 → 4096 → 1000)
   - Dropout (p=0.5) before FC6 and FC7
   - ReLU activations between linear layers

The training loop includes learning rate scheduling:
- Epochs 1-15: lr = 0.01
- Epochs 16-25: lr = 0.001
- Epochs 26-30: lr = 0.0001

## Training Analysis

![Training Metrics](out/training_metrics.png)

### Training Trends
- **Loss**: The training and test loss show rapid decrease in the first few epochs, indicating efficient learning of features. The convergence pattern suggests the model is learning without significant overfitting.

- **Accuracy**: The test accuracy shows steady improvement, particularly in the early epochs, reaching 22% by the end of training.

- **Circular Variance**: The kernel circular variance plots indicate that directional selectivity begins to emerge around 15 epochs into training. This suggests that the network starts learning orientation-specific features relatively early in the training process.

## Kernel Visualization Analysis

![Conv1 Kernels](out/conv1_kernels_epoch_0.png)
![Conv1 Kernels](out/conv1_kernels_epoch_1.png)
![Conv1 Kernels](out/conv1_kernels_epoch_23.png)
![Conv1 Kernels](out/conv1_kernels_epoch_24.png)

The first layer kernels show several interesting patterns:
- Multiple Gabor-like filters that detect edges at various orientations
- Color-opponent kernels that respond to specific color transitions
- Some high-frequency texture detectors

## Individual Filter Analysis

![Kernel Response Plots](out/kernel_responses_01/kernel_0_responses.png)
![Kernel Response Plots](out/kernel_responses_15/kernel_24_responses.png)
![Kernel Response Plots](out/kernel_responses_23/kernel_40_responses.png)