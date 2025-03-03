#########################################
# 1. IMPORT LIBRARIES & SET GLOBAL VARS #
#########################################

import os
from os.path import exists

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import fcluster

from sklearn.linear_model import RidgeCV  # using RidgeCV with a fixed alpha
from sklearn.metrics import r2_score as r2_score_sklearn

import gdown

# Threshold used for selecting reliable voxels.
NCSNR_THRESHOLD = 0.2


#########################################
# 2. HELPER FUNCTIONS                   #
#########################################

def r2_over_nc(y, y_pred, ncsnr):
    """
    Compute R² score normalized by noise ceiling.
    The noise ceiling is computed as:
         NC = ncsnr^2 / (ncsnr^2 + 1/num_trials)
    If ncsnr is None, return the standard R^2.
    """
    # If ncsnr is None, compute and return standard R² score
    if ncsnr is None:
        return r2_score_sklearn(y, y_pred, multioutput="raw_values")
    
    # Number of target trials
    num_trials = 3.0
    
    # Compute noise ceiling (NC)
    NC = (ncsnr ** 2) / ((ncsnr ** 2) + (1.0 / num_trials))
    
    # Compute standard R² score
    r2 = r2_score_sklearn(y, y_pred, multioutput="raw_values")
    
    # Normalize R² by noise ceiling
    normalized_r2 = r2 / NC
    
    return normalized_r2


def get_metadata_concat_hemi(Y):
    """
    Concatenate left- and right-hemisphere metadata for voxels labeled 'nsdgeneral'
    and return the corresponding ncsnr values and metadata DataFrame.
    """
    ncsnr_full = np.concatenate((
        Y['voxel_metadata']['lh']['lh.ncsnr'],
        Y['voxel_metadata']['rh']['rh.ncsnr']
    ))
    
    nsdgeneral_idx = np.concatenate((
        Y['voxel_metadata']['lh']['lh.nsdgeneral.label'],
        Y['voxel_metadata']['rh']['rh.nsdgeneral.label']
    ))
    nsdgeneral_mask = np.logical_and(nsdgeneral_idx == 'nsdgeneral', ncsnr_full > 0)
    ncsnr_nsdgeneral = ncsnr_full[nsdgeneral_mask]
    
    metadata_lh = pd.DataFrame(Y['voxel_metadata']['lh'])
    metadata_rh = pd.DataFrame(Y['voxel_metadata']['rh'])
    nsdgeneral_metadata_df = pd.concat([metadata_lh, metadata_rh])[nsdgeneral_mask]
    
    return ncsnr_nsdgeneral, nsdgeneral_metadata_df


def get_data_dict(Y, brain_data_rep_averaged, ncsnr_nsdgeneral, nsdgeneral_metadata_df, verbose=True):
    """
    For each brain area (both streams and visual ROIs), select voxels with reliable responses
    (ncsnr above threshold) and return a dictionary with responses and ncsnr values.
    """
    data_dict = {}

    # Process streams-based areas.
    for area in ['ventral', 'parietal', 'lateral']:
        data_dict[area] = {}
        lh_area_mask = nsdgeneral_metadata_df['lh.streams.label'].astype(str).str.contains(area, na=False)
        rh_area_mask = nsdgeneral_metadata_df['rh.streams.label'].astype(str).str.contains(area, na=False)
        area_mask = np.logical_or(lh_area_mask, rh_area_mask)
        area_mask = np.logical_and(area_mask, ncsnr_nsdgeneral > NCSNR_THRESHOLD)
        
        if verbose:
            print(f"Size of area {area}: {np.sum(area_mask)}")
        
        area_data = brain_data_rep_averaged[:, area_mask]
        data_dict[area]["responses"] = area_data.copy()
        data_dict[area]["ncsnr"] = ncsnr_nsdgeneral[area_mask].copy()
        
        if verbose:
            print(f"Shape of area {area} responses: {data_dict[area]['responses'].shape}")

    # Process visual ROIs.
    for area in ['V1', 'V2', 'V3', 'V4']:
        data_dict[area] = {}
        lh_area_mask = nsdgeneral_metadata_df['lh.prf-visualrois.label'].astype(str).str.contains(area, na=False)
        rh_area_mask = nsdgeneral_metadata_df['rh.prf-visualrois.label'].astype(str).str.contains(area, na=False)
        area_mask = np.logical_or(lh_area_mask, rh_area_mask)
        area_mask = np.logical_and(area_mask, ncsnr_nsdgeneral > NCSNR_THRESHOLD)
        
        if verbose:
            print(f"Size of area {area}: {np.sum(area_mask)}")
        
        area_data = brain_data_rep_averaged[:, area_mask]
        data_dict[area]["responses"] = area_data.copy()
        data_dict[area]["ncsnr"] = ncsnr_nsdgeneral[area_mask].copy()
        
        if verbose:
            print(f"Shape of area {area} responses: {data_dict[area]['responses'].shape}")

    return data_dict


#########################################
# 3. DOWNLOAD & LOAD NSD DATA           #
#########################################

# Create a data directory if it does not exist.
datadir = os.path.join(os.getcwd(), 'data')
os.makedirs(datadir, exist_ok=True)

# Define subject and corresponding file_id.
subj = 'subj01'  # choose subject: available subjects are subj01, subj02, subj05, subj07
overwrite = False

if subj == 'subj01':
    file_id = '13cRiwhjurCdr4G2omRZSOMO_tmatjdQr'
elif subj == 'subj02':
    file_id = '1MO9reLoV4fqu6Weh4gmE78KJVtxg72ID'
elif subj == 'subj05':
    file_id = '11dPt3Llj6eAEDJnaRy8Ch5CxfeKijX_t'
elif subj == 'subj07':
    file_id = '1HX-6t4c6js6J_vP4Xo0h1fbK2WINpwem'
    
url = f'https://drive.google.com/uc?id={file_id}&export=download'
output = os.path.join(datadir, f'{subj}_nativesurface_nsdgeneral.pkl')

if not exists(output) or overwrite:
    gdown.download(url, output, quiet=False)

# Load NSD data.
Y = np.load(output, allow_pickle=True)
print("Keys in Y:", Y.keys())

# Print shapes of image bricks for each partition.
for partition in ['train', 'val', 'test']:
    print(f"Shape of image brick ({partition}):", Y['image_data'][partition].shape)


#########################################
# 4. PLOT EXAMPLE NSD IMAGE             #
#########################################

idx = 10  # example index for an image
plt.imshow(Y['image_data']['test'][idx])
plt.axis('off')
plt.savefig('nsd_image.png', bbox_inches='tight', dpi=300)
plt.close()


#########################################
# 5. PREPARE FMRI DATA                  #
#########################################

# Concatenate full brain ncsnr and nsdgeneral labels.
ncsnr_full = np.concatenate((
    Y['voxel_metadata']['lh']['lh.ncsnr'].values,
    Y['voxel_metadata']['rh']['rh.ncsnr'].values
))
nsdgeneral_idx = np.concatenate((
    Y['voxel_metadata']['lh']['lh.nsdgeneral.label'].values,
    Y['voxel_metadata']['rh']['rh.nsdgeneral.label'].values
))
print(ncsnr_full.shape, round(np.mean(ncsnr_full), 4), round(np.std(ncsnr_full), 4))
print(np.unique(nsdgeneral_idx))
print(np.count_nonzero(nsdgeneral_idx == 'nsdgeneral'))

# Select only nsdgeneral voxels with positive ncsnr.
nsdgeneral_mask = np.logical_and(nsdgeneral_idx == 'nsdgeneral', ncsnr_full > 0)
ncsnr_nsdgeneral = ncsnr_full[nsdgeneral_mask]
print(ncsnr_nsdgeneral.shape, round(np.mean(ncsnr_nsdgeneral), 4), round(np.std(ncsnr_nsdgeneral), 4))

# Combine metadata for nsdgeneral voxels.
nsdgeneral_metadata = pd.concat((
    Y['voxel_metadata']['lh'],
    Y['voxel_metadata']['rh']
))[nsdgeneral_mask]
ncsnr_nsdgeneral, nsdgeneral_metadata_df = get_metadata_concat_hemi(Y)

# Concatenate train and validation brain data and average over repetitions.
train_brain_data_cat = np.concatenate((
    Y['brain_data']['train']['lh'],
    Y['brain_data']['train']['rh']
), axis=2)
val_brain_data_cat = np.concatenate((
    Y['brain_data']['val']['lh'],
    Y['brain_data']['val']['rh']
), axis=2)
train_brain_data_cat = np.concatenate((train_brain_data_cat, val_brain_data_cat), axis=0)
train_brain_data_cat = np.mean(train_brain_data_cat, axis=1)

# Average test brain data over repetitions.
test_brain_data_cat = np.concatenate((
    Y['brain_data']['test']['lh'],
    Y['brain_data']['test']['rh']
), axis=2)
test_brain_data_cat = np.mean(test_brain_data_cat, axis=1)

# Get fMRI data dictionaries for train and test sets.
train_fmri_data = get_data_dict(Y, train_brain_data_cat, ncsnr_nsdgeneral, nsdgeneral_metadata_df)
test_fmri_data = get_data_dict(Y, test_brain_data_cat, ncsnr_nsdgeneral, nsdgeneral_metadata_df)

# Use both train and validation images for training.
train_image_data = np.concatenate((Y['image_data']['train'], Y['image_data']['val']), axis=0)
test_image_data = Y['image_data']['test']

# Define a torchvision transform: resize, center crop, convert to tensor, and normalize.
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


#########################################
# 6. DEFINE MODIFIED ALEXNET MODEL      #
#########################################

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        
        # Feature extraction layers - convolutional part of AlexNet
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Capture after this (layer 2)
            
            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Capture after this (layer 5)
            
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4 - Changed from (384, 256) to (384, 384) to match trained models
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5 - Changed from (256, 256) to (384, 256) to match trained models
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Capture after this (layer 12)
        )
        
        # Adaptive pooling for variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier network with fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),  # First FC layer (fc1) - capture this
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),  # Second FC layer (fc2) - capture this
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # Output layer
        )

    def forward(self, x: torch.Tensor) -> dict:
        # Dictionary to store intermediate activations
        features = {}
        
        # Process through convolutional layers and capture outputs at key points
        # Layer 1-2: Conv1 + ReLU + MaxPool
        x = self.features[0](x)  # Conv1
        x = self.features[1](x)  # ReLU
        x = self.features[2](x)  # MaxPool
        features["conv_pool_after_layer2"] = torch.flatten(x, 1)
        
        # Layer 3-5: Conv2 + ReLU + MaxPool
        x = self.features[3](x)  # Conv2
        x = self.features[4](x)  # ReLU
        x = self.features[5](x)  # MaxPool
        features["conv_pool_after_layer_5"] = torch.flatten(x, 1)
        
        # Layer 6-12: Conv3, Conv4, Conv5 + ReLUs + MaxPool
        x = self.features[6](x)  # Conv3
        x = self.features[7](x)  # ReLU
        x = self.features[8](x)  # Conv4
        x = self.features[9](x)  # ReLU
        x = self.features[10](x)  # Conv5
        x = self.features[11](x)  # ReLU
        x = self.features[12](x)  # MaxPool
        features["conv_pool_after_layer_12"] = torch.flatten(x, 1)
        
        # Apply adaptive pooling and flatten for fully connected layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # First fully connected layer (fc1)
        x = self.classifier[0](x)  # Dropout
        x = self.classifier[1](x)  # Linear (fc1)
        features["fc1"] = x.clone()
        x = self.classifier[2](x)  # ReLU
        
        # Second fully connected layer (fc2)
        x = self.classifier[3](x)  # Dropout
        x = self.classifier[4](x)  # Linear (fc2)
        features["fc2"] = x.clone()
        x = self.classifier[5](x)  # ReLU
        
        # Final output layer
        x = self.classifier[6](x)  # Final Linear layer
        
        return features


#########################################
# 7. SET UP MODELS (RANDOM & PRETRAINED)#
#########################################

# Set device.
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# Model with random initialization.
model_random = AlexNet().to(device)
model_random.eval()

# Model loaded from an ImageNet checkpoint.
model_loaded = AlexNet().to(device)
# TODO: Replace the placeholder with the actual path
checkpoint = torch.load("out/imagenet.pt", map_location=device)
model_loaded.load_state_dict(checkpoint['model_state_dict'])
model_loaded.eval()

# Model loaded from a barcode checkpoint.
model_barcode = AlexNet(num_classes=32).to(device)
# TODO: Replace the placeholder with the actual path
checkpoint = torch.load("out/barcode.pt", map_location=device)
model_barcode.load_state_dict(checkpoint['model_state_dict'])
model_barcode.eval()


########################################
# 8. PROCESS IMAGES & EXTRACT FEATURES  #
########################################

def get_model_activations(model, image_data, batch_size=32):
    """
    Process images through the given model in batches and return a dictionary
    containing activations for each recorded feature.
    """
    model.eval()
    all_features = {}
    n = len(image_data)
    
    for i in range(0, n, batch_size):
        # Convert each numpy image to a PIL image and apply preprocessing.
        batch_imgs = image_data[i:i+batch_size]
        batch_tensors = torch.stack([preprocess(Image.fromarray(img)) for img in batch_imgs]).to(device)
        with torch.no_grad():
            out = model(batch_tensors)
        
        if i == 0:
            # Initialize storage for each feature key.
            for key in out.keys():
                all_features[key] = []
        for key, feat in out.items():
            all_features[key].append(feat.cpu().numpy())
    
    # Concatenate results for each key.
    activations = {key: np.concatenate(val, axis=0) for key, val in all_features.items()}
    return activations

# Specify the layers we want to evaluate.
desired_layers = ["conv_pool_after_layer2", "conv_pool_after_layer_5",
                  "conv_pool_after_layer_12", "fc1", "fc2"]

# Define the models to be evaluated.
models = {
    'random': model_random,
    'imagenet': model_loaded,
    'barcode': model_barcode
}


##########################################
# 9. REGRESSION & EVALUATION             #
##########################################

# Dictionary to store productivity scores.
model_results = {}

# Get a list of brain areas from the fmri data dictionary.
brain_areas = list(train_fmri_data.keys())

# Define the desired order for brain areas on the x-axis.
desired_areas_order = ["V1", "V2", "V3", "V4", "ventral", "parietal", "lateral"]

# For each model, compute activations, fit Ridge regression, and compute normalized R2 scores
for model_name, model_instance in models.items():
    print(f"\nProcessing model: {model_name}")
    # Extract features for train and test sets
    features_train = get_model_activations(model_instance, train_image_data, batch_size=32)
    features_test = get_model_activations(model_instance, test_image_data, batch_size=32)
    
    # Initialize dictionary to hold scores (rows: layers, columns: brain areas)
    scores = {layer: {} for layer in desired_layers}
    
    for layer in desired_layers:
        if layer not in features_train:
            print(f"Warning: Layer {layer} not found in model outputs. Skipping...")
            continue
        
        X_train = features_train[layer]  # (n_train, feature_dim)
        X_test = features_test[layer]   # (n_test, feature_dim)
        print(f"Layer {layer}: train features {X_train.shape}, test features {X_test.shape}")
        
        for area in brain_areas:
            # 1. Extract fMRI responses for this brain area
            y_train = train_fmri_data[area]["responses"]
            y_test = test_fmri_data[area]["responses"]
            
            # 2. Get noise ceiling values for normalization
            ncsnr = test_fmri_data[area]["ncsnr"]
            
            # 3. Check sample consistency
            if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
                print(f"Warning: Sample count mismatch for {area}. X_train: {X_train.shape[0]}, y_train: {y_train.shape[0]}, X_test: {X_test.shape[0]}, y_test: {y_test.shape[0]}. Skipping...")
                continue
            
            # 4. Ridge Regression with cross-validation
            # Create a list of candidate alphas for RidgeCV
            alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000, 1e5, 1e6, 1e7, 1e8]
            
            # Create and fit RidgeCV model
            ridge_model = RidgeCV(alphas=alphas, store_cv_values=True)
            ridge_model.fit(X_train, y_train)
            
            # Get predictions on test data
            y_pred = ridge_model.predict(X_test)
            
            # Report optimal alpha and standard R² score
            print(f"  Area {area}: Optimal alpha = {ridge_model.alpha_}")
            r2 = r2_score_sklearn(y_test, y_pred, multioutput="raw_values")
            print(f"  Area {area}: Standard R² score = {np.mean(r2):.4f}")
            
            # 5. Compute normalized R² score using noise ceiling
            norm_r2 = r2_over_nc(y_test, y_pred, ncsnr)
            
            # 6. Store average normalized R² score across all voxels
            scores[layer][area] = np.mean(norm_r2)
            print(f"  Area {area}: Normalized R² score = {scores[layer][area]:.4f}")
    
    # 7. Final aggregation and visualization
    # Convert scores dictionary to DataFrame for visualization
    df_scores = pd.DataFrame(scores).T
    
    # Reorder columns to the desired order
    df_scores = df_scores[desired_areas_order]
    model_results[model_name] = df_scores
    
    # Plot heatmap of normalized R² scores
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_scores, annot=True, cmap="viridis", fmt=".3f")
    plt.title(f"Model Productivity (R² over noise ceiling) - {model_name}")
    plt.xlabel("Brain Area")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(f"heatmap_{model_name}.png", dpi=300)
    plt.close()