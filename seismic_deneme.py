import random
from utilities import get_dataloaders
import yaml

# Load configuration
config_path = "config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Get data loaders
train_loader, val_loader = get_dataloaders(config)

# Get dataset indices and labels
train_indices = train_loader.dataset.indices
train_labels = train_loader.dataset.dataset.labels
val_indices = val_loader.dataset.indices
val_labels = val_loader.dataset.dataset.labels

# Sample 100 random indices
train_sample_indices = random.sample(train_indices, 100)
val_sample_indices = random.sample(val_indices, 100)

# Print selected indices and labels
print("Train Sample Indices and Labels:")
for idx in train_sample_indices:
    print(f"Index: {idx}, Label: {train_labels[idx]}")

print("\nValidation Sample Indices and Labels:")
for idx in val_sample_indices:
    print(f"Index: {idx}, Label: {val_labels[idx]}")