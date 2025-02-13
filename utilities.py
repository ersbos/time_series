import os
import glob
import pandas as pd
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import sys

from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class SeismicDataset(Dataset):
    def __init__(self, data_dir, window_size, window_stride, transform=None):
        """
        data_dir: Root folder with CSV files structured by person (e.g., person_one, person_two, etc.)
        window_size: The number of samples for each window segment.
        window_stride: The step size between successive windows.
        transform: Optional function to apply to each signal.
        """
        self.data_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
        self.window_size = window_size
        self.window_stride = window_stride
        self.transform = transform

        self.class_names = sorted(list(set(os.path.basename(os.path.dirname(f)) for f in self.data_files)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        print("Detected classes and indices:", self.class_to_idx)

        self.segments = []
        for file_path in self.data_files:
            try:
                data = pd.read_csv(file_path)
                signal = data['voltage'].values.astype('float32')
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
            n_samples = len(signal)
            if n_samples < self.window_size:
                continue  # Skip files that are shorter than one window.
            n_segments = (n_samples - self.window_size) // self.window_stride + 1
            for i in range(n_segments):
                start = i * self.window_stride
                self.segments.append((file_path, start))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        file_path, start_index = self.segments[idx]
        data = pd.read_csv(file_path)
        signal = data['voltage'].values.astype('float32')
        window_signal = signal[start_index: start_index + self.window_size]

        if self.transform:
            window_signal = self.transform(window_signal)

        class_name = os.path.basename(os.path.dirname(file_path))
        label = self.class_to_idx[class_name]
        window_signal = torch.tensor(window_signal).unsqueeze(1)
        return window_signal, label


def get_dataloaders(config):
    from torch.utils.data import random_split
    data_dir = config['data_dir']
    window_size = config['window']['size']
    window_stride = config['window']['stride']

    dataset = SeismicDataset(data_dir, window_size, window_stride)
    train_size = int(config['data_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, config, device):
    # Compute class weights to address imbalance.
    train_indices = train_loader.dataset.indices
    all_labels = []
    # train_loader.dataset is a Subset so use its underlying dataset.
    for i in train_indices:
        _, label = train_loader.dataset.dataset[i]
        all_labels.append(label)
    counts = Counter(all_labels)
    max_count = max(counts.values())
    num_classes = len(counts)
    weights_list = [max_count / counts[i] for i in range(num_classes)]
    class_weights = torch.tensor(weights_list, dtype=torch.float32).to(device)
    print("Computed class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    num_epochs = config['training']['epochs']

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            batch_acc = (predicted == labels).float().mean().item()
            sys.stdout.write(
                f'\rEpoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{total_batches}] '
                f'Loss: {loss.item():.4f} Batch Acc: {batch_acc*100:.2f}%'
            )
            sys.stdout.flush()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        wandb.log({'train_loss': epoch_train_loss, 'epoch': epoch})

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        y_preds = []
        y_trues = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_preds.extend(predicted.cpu().numpy())
                y_trues.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        wandb.log({'val_loss': epoch_val_loss, 'val_accuracy': val_accuracy, 'epoch': epoch})

        print(f"\nEpoch {epoch+1}/{num_epochs} finished: "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Compute the confusion matrix.
        cm = confusion_matrix(y_trues, y_preds)
        # Normalize the confusion matrix by row (true classes).
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

        # Get class names from the dataset if available.
        if hasattr(train_loader.dataset.dataset, "class_names"):
            class_names = train_loader.dataset.dataset.class_names
        else:
            class_names = [str(i) for i in range(num_classes)]

        # Plot the normalized confusion matrix.
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Normalized Confusion Matrix")
        plt.tight_layout()

        # Log the image to wandb.
        wandb.log({"normalized_confusion_matrix": wandb.Image(fig), "epoch": epoch})
        plt.close(fig)