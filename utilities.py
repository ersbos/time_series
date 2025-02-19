import torch
import torch.onnx
import os
import glob
import pandas as pd
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import sys
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau


from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Footstep_detector import detect_step_events

# Enable GPU optimizations
cudnn.benchmark = True  
scaler = GradScaler()


class SeismicDataset(Dataset):
    def __init__(self, data_dir, window_size, window_stride, transform=None,
                 threshold_factor=3, noise_sigma=25.0,
                 undersample_noise=1, noise_sampling_ratio=1.0):
        """
        data_dir: Root folder with CSV files structured by person.
        window_size: The number of samples in each window segment.
        window_stride: The step size between successive windows.
        transform: Optional function to apply to each signal window.
        threshold_factor: Passed to detect_step_events; used for thresholding.
        noise_sigma: Passed to detect_step_events; the assumed sensor noise level.

        undersample_noise: Boolean, if True will randomly drop noise segments.
        noise_sampling_ratio: The desired ratio between the number of noise segments
                              and the number of step (non-noise) segments.
                              E.g. 1.0 means a 1:1 ratio; default is 1.0.

        Labelation:
          - For each CSV file, all windows are extracted uniformly using window_stride.
          - The detect_step_events function is used on the full signal.
          - Any window whose starting index is among the detected step indices is
            labeled with the person’s label (from 0 to num_persons-1),
            while all other windows will be labeled as noise (label = num_persons).
        """
        self.data_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
        self.window_size = window_size
        self.window_stride = window_stride
        self.transform = transform
        self.threshold_factor = threshold_factor
        self.noise_sigma = noise_sigma
        self.undersample_noise = undersample_noise
        self.noise_sampling_ratio = noise_sampling_ratio

        # Get a sorted list of person classes from the folder names.
        self.class_names = sorted(list(set(os.path.basename(os.path.dirname(f)) for f in self.data_files)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        print("Detected person classes and indices:", self.class_to_idx)

        # Define the noise class label as an extra category.
        self.noise_label = len(self.class_names)
        print("Noise class label index:", self.noise_label)

        self.segments = []  # List of tuples: (file_path, start_index)
        self.labels = []    # List of labels corresponding to each window

        # Process each CSV file.
        for file_path in self.data_files:
            try:
                data = pd.read_csv(file_path)
                signal = data['voltage'].values.astype('float32')
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            n_samples = len(signal)
            if n_samples < self.window_size:
                continue

            # Get the person label from the parent folder.
            class_name = os.path.basename(os.path.dirname(file_path))
            person_label = self.class_to_idx[class_name]

            # Run step detection (using the same window_size used for detection).
            # detected_indices are the starting indices of windows in which a step was detected.
            detected_indices = detect_step_events(signal,
                                                  window_size=self.window_size,
                                                  threshold_factor=self.threshold_factor,
                                                  noise_sigma=self.noise_sigma)
            detected_set = set(detected_indices)

            # Uniformly generate segments using window_stride.
            n_segments = (n_samples - self.window_size) // self.window_stride + 1
            for i in range(n_segments):
                start = i * self.window_stride
                self.segments.append((file_path, start))
                # If the segment is one of the detected ones, assign the person label.
                # Otherwise, assign the noise label.
                if start in detected_set:
                    self.labels.append(person_label)
                else:
                    self.labels.append(self.noise_label)

        # Perform undersampling of noise segments if requested.
        if self.undersample_noise == 1:
            # Indices for non-noise (step) and noise windows.
            step_indices = [i for i, label in enumerate(self.labels) if label != self.noise_label]
            noise_indices = [i for i, label in enumerate(self.labels) if label == self.noise_label]
            print(f"Before undersampling: {len(step_indices)} step segments, {len(noise_indices)} noise segments.")

            # Determine the desired number of noise segments.
            desired_noise_count = int(len(step_indices) * self.noise_sampling_ratio)
            if len(noise_indices) > desired_noise_count:
                # Randomly sample the desired number of noise indices.
                sampled_noise_indices = np.random.choice(noise_indices, desired_noise_count, replace=False).tolist()
                # Combine the selected step and noise segments, preserving the original order.
                selected_indices = sorted(step_indices + sampled_noise_indices)
                self.segments = [self.segments[i] for i in selected_indices]
                self.labels = [self.labels[i] for i in selected_indices]
                print(f"After undersampling: {len(step_indices)} step segments, {desired_noise_count} noise segments.")
            else:
                print("Noise segments are fewer than or equal to the undersampling target; no undersampling applied.")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        file_path, start_index = self.segments[idx]
        try:
            data = pd.read_csv(file_path)
            signal = data['voltage'].values.astype('float32')
        except Exception as e:
            raise RuntimeError(f"Error reading {file_path}: {e}")

        window_signal = signal[start_index: start_index + self.window_size]
        if self.transform:
            window_signal = self.transform(window_signal)

        # Add the channel dimension as the last dimension.
        window_signal = torch.tensor(window_signal).unsqueeze(-1)  # Shape: [window_length, 1]
        label = self.labels[idx]
        return window_signal, label


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False,
                 save_path_pth=None, save_path_onnx=None, dummy_input=None, device="cpu"):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement before stopping.
            delta (float): Minimum improvement in the validation loss to be considered as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            save_path_pth (str): File path to save the best model's state dict (.pth).
            save_path_onnx (str): File path to export the best model in ONNX format (.onnx).
            dummy_input (torch.Tensor): A valid dummy input tensor for exporting the model.
            device (str): The device on which dummy_input resides.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

        # Save-related parameters.
        self.save_path_pth = save_path_pth
        self.save_path_onnx = save_path_onnx
        self.dummy_input = dummy_input.to(device) if dummy_input is not None else None
        self.device = device

    def step(self, val_loss, model):
        """
        Call this method after each epoch to update the early stopping counter.
        If a new best loss is observed, save the model globally.

        Args:
            val_loss (float): The current epoch's validation loss.
            model (torch.nn.Module): The model to potentially save.
        """
        # For the first call, set the best loss and save the model.
        if self.best_loss is None:
            self.best_loss = val_loss
            save_model(model, self.save_path_pth, self.save_path_onnx, self.dummy_input, self.device, self.verbose)
            if self.verbose:
                print(f"Initial loss set to {val_loss:.6f} and model saved.")
        # If the loss hasn't improved by at least delta, increment the counter.
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}. "
                      f"No significant improvement (delta={self.delta}).")
            if self.counter >= self.patience:
                self.early_stop = True
        # If the validation loss improved sufficiently, reset the counter and save the model.
        else:
            if self.verbose:
                print(f"Validation loss improved from {self.best_loss:.6f} to {val_loss:.6f}. Saving model and resetting counter.")
            self.best_loss = val_loss
            self.counter = 0
            save_model(model, self.save_path_pth, self.save_path_onnx, self.dummy_input, self.device, self.verbose)

def get_dataloaders(config):
    from torch.utils.data import random_split
    data_dir = config['data_dir']
    window_size = config['window']['size']
    window_stride = config['window']['stride']

    # Pass the undersampling parameters from config if they exist.
    undersample_noise =  config['training']['undersample_noise']
    noise_sampling_ratio =  config['training']['noise_sampling_ratio']

    dataset = SeismicDataset(data_dir, window_size, window_stride,
                             undersample_noise=undersample_noise,
                             noise_sampling_ratio=noise_sampling_ratio)
    train_size = int(config['data_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=6, pin_memory=True)
    print("Dataloader is finished!")
    return train_loader, val_loader


def save_model(model, save_path_pth=None, save_path_onnx=None, dummy_input=None, device="cpu", verbose=False):
    """
    Saves the model in both .pth (PyTorch state_dict) and .onnx formats if respective paths are provided.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        save_path_pth (str): File path to save the model's state_dict (.pth).
        save_path_onnx (str): File path to export the model in ONNX format (.onnx).
        dummy_input (torch.Tensor): A valid dummy input tensor required for exporting the model to ONNX.
        device (str): The device on which dummy_input resides.
        verbose (bool): If True, prints messages about the saving process.
    """
    # Save the PyTorch checkpoint.
    if save_path_pth is not None:
        torch.save(model.state_dict(), save_path_pth)
        if verbose:
            print(f"Saved PyTorch model state dict to {save_path_pth}")
    # Export model to ONNX.
    if save_path_onnx is not None and dummy_input is not None:
        model.eval()  # Ensure model is in evaluation mode for a stable export.
        torch.onnx.export(
            model,
            dummy_input,
            save_path_onnx,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        if verbose:
            print(f"Exported model to ONNX format at {save_path_onnx}")


def train_model(model, train_loader, val_loader, config, device):
    # Compute class weights to address imbalance.
    train_indices = train_loader.dataset.indices
    all_labels = [train_loader.dataset.dataset.labels[i] for i in train_indices]

    counts = Counter(all_labels)
    max_count = max(counts.values())
    num_classes = len(counts)
    weights_list = [max_count / counts[i] for i in range(num_classes)]
    class_weights = torch.tensor(weights_list, dtype=torch.float32)
    print("Computed class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = config['training']['epochs']
    save_path = config['saving']['path']
    save_path_pth = save_path +  "best_model.pth"
    save_path_onnx = save_path + "best_model.onnx"
    train_losses = []   # Initialize lists for tracking losses
    val_losses = []
    num_inputs = num_inputs = config['model']['num_inputs']  # This should match the num_inputs used during model initialization.
    batch_size = config["training"]["batch_size"]
    time_steps = config["window"]["size"]

    dummy_input = torch.randn(batch_size, time_steps, num_inputs).to(device)

    early_stopping = EarlyStopping(
            patience=5,
            delta=0.001,
            verbose=True,
            save_path_pth= save_path_pth,
            save_path_onnx=save_path_onnx,
            dummy_input=dummy_input,
            device=device
        )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            batch_acc = (predicted == labels).float().mean().item()
            sys.stdout.write(
                f'\rEpoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{total_batches}] '
                f'Loss: {loss.item():.4f} Batch Acc: {batch_acc*100:.2f}%'
            )
            sys.stdout.flush()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        wandb.log({'train_loss': epoch_train_loss, 'epoch': epoch})

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        y_preds = []
        y_trues = []

        with torch.no_grad():
            with autocast():
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
        val_losses.append(epoch_val_loss)
        val_accuracy = correct / total

        scheduler.step(epoch_val_loss)
        # Log validation loss first (if desired)
        wandb.log({'val_loss': epoch_val_loss, 'epoch': epoch})
        wandb.log({'val_accuracy': val_accuracy, 'epoch': epoch})

        print(f"\nEpoch {epoch+1}/{num_epochs} finished: "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Compute the confusion matrix.
        cm = confusion_matrix(y_trues, y_preds)
        # Normalize the confusion matrix by row (true classes).
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

        # Get class names from the dataset if available.
        if hasattr(train_loader.dataset.dataset, "class_names"):
            class_names = train_loader.dataset.dataset.class_names + ['noise']
        else:
            class_names = [str(i) for i in range(num_classes)] + ['noise']

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

        # Optionally, plot the combined training and validation loss curve.
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epoch+2), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, epoch+2), val_losses, label='Val Loss', marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()

        # Log the loss curve to wandb.
        wandb.log({"loss_curve": wandb.Image(plt.gcf()), "epoch": epoch})
        plt.close()  # Close the current figure to free up memory.
        early_stopping.step(epoch_val_loss, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    save_model(model,
           save_path_pth=save_path_pth,
           save_path_onnx=save_path_pth,
           dummy_input=dummy_input,
           device=device,
           verbose=True)