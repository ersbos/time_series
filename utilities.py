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
import contextlib
import io

from Footstep_detector import detect_step_events

# Enable GPU optimizations
cudnn.benchmark = True  
scaler = GradScaler()


class SeismicDataset(Dataset):
    def __init__(self, data_dir, window_size, window_stride, transform=None,
                 threshold_factor=3, noise_sigma=16.0,
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
        # Get a list of CSV files.
        self.data_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
        self.window_size = window_size
        self.window_stride = window_stride
        self.transform = transform
        self.threshold_factor = threshold_factor
        self.noise_sigma = noise_sigma
        self.undersample_noise = undersample_noise
        self.noise_sampling_ratio = noise_sampling_ratio

        # Preload CSV files into memory.
        # Store each file's signal data in a dictionary.
        print("Cache operations has began...")
        self.data_cache = {}
        for file_path in self.data_files:
            try:
                data = pd.read_csv(file_path)
                signal = data['voltage'].values.astype('float32')
                self.data_cache[file_path] = signal
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Get a sorted list of person classes using the folder names.
        self.class_names = sorted(list(set(os.path.basename(os.path.dirname(f)) for f in self.data_files)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        print("Detected person classes and indices:", self.class_to_idx)

        # Define the noise class label as an extra category.
        self.noise_label = len(self.class_names)
        print("Noise class label index:", self.noise_label)

        self.segments = []  # List of tuples: (file_path, start_index)
        self.labels = []    # List of labels corresponding to each window

        # Process every cached CSV file.
        for file_path, signal in self.data_cache.items():
            n_samples = len(signal)
            if n_samples < self.window_size:
                continue

            # Get the person label from the parent folder.
            class_name = os.path.basename(os.path.dirname(file_path))
            person_label = self.class_to_idx[class_name]

            # Run step detection on the full signal.
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
                # If the segment is among the detected ones, assign the person label.
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
        # Retrieve the preloaded signal from the cache.
        signal = self.data_cache[file_path]
        window_signal = signal[start_index: start_index + self.window_size]

        if self.transform:
            window_signal = self.transform(window_signal)

        # Convert the window signal into a torch tensor and add a channel dimension.
        window_signal = torch.tensor(window_signal).unsqueeze(-1)  # Shape: [window_size, 1]
        label = self.labels[idx]
        return window_signal, label
    
class SeismicDatasetBasic(Dataset):
    def __init__(self, data_dir, window_size, window_stride, transform=None):
        """
        data_dir: Root folder with CSV files structured by person.
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
        # Precompute and store labels alongside segments.
        self.labels = []  # this will hold the label for each window segment

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

            n_segments = (n_samples - self.window_size) // self.window_stride + 1
            # Get the label now, without reading the CSV each time.
            class_name = os.path.basename(os.path.dirname(file_path))
            label = self.class_to_idx[class_name]

            for i in range(n_segments):
                start = i * self.window_stride
                self.segments.append((file_path, start))
                self.labels.append(label)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        file_path, start_index = self.segments[idx]
        data = pd.read_csv(file_path)
        signal = data['voltage'].values.astype('float32')
        window_signal = signal[start_index: start_index + self.window_size]

        if self.transform:
            window_signal = self.transform(window_signal)

        # Note: we don't need to recompute the label, use the precomputed one.
        label = self.labels[idx]

        window_signal = torch.tensor(window_signal).unsqueeze(1)
        return window_signal, label

class PreloadedSeismicDataset(Dataset):
    def __init__(self, data_dir, window_size, window_stride, transform=None):
        """
        This dataset caches CSV file data in memory (CPU RAM) during initialization.
        Then __getitem__ only works on already loaded arrays instead of reading from disk.

        Args:
            data_dir (str): Root folder with CSV files structured by person.
            window_size (int): The number of samples in each window segment.
            window_stride (int): The step size between successive windows.
            transform (callable, optional): A function/transform to apply to each window.
        """
        self.data_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
        self.window_size = window_size
        self.window_stride = window_stride
        self.transform = transform

        # Preload all CSV files into a dict to cache them in RAM.
        # Keys are file paths and values are the corresponding signal arrays (numpy arrays).
        print("cache operations has began...")
        self.data_cache = {}
        for file_path in self.data_files:
            try:
                data = pd.read_csv(file_path)
                signal = data['voltage'].values.astype('float32')
                self.data_cache[file_path] = signal
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Get sorted list of person classes from the folder names.
        self.class_names = sorted(list(set(os.path.basename(os.path.dirname(f)) for f in self.data_files)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        print("Detected classes and indices:", self.class_to_idx)

        self.segments = []
        self.labels = []  # precomputed labels

        # Process each preloaded file.
        for file_path, signal in self.data_cache.items():
            n_samples = len(signal)
            if n_samples < self.window_size:
                continue

            # Determine the label from parent folder.
            class_name = os.path.basename(os.path.dirname(file_path))
            label = self.class_to_idx[class_name]

            # Create window segments for this file.
            n_segments = (n_samples - self.window_size) // self.window_stride + 1
            for i in range(n_segments):
                start = i * self.window_stride
                self.segments.append((file_path, start))
                self.labels.append(label)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        file_path, start_index = self.segments[idx]
        # Retrieve the preloaded signal from the cache.
        signal = self.data_cache[file_path]
        window_signal = signal[start_index: start_index + self.window_size]

        # Apply transformation if provided.
        if self.transform:
            window_signal = self.transform(window_signal)

        # Convert to a tensor and add a channel dimension.
        window_signal = torch.tensor(window_signal).unsqueeze(1)  # Shape: [window_size, 1]
        label = self.labels[idx]
        return window_signal, label

class EarlyStopping:
    def __init__(self, patience=8, delta=0.001, verbose=False,
                 base_save_path=None, dummy_input=None, device="cpu"):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement before stopping.
            delta (float): Minimum improvement in the validation loss to be considered as an improvement.
            verbose (bool): If True, prints messages for each validation loss improvement.
            base_save_path (str): Base folder path where checkpoint subfolders (per epoch) will be created.
            dummy_input (torch.Tensor): A valid dummy input tensor for exporting the model.
            device (str): The device on which dummy_input resides.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

        # Save-related parameters: using a base path to create subfolders per epoch.
        self.base_save_path = base_save_path
        self.dummy_input = dummy_input.to(device) if dummy_input is not None else None
        self.device = device

    def step(self, epoch, val_loss, model):
        """
        Call this method after each epoch to update the early stopping counter.
        If a new best loss is observed, save the model inside the checkpoint folder for that epoch.

        Args:
            epoch (int): The current epoch number (1-indexed).
            val_loss (float): The current epoch's validation loss.
            model (torch.nn.Module): The model to potentially save.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            save_model(model, self.base_save_path, epoch, self.dummy_input, self.device, self.verbose)
            if self.verbose:
                print(f"Initial loss set to {val_loss:.6f} and model saved at epoch {epoch}.")
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}. "
                      f"No significant improvement (delta={self.delta}).")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"Validation loss improved from {self.best_loss:.6f} to {val_loss:.6f}. "
                      f"Saving model at epoch {epoch} and resetting the counter.")
            self.best_loss = val_loss
            self.counter = 0
            save_model(model, self.base_save_path, epoch, self.dummy_input, self.device, self.verbose)

def get_dataloaders(config):
    from torch.utils.data import random_split
    data_dir = config['data_dir']
    window_size = config['window']['size']
    window_stride = config['window']['stride']

    # Pass the undersampling parameters from config if they exist.
    undersample_noise =  config['training']['undersample_noise']
    noise_sampling_ratio =  config['training']['noise_sampling_ratio']
    number_of_workers = config['training']['number_of_workers']
    '''
    dataset = SeismicDataset(data_dir, window_size, window_stride,
                             undersample_noise=undersample_noise,
                             noise_sampling_ratio=noise_sampling_ratio)
    '''
    dataset = PreloadedSeismicDataset(data_dir,window_size,window_stride)
    train_size = int(config['data_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=number_of_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=number_of_workers, pin_memory=True)
    print("Dataloader is finished!")
    return train_loader, val_loader


def save_model(model, base_save_path, epoch, dummy_input=None, device="cpu", verbose=False):
    """
    Saves the model in both .pth (PyTorch state_dict) and .onnx formats inside an epoch subfolder.
    Diagnostic output from torch.onnx.export is captured and only printed in the event of an error.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        base_save_path (str): The base folder in which epoch subfolders will be created.
        epoch (int): The current epoch number (1-indexed) to create a checkpoint folder.
        dummy_input (torch.Tensor): A valid dummy input tensor required for exporting the model to ONNX.
        device (str): The device on which dummy_input resides.
        verbose (bool): If True, prints messages about the saving process (when no errors occur).
    """
    epoch_folder = os.path.join(base_save_path, f"epoch_{epoch}")
    os.makedirs(epoch_folder, exist_ok=True)

    # Save the PyTorch checkpoint.
    save_path_pth = os.path.join(epoch_folder, "best_model.pth")
    torch.save(model.state_dict(), save_path_pth)
    if verbose:
        print(f"Saved PyTorch model state dict to {save_path_pth}")

    # Export model to ONNX with conditional output.
    if dummy_input is not None:
        model.eval()  # Ensure model is in evaluation mode for stability.
        save_path_onnx = os.path.join(epoch_folder, "best_model.onnx")
        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer):
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
        except Exception as e:
            # If an error occurs, print the captured output and the error.
            captured_output = output_buffer.getvalue()
            print("Error during ONNX export:")
            print(captured_output)
            raise e

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
    factor = config['training']['factor']
    scheduler_patience = config['training']['patience']
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=scheduler_patience)

    num_epochs = config['training']['epochs']
    base_save_path = config['saving']['path']

    train_losses = []   # Initialize lists for tracking losses
    val_losses = []
    val_accuracies = []
    num_inputs = num_inputs = config['model']['num_inputs']  # This should match the num_inputs used during model initialization.
    batch_size = config["training"]["batch_size"]
    time_steps = config["window"]["size"]

    dummy_input = torch.randn(batch_size, time_steps, num_inputs).to(device)

    patience = config["early_stopping"]["patience"]
    delta = config["early_stopping"]["delta"]
    early_stopping = EarlyStopping(
            patience=patience,
            delta=delta,
            verbose=True,
            base_save_path=base_save_path,
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
        val_accuracies.append(val_accuracy)

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
        early_stopping.step(epoch,epoch_val_loss, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    save_model(model,
           base_save_path=base_save_path,
           epoch=epoch,
           dummy_input=dummy_input,
           device=device,
           verbose=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.tight_layout()

    # Save the plot to the current WandB run directory.
    #val_acc_save_path = os.path.join(wandb.run.dir, "val_accuracy.png")
    #plt.savefig(val_acc_save_path)
    #print(f"Validation accuracy plot saved to {val_acc_save_path}")

    # Optionally, log the image to WandB.
    wandb.log({"final_val_accuracy_plot": wandb.Image(plt.gcf())})
    plt.close()