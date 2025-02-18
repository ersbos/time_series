import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size]
        return x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 1x1 convolution for residual if needed.
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dilation_base=1.5, dropout=0.2, num_classes=10):
        """
        num_inputs: Number of input channels.
        num_channels: List with channels for each temporal block.
        kernel_size: Convolution kernel size (small value suggested for short signals).
        dilation_base: Base for computing dilation factors (smaller than 2 might be more suitable).
        dropout: Dropout rate.
        num_classes: Number of output classes.
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # Using a less aggressive dilation progression
            dilation = int(dilation_base ** i)
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # Adjust padding so that the output length doesn't exceed input length.
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                          stride=1, dilation=dilation, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # Permute input from [batch, time_steps, features] to [batch, features, time_steps]
        x = x.permute(0, 2, 1)
        y = self.network(x)
        # Global average pooling over the temporal dimension.
        y = torch.mean(y, dim=2)
        return self.fc(y)
class RCNN(nn.Module):
    def __init__(self, num_inputs, conv_channels, kernel_size=3, dropout=0.2, rnn_hidden_size=64,
                 num_rnn_layers=1, num_classes=2):
        """
        num_inputs: Number of input channels.
        conv_channels: Number of output channels for the initial convolution.
        kernel_size: Kernel size of the convolution.
        dropout: Dropout rate after the convolution.
        rnn_hidden_size: Number of features in the hidden state of the GRU.
        num_rnn_layers: Number of stacked GRU layers.
        num_classes: Number of output classes.
        """
        super(RCNN, self).__init__()
        # 1D convolution to extract local features
        self.conv = nn.Conv1d(num_inputs, conv_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # GRU (bidirectional) to capture global temporal relationships
        self.rnn = nn.GRU(input_size=conv_channels,
                          hidden_size=rnn_hidden_size,
                          num_layers=num_rnn_layers,
                          batch_first=True,
                          bidirectional=True)
        # Fully connected output layer (2*rnn_hidden_size for bidirectional GRU)
        self.fc = nn.Linear(2 * rnn_hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch, sequence_length, features]
        # Permute to [batch, features, sequence_length] for convolution
        x = x.permute(0, 2, 1)
        conv_features = self.conv(x)
        conv_features = self.relu(conv_features)
        conv_features = self.dropout(conv_features)
        # Permute back to [batch, sequence_length, conv_channels] for the GRU
        conv_features = conv_features.permute(0, 2, 1)
        # Pass features through the GRU
        rnn_out, _ = self.rnn(conv_features)
        # Global average pooling over the time dimension
        out = torch.mean(rnn_out, dim=1)
        logits = self.fc(out)
        return logits


class TimeSeriesCNN(nn.Module):
    def __init__(self, num_inputs, num_classes):
        """
        A deeper end-to-end convolutional classifier for time series.

        Architecture (each block now has two convolutional layers):
          - Block 1: 
              * Conv1d with 64 filters, kernel size=8, padding=4.
              * BatchNorm1d and ReLU.
              * An additional Conv1d with 64 filters, kernel size=8, padding=4, BatchNorm1d, and ReLU.
          - Block 2:
              * Conv1d with 128 filters, kernel size=5, padding=2.
              * BatchNorm1d and ReLU.
              * An additional Conv1d with 128 filters, kernel size=5, padding=2, BatchNorm1d, and ReLU.
          - Block 3:
              * Conv1d with 64 filters, kernel size=3, padding=1.
              * BatchNorm1d and ReLU.
              * An additional Conv1d with 64 filters, kernel size=3, padding=1, BatchNorm1d, and ReLU.
          - Global average pooling over the time dimension.
          - Fully connected layer mapping 64 channels to num_classes.
        """
        super(TimeSeriesCNN, self).__init__()

        # Block 1: two layers with 64 filters, keeping the same kernel and padding.
        self.conv1a = nn.Conv1d(num_inputs, 64, kernel_size=8, padding=4)
        self.bn1a = nn.BatchNorm1d(64)
        self.conv1b = nn.Conv1d(64, 64, kernel_size=8, padding=4)
        self.bn1b = nn.BatchNorm1d(64)

        # Block 2: two layers with 128 filters.
        self.conv2a = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2a = nn.BatchNorm1d(128)
        self.conv2b = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn2b = nn.BatchNorm1d(128)

        # Block 3: two layers with 64 filters.
        self.conv3a = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm1d(64)
        self.conv3b = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm1d(64)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch, time_steps, features]
        # Permute to [batch, features, time_steps] for Conv1d.
        x = x.permute(0, 2, 1)

        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))

        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))

        x = self.global_pool(x)  # x now has shape [batch, 64, 1]
        x = x.squeeze(2)         # shape becomes [batch, 64]
        return self.fc(x)


class LSTMNet(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_layers, num_classes, dropout=0.2, bidirectional=True):
        """
        A simple LSTM-based classifier for time series.

        Architecture:
          - LSTM (or stacked LSTM) reads in the sequence.
          - The last hidden state is used as a representation.
          - A fully connected layer maps the representation to num_classes.

        Args:
          num_inputs (int): Number of input features per time step.
          hidden_size (int): Hidden size of the LSTM.
          num_layers (int): Number of LSTM layers.
          num_classes (int): Number of output classes.
          dropout (float): Dropout rate to use between LSTM layers.
          bidirectional (bool): If True, use a bidirectional LSTM.
        """
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_out_size, num_classes)

    def forward(self, x):
        # x shape: [batch, time_steps, features]
        lstm_out, (hn, cn) = self.lstm(x)
        # Here we use the last layer's final hidden state.
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states.
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_hidden = hn[-1]
        logits = self.fc(last_hidden)
        return logits