import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """
    Crops (or "chomps") the last 'chomp_size' elements from the time dimension.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size]
        return x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        A single temporal block for the TCN.
        The 'padding' is chosen such that the convolution is causal.
        The extra timesteps added by the convolution are removed with a chomp.
        Batch normalization is applied after each convolution.
        """
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

        # Use a 1x1 convolution if the number of channels does not match for the residual connection.
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        # First layer: conv1 -> chomp -> BatchNorm -> ReLU -> Dropout.
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second layer: conv2 -> chomp -> BatchNorm -> ReLU -> Dropout.
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection: if shapes don't match, apply downsample.
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, num_classes=2):
        """
        num_inputs: Number of input channels.
        num_channels: List containing the number of channels for each convolutional layer.
        num_classes: Number of output classes.
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                          stride=1, dilation=dilation_size, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        # Global average pooling over the temporal dimension.
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x shape: [batch, sequence_length, features]
        # Permute to [batch, features, sequence_length] as required by Conv1d layers.
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