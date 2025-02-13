from networks import TCN, RCNN

def get_model(config):
    """
    Construct the model using parameters from the configuration.
    The config should include 'model.type', which can be either 'TCN' or 'RCNN'.

    For TCN:
      - model.num_inputs: Number of input channels.
      - model.num_channels: List with the number of channels per CNN layer.
      - model.kernel_size: (Optional) Kernel size for convolutions.
      - model.dropout: (Optional) Dropout rate.
      - model.num_classes: Number of output classes.

    For RCNN:
      - model.num_inputs: Number of input channels.
      - model.conv_channels: Number of channels for the convolutional layer.
      - model.kernel_size: (Optional) Kernel size for the convolution.
      - model.dropout: (Optional) Dropout rate.
      - model.rnn_hidden_size: Hidden size for the GRU.
      - model.num_rnn_layers: Number of stacked GRU layers.
      - model.num_classes: Number of output classes.
    """
    model_type = config['model'].get('type', 'TCN')

    if model_type == 'TCN':
        num_inputs = config['model']['num_inputs']
        num_channels = config['model']['num_channels']
        kernel_size = config['model'].get('kernel_size', 2)
        dropout = config['model'].get('dropout', 0.2)
        num_classes = config['model']['num_classes']
        model = TCN(num_inputs, num_channels, kernel_size,dropout, num_classes)
    elif model_type == 'RCNN':
        num_inputs = config['model']['num_inputs']
        conv_channels = config['model']['conv_channels']
        kernel_size = config['model'].get('kernel_size', 3)
        dropout = config['model'].get('dropout', 0.2)
        rnn_hidden_size = config['model'].get('rnn_hidden_size', 64)
        num_rnn_layers = config['model'].get('num_rnn_layers', 1)
        num_classes = config['model']['num_classes']

        model = RCNN(num_inputs, conv_channels, kernel_size, dropout,
                     rnn_hidden_size, num_rnn_layers, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model