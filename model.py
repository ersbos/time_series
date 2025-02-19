from networks import TCN, RCNN, TimeSeriesCNN, LSTMNet, HybridTCN_LSTM

def get_model(config):
    """
    Construct the model using parameters from the configuration.
    The config should include 'model.type', which can be 'TCN', 'RCNN', or 'TS_CNN'.
    
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
      - model.num_classes: Number of output classes
    """
    model_type = config['model'].get('type', 'TCN')

    if model_type == 'TCN':
        num_inputs = config['model']['num_inputs']
        num_channels = config['model']['num_channels']
        kernel_size = config['model'].get('kernel_size', 2)
        dropout = config['model'].get('dropout', 0.2)
        num_classes = config['model']['num_classes']
        dilation_base = config['model']['dilation_base']
        model = TCN(num_inputs, num_channels, kernel_size, dilation_base, dropout, num_classes)
    elif model_type== "HybridTCN_LSTM":
        num_inputs = config['model']['num_inputs']
        num_channels = config['model']['num_channels']
        kernel_size = config['model'].get('kernel_size', 2)
        dropout = config['model'].get('dropout', 0.2)
        num_classes = config['model']['num_classes']
        lstm_hidden_size= config['model']['lstm_hidden_size']
        lstm_num_layers= config['model']['lstm_num_layers']
        dilation_base = config['model']['dilation_base']
        model = HybridTCN_LSTM(num_inputs,num_channels,kernel_size,dropout,num_classes,
                               lstm_hidden_size,lstm_num_layers,dilation_base)
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
    elif model_type == 'TS_CNN':
        # Our new time series CNN.
        num_inputs = config['model']['num_inputs']
        num_classes = config['model']['num_classes']
        dropout = config['model'].get('dropout', 0.2)
        model = TimeSeriesCNN(num_inputs, num_classes,dropout)
    elif model_type == 'LSTM':
        num_inputs = config['model']['num_inputs']
        hidden_size = config['model'].get('hidden_size', 64)
        num_layers = config['model'].get('num_layers', 1)
        dropout = config['model'].get('dropout', 0.2)
        bidirectional = config['model'].get('bidirectional', True)
        num_classes = config['model']['num_classes']
        model = LSTMNet(num_inputs, hidden_size, num_layers, num_classes, dropout, bidirectional)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model