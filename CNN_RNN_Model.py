import torch.nn as nn

class CNN_RNN_Model(nn.Module):
    def __init__(self, num_classes=4, input_channels=6, feature_length=60, rnn_hidden_size=128, num_rnn_layers=4):
        super(CNN_RNN_Model, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),  # Conv layer
            nn.ReLU(),  # Activation
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsampling
            nn.Dropout(p=0.5),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Another Conv layer
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # Another Conv layer
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),

        )
        
        # Compute the output size after CNN layers
        cnn_output_size = 448 #32 * (feature_length // 4)  # Adjust this if CNN structure changes

        # RNN for temporal dependency learning
        self.rnn = nn.LSTM(input_size=cnn_output_size, 
                           hidden_size=rnn_hidden_size, 
                           num_layers=num_rnn_layers, 
                           batch_first=True, dropout=0.3)

        # Fully connected layer for classification
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, seq_length, input_channels, feature_length)
        batch_size, seq_length, channels, feature_length = x.size()

        # Merge batch and sequence dimensions for CNN processing
        x = x.view(-1, channels, feature_length)  # Shape: (batch_size * seq_length, channels, feature_length)
        
        # Pass through CNN
        x = self.cnn(x)  # Shape: (batch_size * seq_length, 32, feature_length // 4)
        x = x.view(batch_size, seq_length, -1)  # Reshape back: (batch_size, seq_length, cnn_output_size)

        # Pass through RNN
        rnn_out, _ = self.rnn(x)  # Shape: (batch_size, seq_length, rnn_hidden_size)

        # Use the final output of the RNN for classification
        final_output = rnn_out[:, -1, :]  # Shape: (batch_size, rnn_hidden_size)

        # Pass through the fully connected layer
        out = self.fc(final_output)  # Shape: (batch_size, num_classes)

        return out

