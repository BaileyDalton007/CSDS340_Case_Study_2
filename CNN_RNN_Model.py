import torch.nn as nn

class CNN_RNN_Model(nn.Module):
    def __init__(self, num_classes=4, input_channels=2, feature_length=60, rnn_hidden_size=128, num_rnn_layers=2):
        super(CNN_RNN_Model, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # compute the output size after CNN layers
        # adjust this if CNN structure changes
        cnn_output_size = 32 * (feature_length // 4)

        # RNN for temporal dependency learning
        self.rnn = nn.LSTM(input_size=cnn_output_size, 
                           hidden_size=rnn_hidden_size, 
                           num_layers=num_rnn_layers, 
                           batch_first=True)

        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        # input shape: (batch_size, seq_length, input_channels, feature_length)
        batch_size, seq_length, channels, feature_length = x.size()

        # merge batch and sequence dimensions for CNN processing
        # shape: (batch_size * seq_length, channels, feature_length)
        x = x.view(-1, channels, feature_length)
        
        x = self.cnn(x)  # shape: (batch_size * seq_length, 32, feature_length // 4)
        x = x.view(batch_size, seq_length, -1)  # reshape back: (batch_size, seq_length, cnn_output_size)

        rnn_out, _ = self.rnn(x)  # shape: (batch_size, seq_length, rnn_hidden_size)

        final_output = rnn_out[:, -1, :]  # shape: (batch_size, rnn_hidden_size)

        out = self.fc(final_output)  # shape: (batch_size, num_classes)

        return out