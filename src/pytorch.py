import torch.nn as nn
import torch.nn.functional as F


class DanQ(nn.Sequential):
    def __init__(self, outputs: int = 1):
        super(DanQ, self).__init__()

        self.conv = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.pooling = nn.MaxPool1d(kernel_size=13)
        self.dropout1 = nn.Dropout(0.2)
        self.bidirectional_lstm = nn.LSTM(
            input_size=320,
            hidden_size=16,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(75 * 640, 925)
        self.out = nn.Linear(925, outputs)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pooling(x)
        x = self.dropout1(x)

        x = self.bidirectional_lstm(x)
        x = self.dropout2(x)
        x = self.flatten()

        x = F.relu(self.linear(x))
        x = F.sigmoid(self.out(x))
        return x
