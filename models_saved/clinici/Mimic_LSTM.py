import torch
import torch.nn as nn

class Mimic_LSTM(nn.Module):
    def __init__(self, seq_input_size=14, static_input_size=7, hidden_size=64):
        super(Mimic_LSTM, self).__init__()
        self.lstm = nn.LSTM(seq_input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + static_input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1) 
        )

    def forward(self, x_seq, lengths, x_static):
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_x)
        lstm_out = hn[-1] 
        combined = torch.cat((lstm_out, x_static), dim=1)
        return self.classifier(combined).squeeze(1)
