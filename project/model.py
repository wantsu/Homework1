import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# parameters
in_feature = 300
num_hidden_1 = 512
num_hidden_2 = 256
out_feature = 2


## Models
class MLP(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(MLP, self).__init__()
        self.Net = nn.Sequential(
            nn.Linear(in_feature, num_hidden_1),
            nn.BatchNorm1d(num_hidden_1),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.BatchNorm1d(num_hidden_2),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(num_hidden_2, out_feature)
        )

    def forward(self, input):
        logits = self.Net(input)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class CNN(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(CNN, self).__init__()
        self.Net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),  # 1*(128 - 1) - 128 + 2  = 1
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),  # 2x(64-1) - 128 + 2 = 0    #300 => 150
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),  # 2x(25-1) - 50 + 2  = 0   # 150 => 75
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),  # #75 => 37
            nn.LeakyReLU(0.1)

        )
        # self.hidden = nn.Linear(37*16, num_hidden_1)
        self.outlayer = nn.Linear(37 * 32, out_feature)

    def forward(self, input):
        input = input.unsqueeze(1)
        out = self.Net(input)
        logits = self.outlayer(out.view(-1, 37 * 32))
        probas = F.softmax(logits, dim=1)
        return logits, probas


class RNN(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(in_feature, num_hidden_1, num_layers=2, bidirectional=True)  # 512   128
        self.mlp = nn.Sequential(
            nn.Linear(num_hidden_1 * 2, num_hidden_2),
            nn.BatchNorm1d(num_hidden_2),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(num_hidden_2, out_feature)
        )

    def forward(self, input):
        input = input.unsqueeze(2)
        input = input.permute(2, 0, 1)  # batch seq_len size=> seq_len batch_size size
        out, hidden = self.rnn(input)  # seq_len batch_size => seq_len batch_size embedding_size
        logits = self.mlp(out.view(-1, num_hidden_1 * 2))  # seq_len, batch, num_directions * hidden_size
        probas = F.softmax(logits, dim=1)
        return logits, probas


