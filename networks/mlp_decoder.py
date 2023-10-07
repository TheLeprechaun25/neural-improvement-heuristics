import torch
from torch import nn
import torch.nn.functional as F


class MLP_Decoder(nn.Module):
    def __init__(self):
        super(MLP_Decoder, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, e, mask):
        e = F.relu(self.fc1(e))
        e = F.relu(self.fc2(e))
        e = self.fc3(e)

        e = torch.reshape(e.clone(), (e.shape[0], e.shape[1] * e.shape[2]))
        e[:, mask[0, :]] = -1e10

        log_p = F.log_softmax(e, dim=-1)

        return log_p
