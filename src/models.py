from torch import nn


class ShapPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 32)
        self._non_linearity = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self._non_linearity(x)
        x = self.fc2(x)
        return x
