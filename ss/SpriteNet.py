import torch, torch.nn as nn

class SpriteNet(nn.Module):
    def __init__(self, inChannels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inChannels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1) # 1x1 for color mixing
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)) # RGB in [0,1]
