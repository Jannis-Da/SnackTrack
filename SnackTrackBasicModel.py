import torch.nn as nn

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Output for 4 classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x