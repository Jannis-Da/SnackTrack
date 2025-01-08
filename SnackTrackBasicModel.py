import torch.nn as nn

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(8, 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride=2, padding=0)
        self.activation2 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4*4*64, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.final_linear = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        #x = self.dropout1(x)   # Dropout not used in final model, because of a too small dataset
        x = self.linear2(x)
        #x = self.dropout2(x)
        x = self.final_linear(x)
        return x