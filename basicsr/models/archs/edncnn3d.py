import torch.nn as nn

class EDnCNN3D(nn.Module):
    def __init__(self):
        super(EDnCNN3D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 7), stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 5), stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 1), stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(25*25, 256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        param x: (batch_size, 1, time_steps, height, width)
        '''
        x = self.layers(x)
        x = x.mean(2)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
