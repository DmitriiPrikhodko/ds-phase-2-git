class BigCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flat = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,  # (w - k +2*p) / s + 1
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # на выходе 32х32
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,  # (w - k +2*p) / s + 1
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # на выходе 32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # на выходе 16x16
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # (w - k +2*p) / s + 1
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # на выходе 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,  # (w - k +2*p) / s + 1
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # на выходе 16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # на выходе 8x8
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,  # (w - k +2*p) / s + 1
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # на выходе 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,  # (w - k +2*p) / s + 1
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # на выходе 8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # на выходе 4x4
        )
        self.fc1 = nn.Linear(4 * 4 * 256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        x_1 = self.fc1(self.flat(out))
        x_2 = self.fc2(F.relu(x_1))
        x_3 = self.fc3(F.relu(x_2))
        x_final = x_3
        return x_final
