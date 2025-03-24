class OCRCNN(nn.Module):
    def __init__(self):
        super(OCRCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust for 28x28 image size after conv and pooling layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 47)  # 47 classes for the EMNIST Balanced dataset (letters + digits)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output for fully connected layers
        
        x = F.relu(self.fc1(x))     # Fully connected layer 1 with ReLU
        x = F.relu(self.fc2(x))     # Fully connected layer 2 with ReLU
        x = self.fc3(x)             # Output layer (logits, no activation here)
        return x


model = OCRCNN()
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
num_epochs = 100
if torch.cuda.is_available():
    print("CUDA Available. Continuing with 100 epochs...")
else:
    num_epochs = 0

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# Training loop
