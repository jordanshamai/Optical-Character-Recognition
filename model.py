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
