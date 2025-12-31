import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from PIL import Image

# Function to display a batch of images
def show_batch(loader, class_names, num_images=8):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean  # Reverse normalization
    images = torch.clamp(images, 0, 1)  # Ensure values are in [0, 1]
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = images[i].permute(1, 2, 0).numpy()  # CHW -> HWC
            ax.imshow(img)
            ax.set_title(f"Label: {class_names[labels[i]]}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_batch.png')
    plt.show()
    print("\nSample batch saved as 'sample_batch.png'")

# Wrapper to apply different transforms to validation set
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        if self.transform:
            # Get original image path and label
            original_idx = self.subset.indices[index]
            img_path, label = self.subset.dataset.samples[original_idx]
            # Load and transform image
            x = Image.open(img_path).convert('RGB')
            x = self.transform(x)
            return x, label
        else:
            return self.subset[index]
    
    def __len__(self):
        return len(self.subset)


# Define CNN Model
class ASL_CNN(nn.Module):
    def __init__(self, num_classes):
        super(ASL_CNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # Conv Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Conv Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Conv Block 4: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),                    # Dropout to prevent overfitting
            nn.Linear(256 * 4 * 4, 512),        # Flatten and connect to 512 neurons
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)         # Output layer
        )
        
    def forward(self, x):
        x = self.conv_layers(x)              # Pass through conv layers
        x = x.view(x.size(0), -1)            # Flatten: (batch, 256, 4, 4) -> (batch, 4096)
        x = self.fc_layers(x)                # Pass through fully connected layers
        return x
    
# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(loader)}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# Validation function
def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradient calculation needed
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    IMAGE_SIZE = 200

    # Create directory for saving models
    os.makedirs('models', exist_ok=True)

    # Training data transformations (with augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Validation data transformations (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Load only training dataset
    TRAIN_PATH = 'C:\\Users\\sanka\\Documentos\\Github Desktop\\SilentVoice\\data\\Datasets\\asl_alphabet_train\\asl_alphabet_train'



    print(f"\nChecking path...")
    print(f"Train path exists: {os.path.exists(TRAIN_PATH)}")

    # Load the full training dataset
    full_train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transforms)
    # Split into train (80%) and validation (20%)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Apply validation transforms to validation set
    val_dataset = TransformDataset(val_dataset, transform=val_transforms)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0  # Changed to 0 for Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0  # Changed to 0 for Windows compatibility
    )

    # Print dataset information
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f"\nDataset Info:")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Total samples: {len(full_train_dataset)}")

    # Display sample batch
    print("\nDisplaying sample batch...")
    show_batch(train_loader, class_names)
    
    print("\n✓ Data loading successful! Ready to define the model.")

    print("\n" + "="*50)
    print("Creating CNN Model")
    print("="*50)
    
    model = ASL_CNN(num_classes).to(device)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ Model created successfully!")

    # Define loss function and optimizer
    print("\n" + "="*50)
    print("Setting up Training")
    print("="*50)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=3)
    
    print(f"Loss Function: CrossEntropyLoss")
    print(f"Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"Scheduler: ReduceLROnPlateau (reduces LR if validation loss plateaus)")
    print("\n✓ Training setup complete!")

print("\n" + "="*50)
print("Starting Training")
print("="*50)

best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

    # Train
    train_loss, train_acc = train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # Validate
    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion,
        device
    )

    # Step scheduler on validation loss
    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_asl_cnn.pt")
        print(f"✓ Saved new best model (Val Acc: {best_val_acc:.2f}%)")

print("\nTraining complete!")
