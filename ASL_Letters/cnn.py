import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, '..', 'data', 'Datasets', 'asl_alphabet_train')
TEST_DIR = os.path.join(BASE_DIR, '..', 'data', 'Datasets', 'asl_alphabet_test')
MODEL_PATH = os.path.join(BASE_DIR, 'best_asl_model.pth')


# CNN Model
class ASLCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# Custom dataset for test images (unlabeled)
class ASLTestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])


# Inference function for a single image
def predict_image(image_path, model, transform, classes, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return classes[predicted.item()], confidence.item() * 100


# Required on Windows so DataLoader workers don't re-run top-level code
if __name__ == '__main__':
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load training data and split into train/val
    full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

    # Limit to first 25 images per class
    from collections import defaultdict
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_train_dataset.samples):
        class_to_indices[label].append(idx)
    limited_indices = []
    for label in sorted(class_to_indices):
        limited_indices.extend(class_to_indices[label][:25])

    # Stratified split on limited indices
    targets = [full_train_dataset.targets[i] for i in limited_indices]
    train_idx_local, val_idx_local = train_test_split(
        range(len(limited_indices)),
        test_size=0.15,
        stratify=targets,
        random_state=42
    )
    train_idx = [limited_indices[i] for i in train_idx_local]
    val_idx = [limited_indices[i] for i in val_idx_local]

    # Train subset uses augmentation transforms
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)

    # Validation subset uses clean val transforms
    val_dataset_full = datasets.ImageFolder(TRAIN_DIR, transform=val_transform)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_idx)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Test data (flat directory, one image per class)
    test_dataset = ASLTestDataset(TEST_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f'Training samples:   {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Test samples:       {len(test_dataset)}')
    print(f'Number of classes:  {len(full_train_dataset.classes)}')
    print(f'Classes: {full_train_dataset.classes}')

    # Model, loss, optimiser
    model = ASLCNN(num_classes=len(full_train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    # Training loop
    num_epochs = 20
    best_acc = 0.0
    train_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'classes': full_train_dataset.classes
            }, MODEL_PATH)
            print(f'  *** New best model saved! ***')
        print()

    print(f'Training complete! Best validation accuracy: {best_acc:.2f}%')

    # Plot training curves
    curves_path = os.path.join(BASE_DIR, 'training_curves.png')
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(curves_path)
    print(f'Training curves saved to {curves_path}')

    # Predictions on test set
    print('\nMaking predictions on test set...')
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

            for i, filename in enumerate(filenames):
                pred_class = full_train_dataset.classes[predicted[i].item()]
                conf = confidences[i].item() * 100
                predictions.append((filename, pred_class, conf))

    preds_path = os.path.join(BASE_DIR, 'test_predictions.txt')
    with open(preds_path, 'w') as f:
        f.write('Filename,Predicted_Class,Confidence\n')
        for filename, pred_class, conf in predictions:
            f.write(f'{filename},{pred_class},{conf:.2f}\n')

    print(f'Predictions saved to {preds_path}')
    print('\nSample predictions:')
    for filename, pred_class, conf in predictions[:10]:
        print(f'  {filename}: {pred_class} ({conf:.2f}%)')

    # Example usage:
    # letter, conf = predict_image('test_image.jpg', model, val_transform,
    #                              full_train_dataset.classes, device)
    # print(f'Predicted: {letter} (Confidence: {conf:.2f}%)')
