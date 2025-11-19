import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import time
import os

# Import your CNN model
import sys
sys.path.insert(0, '../')
from models.cnn_test import CIFAR10CNN

# CIFAR-10 classes
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

print("="*60)
print("TRAINING CNN ON CIFAR-10")
print("="*60)

# 1. Prepare CIFAR-10 dataset
print("\n1. Preparing CIFAR-10 dataset...")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))
])

trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

print(f"✓ Training set: {len(trainset)} images")
print(f"✓ Test set: {len(testset)} images")

# 2. Setup training
print("\n2. Setting up training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")

model = CIFAR10CNN(num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = 15
print(f"✓ Epochs: {num_epochs}")
print(f"✓ Learning rate: 0.001")

# Create checkpoint directory
os.makedirs("results/checkpoints", exist_ok=True)

# 3. Training loop
print("\n3. Starting training...")
print("="*60)

best_test_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-"*60)
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / 100
            accuracy = 100 * correct / total
            print(f"Batch [{batch_idx+1}/{len(trainloader)}] "
                  f"Loss: {avg_loss:.3f} | Acc: {accuracy:.2f}%")
            running_loss = 0.0
    
    epoch_time = time.time() - start_time
    train_accuracy = 100 * correct / total
    print(f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Save best model
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        torch.save(model.state_dict(), "results/checkpoints/cnn_best.pt")
        print(f"✓ Best model saved (Acc: {best_test_acc:.2f}%)")
    
    # Step learning rate scheduler
    scheduler.step()
    print("="*60)

# 4. Save final model
print("\n4. Saving final model...")
torch.save(model.state_dict(), "results/checkpoints/cnn_final.pt")
print("✓ Model saved to: results/checkpoints/cnn_final.pt")
print(f"✓ Best test accuracy achieved: {best_test_acc:.2f}%")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
