import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import time

# CIFAR-10 classes
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

print("="*60)
print("FINE-TUNING ViT ON CIFAR-10")
print("="*60)

# 1. Load pretrained ViT and modify for CIFAR-10
print("\n1. Loading pretrained ViT...")
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=10,  # CIFAR-10 has 10 classes
    ignore_mismatched_sizes=True  # Allow head replacement
)
processor = ViTImageProcessor.from_pretrained(model_name)
print("✓ ViT loaded with 10-class classification head")

# 2. Prepare CIFAR-10 dataset
print("\n2. Preparing CIFAR-10 dataset...")
def transform_images(examples):
    """Transform CIFAR-10 images to ViT input format"""
    images = [img.resize((224, 224)) for img in examples]  # Resize to 224x224
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values']

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

print(f"✓ Training set: {len(trainset)} images")
print(f"✓ Test set: {len(testset)} images")

# 3. Training setup
print("\n3. Setting up training...")
device = torch.device("cpu")  # Use CPU (you don't have GPU)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 2e-5)  # Small learning rate for fine-tuning

num_epochs = 2  # 2 epochs should be enough (more takes too long on CPU)
print(f" Device: {device}")
print(f" Epochs: {num_epochs}")
print(f" Learning rate: 2e-5")

# 4. Training loop
print("\n4. Starting training...")
print("="*60)

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
        loss = criterion(outputs.logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 200 batches
        if (batch_idx + 1) % 200 == 0:
            avg_loss = running_loss / 200
            accuracy = 100 * correct / total
            print(f"Batch [{batch_idx+1}/{len(trainloader)}] "
                  f"Loss: {avg_loss:.3f} | Acc: {accuracy:.2f}%")
            running_loss = 0.0
    
    epoch_time = time.time() - start_time
    train_accuracy = 100 * correct / total
    print(f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    
    # Evaluate on test set after each epoch
    print("\nEvaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("="*60)

# 5. Save fine-tuned model
print("\n5. Saving fine-tuned model...")
model.save_pretrained("results/checkpoints/vit_finetuned_cifar10")
processor.save_pretrained("results/checkpoints/vit_finetuned_cifar10")
print(" Model saved to: results/checkpoints/vit_finetuned_cifar10")

print("\n" + "="*60)
print("FINE-TUNING COMPLETE!")
print("="*60)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
