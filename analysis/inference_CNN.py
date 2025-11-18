import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# RESULTS STORAGE (EVAL)
predictions = []
confidences = []
true_labels = []
latencies = []

# CNN MODEL DEFINITION
class CIFAR10CNN(nn.Module):
    """SIMPLE CNN FOR CIFAR-10 CLASSIFICATION"""
    
    def __init__(self, num_classes: int = 10):
        super(CIFAR10CNN, self).__init__()
        
        # BLOCK 1 (32x32) -> (16x16)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.20)
        
        # BLOCK 2 (16x16) -> (8x8)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.30)
        
        # BLOCK 3 (8x8) -> (4x4)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.40)
        
        # CLASSIFIER
        self.fc = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout_fc = nn.Dropout(0.50)
        self.out = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BLOCK 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # BLOCK 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # BLOCK 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # FLATTEN
        x = x.view(x.size(0), -1)
        
        # CLASSIFIER
        x = F.relu(self.bn_fc(self.fc(x)))
        x = self.dropout_fc(x)
        x = self.out(x)
        
        return x

# 1. LOAD CIFAR10 DATASET
print("LOADING CIFAR10 DATASET")
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])
testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
print(f"Test set size: {len(testset)}\n")

# 2. LOAD TRAINED CNN MODEL
print("LOADING TRAINED CNN MODEL")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CIFAR10CNN(num_classes=10).to(device)
cnn_model.load_state_dict(torch.load('results/checkpoints/cnn_best.pt', map_location=device))
cnn_model.eval()
print(f"Trained CNN model loaded successfully (Device: {device})\n")

# 3. TEST ON 1 IMAGE
print("TESTING ON 1 IMAGE FROM CIFAR10 TEST SET")
image, label = testset[42]
true_class = CIFAR10_CLASSES[label]

# Add batch dimension
image_input = image.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = cnn_model(image_input)
    logits = outputs
    predicted_class_idx = logits.argmax(-1).item()

print(f"True class: {true_class}")
print(f"Predicted class: {CIFAR10_CLASSES[predicted_class_idx] if predicted_class_idx < 10 else predicted_class_idx}")
print(f"Logits Shape: {logits.shape}")
print(f"Sample Logits: {logits[0][:10]}\n")

# 4. MEASURE LATENCY & ACCURACY ON 100 IMAGES
print("MEASURING INFERENCE LATENCY AND ACCURACY ON 100 IMAGES")
num_test_images = 100

for i in range(num_test_images):
    image, label = testset[i]
    image_input = image.unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = cnn_model(image_input)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)
    
    # STORE EVAL METRICS
    logits = outputs
    probs = F.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).item()
    conf = probs[0][pred].item()
    
    predictions.append(pred)
    confidences.append(conf)
    true_labels.append(label)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{num_test_images} images")

# STATISTICS - LATENCY
latencies = np.array(latencies)
mean_latency = np.mean(latencies)
std_latency = np.std(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)

print(f"\nLatency over {num_test_images} images:")
print(f"{'-------------------------':<35}")
print(f"{'Mean Latency :':<35} {mean_latency:.2f} ms")
print(f"{'Std Dev Latency :':<35} {std_latency:.2f} ms")
print(f"{'Min Latency :':<35} {min_latency:.2f} ms")
print(f"{'Max Latency :':<35} {max_latency:.2f} ms")
print(f"{'-------------------------':<35}")

# STATISTICS - EVALUATION
predictions = np.array(predictions)
confidences = np.array(confidences)
true_labels = np.array(true_labels)

accuracy = (predictions == true_labels).mean()
mean_conf = confidences.mean()
median_conf = np.median(confidences)
std_conf = confidences.std()

print(f"\nEvaluation over {num_test_images} images:")
print(f"{'-------------------------':<35}")
print(f"{'Accuracy :':<35} {accuracy * 100:.2f} %")
print(f"{'Mean Confidence :':<35} {mean_conf * 100:.2f} %")
print(f"{'Median Confidence :':<35} {median_conf * 100:.2f} %")
print(f"{'Std Dev Confidence :':<35} {std_conf * 100:.2f} %")

# HIGH CONFIDENCE PREDICTIONS
high_conf_mask = confidences >= 0.9
if high_conf_mask.sum() > 0:
    high_conf_accuracy = (predictions[high_conf_mask] == true_labels[high_conf_mask]).mean()
    frac_high_conf = high_conf_mask.sum() / num_test_images
    print(f"{'High Conf (>=90%) Accuracy :':<35} {high_conf_accuracy * 100:.2f} %")
    print(f"{'Fraction of High Conf Predictions :':<35} {frac_high_conf * 100:.2f} %")

# MEDIUM CONFIDENCE PREDICTIONS
med_conf_mask = (confidences > 0.7) & (confidences < 0.9)
if med_conf_mask.sum() > 0:
    med_conf_accuracy = (predictions[med_conf_mask] == true_labels[med_conf_mask]).mean()
    frac_med_conf = med_conf_mask.sum() / num_test_images
    print(f"{'Med Conf (70-90%) Accuracy :':<35} {med_conf_accuracy * 100:.2f} %")
    print(f"{'Fraction of Med Conf Predictions :':<35} {frac_med_conf * 100:.2f} %")

# LOW CONFIDENCE PREDICTIONS
low_conf_mask = confidences <= 0.7
if low_conf_mask.sum() > 0:
    low_conf_accuracy = (predictions[low_conf_mask] == true_labels[low_conf_mask]).mean()
    frac_low_conf = low_conf_mask.sum() / num_test_images
    print(f"{'Low Conf (<=70%) Accuracy :':<35} {low_conf_accuracy * 100:.2f} %")
    print(f"{'Fraction of Low Conf Predictions :':<35} {frac_low_conf * 100:.2f} %")

# 5. SAVE RESULTS TO LOG FILE
print("\nSAVING RESULTS TO 'cnn_test_log.txt'")
os.makedirs("results/logs", exist_ok=True)
results_file = "results/logs/cnn_test_log.txt"

with open(results_file, "w") as f:
    f.write("CNN Performance on CIFAR-10 Test Set\n")
    f.write(f"Device: {device}\n")
    f.write(f"Model: CIFAR10CNN (~3.2M parameters)\n")
    f.write(f"Number of Test Images: {num_test_images}\n\n")
    
    f.write("-"*50 + "\n")
    f.write("LATENCY STATISTICS\n")
    f.write("-"*50 + "\n")
    f.write(f"{'Mean Latency :':<35} {mean_latency:.2f} ms\n")
    f.write(f"{'Std Dev Latency :':<35} {std_latency:.2f} ms\n")
    f.write(f"{'Min Latency :':<35} {min_latency:.2f} ms\n")
    f.write(f"{'Max Latency :':<35} {max_latency:.2f} ms\n\n")
    
    f.write("-"*50 + "\n")
    f.write("ACCURACY AND CONFIDENCE\n")
    f.write("-"*50 + "\n")
    f.write(f"{'Accuracy :':<35} {accuracy * 100:.2f} %\n")
    f.write(f"{'Mean Confidence :':<35} {mean_conf * 100:.2f} %\n")
    f.write(f"{'Median Confidence :':<35} {median_conf * 100:.2f} %\n")
    f.write(f"{'Std Dev Confidence :':<35} {std_conf * 100:.2f} %\n\n")
    
    if high_conf_mask.sum() > 0:
        f.write(f"{'High Conf (>=90%) Accuracy :':<35} {high_conf_accuracy * 100:.2f} %\n")
        f.write(f"{'Fraction of High Conf Predictions :':<35} {frac_high_conf * 100:.2f} %\n\n")
    
    if med_conf_mask.sum() > 0:
        f.write(f"{'Med Conf (70-90%) Accuracy :':<35} {med_conf_accuracy * 100:.2f} %\n")
        f.write(f"{'Fraction of Med Conf Predictions :':<35} {frac_med_conf * 100:.2f} %\n\n")
    
    if low_conf_mask.sum() > 0:
        f.write(f"{'Low Conf (<=70%) Accuracy :':<35} {low_conf_accuracy * 100:.2f} %\n")
        f.write(f"{'Fraction of Low Conf Predictions :':<35} {frac_low_conf * 100:.2f} %\n")

print("RESULTS SAVED")
