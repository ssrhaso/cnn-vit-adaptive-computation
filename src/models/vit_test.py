import torch
import torchvision
import numpy as np
import random
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10

from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import time


CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 

# 1 . DWONALOD AND LOAD CIFAR10 DATASET
print("LOADING CIFAR10 DATASET")
testset = CIFAR10(root='./data', train=False, download=True)       # Download and load the test set
print(f"Test set size: {len(testset)}\n")

# 2. LOAD PRE-TRAINED VIT MODEL 
print("LOADING PRE-TRAINED VIT MODEL")
model_name = "google/vit-base-patch16-224"
vit_model = ViTForImageClassification.from_pretrained(model_name)
vit_processor = ViTImageProcessor.from_pretrained(model_name)
vit_model.eval()  
print("Pre-trained ViT model loaded successfully.\n")

# 3. TEST ON 1 IMAGE
print("TESTING ON 1 IMAGE FROM CIFAR10 TEST SET")
image, label = testset[42]                                          # Get an image and its label from the test set 
true_class = CIFAR10_CLASSES[label]                                 # Get the true class (label) 

# PREPROCESS
inputs = vit_processor(images = image, return_tensors = "pt")                           

# INFERENCE
with torch.no_grad():                                               # Disable gradient calculation for inference (saves memory and computations)
    outputs = vit_model(**inputs)                                   # Forward pass through the model
    logits = outputs.logits                                         # Get the logits (raw model outputs before softmax)
    predicted_class_idx = logits.argmax(-1).item()                  # Get the index of the class with the highest logit score 

print(f"True class: {true_class}")
print(f"Predicted class: (class index {predicted_class_idx})")      
print(f"Logits Shape: {logits.shape}")
print(f"Sample Logits: {logits[0][:10]}")                           # Print first 10 logits 






# 4. MEASURE LATENCY (CPU) ON 100 IMAGES
print("\nMEASURING INFERENCE LATENCY ON 100 IMAGES (CPU)")
num_test_images = 100
latencies = []

for i in range(num_test_images):
    image, _ = testset[i]                                           # Get an image from the test set
    inputs = vit_processor(images = image, return_tensors = "pt")  # Preprocess the image
    
    start_time = time.time()                                        # Start time
    with torch.no_grad():                                           # Disable gradient calculation for inference
        outputs = vit_model(**inputs)                               # Forward pass through the model    
    end_time = time.time()                                          # End time
    
    latency_ms = (end_time - start_time) * 1000                     # Calculate latency in milliseconds
    latencies.append(latency_ms)                                    # Store latency
    
    if (i + 1) % 10 == 0:                                           # Print progress every 10 images
        print(f"Processed {i + 1}/{num_test_images} images")
    

# STATISTICS
mean_latency = np.mean(latencies)
std_latency = np.std(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)

print(f"\nLatency over {num_test_images} images:")
print(f"{'-------------------------':<25}")
print(f"{'Mean Latency :':<25} {mean_latency:.2f} ms")
print(f"{'Std Dev Latency :':<25} {std_latency:.2f} ms")
print(f"{'Min Latency :':<25} {min_latency:.2f} ms")
print(f"{'Max Latency :':<25} {max_latency:.2f} ms")
print(f"{'-------------------------':<25}")


# 5. RESULTS SAVING
print("\nSAVING RESULTS TO 'vit_test_log.txt'")
results_file = "results/logs/vit_test_log.txt"
with open(results_file, "w") as f:
    
    f.write("ViT Latency on CIFAR10 Test Set (CPU)\n")
    f.write(f"CPU: Intel(R) Core(TM) Ultra 7-155H CPU , 16 CORE @ 3.80MHz\n")
    f.write(f"ViT model :{model_name}\n")
    f.write(f"Number of Test Images: {num_test_images}\n")
    f.write(f"{'-------------------------':<25}\n")
    f.write(f"{'Mean Latency :':<25} {mean_latency:.2f} ms\n")
    f.write(f"{'Std Dev Latency :':<25} {std_latency:.2f} ms\n")
    f.write(f"{'Min Latency :':<25} {min_latency:.2f} ms\n")
    f.write(f"{'Max Latency :':<25} {max_latency:.2f} ms\n")
    f.write(f"{'-------------------------':<25}\n")
    
print("RESULTS SAVED")
