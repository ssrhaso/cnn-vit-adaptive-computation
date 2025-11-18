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

# RESULTS STORAGE (EVAL)
predictions = []
confidences = []
true_labels = []


# 1 . DWONALOD AND LOAD CIFAR10 DATASET
print("LOADING CIFAR10 DATASET")
testset = CIFAR10(root='./data', train=False, download=True)       # Download and load the test set
print(f"Test set size: {len(testset)}\n")

# 2. LOAD PRE-TRAINED VIT MODEL 
print("LOADING PRE-TRAINED VIT MODEL")
model_name = r"results/checkpoints/vit_finetuned_V2"
vit_model = ViTForImageClassification.from_pretrained(model_name)
vit_processor = ViTImageProcessor.from_pretrained(model_name)
vit_model.eval()  
print("Pre-trained ViT model loaded successfully\n")

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
    image, true_label = testset[i]                                              # Get an image from the test set
    inputs = vit_processor(images = image, return_tensors = "pt")  # Preprocess the image
    
    start_time = time.time()                                        # Start time
    with torch.no_grad():                                           # Disable gradient calculation for inference
        outputs = vit_model(**inputs)                               # Forward pass through the model    
    end_time = time.time()                                          # End time
    
    latency_ms = (end_time - start_time) * 1000                     # Calculate latency in milliseconds
    latencies.append(latency_ms)                                    # Store latency
    
    # STORE EVAL METRICS FOR EACH IMAGE
    logits = outputs.logits                                         # Get the logits (raw model outputs before softmax)
    
    probs = torch.nn.functional.softmax(logits, dim = 1 )           # PROBABILIY- Convert logits to probabilities using softmax  
    pred = logits.argmax(dim = 1).item()                            # PREDICTION- Get the index of the class with the highest logit score
    conf = probs[0][pred].item()                                    # CONFIDENCE- Get the confidence of the predicted class
    
    # STORE EVAL METRICS
    predictions.append(pred)                                        # PREDICTIONS
    confidences.append(probs[0][pred].item())                       # CONFIDENCES
    true_labels.append(true_label)                                       # TRUE LABELS
    
    
    if (i + 1) % 10 == 0:                                           # Print progress every 10 images
        print(f"Processed {i + 1}/{num_test_images} images")
    

# STATISTICS

# LATENCY
mean_latency = np.mean(latencies)
std_latency = np.std(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)
print(f"\nLatency over {num_test_images} images:")
print("-"*50 + "\n")
print(f"{'Mean Latency :':<25} {mean_latency:.2f} ms")
print(f"{'Std Dev Latency :':<25} {std_latency:.2f} ms")
print(f"{'Min Latency :':<25} {min_latency:.2f} ms")
print(f"{'Max Latency :':<25} {max_latency:.2f} ms")
print("-"*50 + "\n")

# EVAL
predictions = np.array(predictions)
confidences = np.array(confidences)
true_labels = np.array(true_labels)
mean_conf = confidences.mean()
median_conf = np.median(confidences)
std_conf = confidences.std()

accuracy = np.mean(predictions == true_labels).mean()
print(f"\nEvaluation over {num_test_images} images:")
print("-"*50 + "\n")
print(f"{'Accuracy :':<25} {accuracy * 100:.2f} %")
print(f"{'Mean Confidence :':<25} {mean_conf * 100:.2f} %")
print(f"{'Median Confidence :':<25} {median_conf * 100:.2f} %")
print(f"{'Std Dev Confidence :':<25} {std_conf * 100:.2f} %")

# HIGH CONF PREDICTION
high_conf_mask = confidences >= 0.9
if high_conf_mask.sum() > 0:
    high_conf_accuracy = (predictions[high_conf_mask] == true_labels[high_conf_mask]).mean()
    frac_high_conf = high_conf_mask.sum() / num_test_images
    print("-"*50 + "\n")
    print(f"{'High Conf (>=90%) Accuracy :':<25} {high_conf_accuracy * 100:.2f} %")
    print(f"{'Fraction of High Conf Predictions :':<25} {frac_high_conf * 100:.2f} %")

# MEDIUM CONF PREDICTION
med_conf_mask = (confidences > 0.7) & (confidences <= 0.9)
if med_conf_mask.sum() > 0:
    med_conf_accuracy = (predictions[med_conf_mask] == true_labels[med_conf_mask]).mean()
    frac_med_conf = med_conf_mask.sum() / num_test_images
    print("-"*50 + "\n")
    print(f"{'Med Conf (70-90%) Accuracy :':<25} {med_conf_accuracy * 100:.2f} %")
    print(f"{'Fraction of Med Conf Predictions :':<25} {frac_med_conf * 100:.2f} %")
    
# LOW CONF PREDICTION
low_conf_mask = confidences <= 0.7
if low_conf_mask.sum() > 0:
    low_conf_accuracy = (predictions[low_conf_mask] == true_labels[low_conf_mask]).mean()
    frac_low_conf = low_conf_mask.sum() / num_test_images
    print("-"*50 + "\n")
    print(f"{'Low Conf (<=70%) Accuracy :':<25} {low_conf_accuracy * 100:.2f} %")
    print(f"{'Fraction of Low Conf Predictions :':<25} {frac_low_conf * 100:.2f} %")

    









# 5. RESULTS SAVING
print("\nSAVING RESULTS TO 'vit_test_log.txt'")
results_file = "results/logs/vit_test_log.txt"
with open(results_file, "w") as f:
    
    f.write("ViT Latency on CIFAR10 Test Set (CPU)\n")
    f.write(f"CPU: Intel(R) Core(TM) Ultra 7-155H CPU , 16 CORE @ 3.80MHz\n")
    f.write(f"ViT model :{model_name}\n")
    f.write(f"Number of Test Images: {num_test_images}\n")
    f.write("-"*50 + "\n")
    f.write("LATENCY STATISTICS\n")
    f.write("-"*50 + "\n")
    f.write(f"{'-------------------------':<25}\n")
    f.write(f"{'Mean Latency :':<25} {mean_latency:.2f} ms\n")
    f.write(f"{'Std Dev Latency :':<25} {std_latency:.2f} ms\n")
    f.write(f"{'Min Latency :':<25} {min_latency:.2f} ms\n")
    f.write(f"{'Max Latency :':<25} {max_latency:.2f} ms\n")
    f.write(f"{'-------------------------':<25}\n")
    
    
    f.write("\nViT Evaluation on CIFAR10 Test Set (CPU)\n")
    f.write("-"*50 + "\n")
    f.write("ACCURACY AND CONFIDENCE\n")
    f.write("-"*50 + "\n")   
    f.write(f"{'Accuracy :':<25} {accuracy * 100:.2f} %\n")
    f.write(f"{'Mean Confidence :':<25} {mean_conf * 100:.2f} %\n")
    f.write(f"{'Median Confidence :':<25} {median_conf * 100:.2f} %\n")
    f.write(f"{'Std Dev Confidence :':<25} {std_conf * 100:.2f} %\n")
    
    if high_conf_mask.sum() > 0:
        f.write(f"{'High Conf (>=90%) Accuracy :':<25} {high_conf_accuracy * 100:.2f} %\n")
        f.write(f"{'Fraction of High Conf Predictions :':<25} {frac_high_conf * 100:.2f} %\n")
    
    if med_conf_mask.sum() > 0:
        f.write(f"{'Med Conf (70-90%) Accuracy :':<25} {med_conf_accuracy * 100:.2f} %\n")
        f.write(f"{'Fraction of Med Conf Predictions :':<25} {frac_med_conf * 100:.2f} %\n")
    
    if low_conf_mask.sum() > 0:
        f.write(f"{'Low Conf (<=70%) Accuracy :':<25} {low_conf_accuracy * 100:.2f} %\n")
        f.write(f"{'Fraction of Low Conf Predictions :':<25} {frac_low_conf * 100:.2f} %\n")
        
    
    
print("RESULTS SAVED")
