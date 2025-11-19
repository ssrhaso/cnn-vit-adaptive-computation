import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from transformers import ViTForImageClassification, ViTImageProcessor
import time
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.cnn_test import CIFAR10CNN



class AdaptiveClassifier:
    def __init__(
        self,
        cnn_checkpoint_path: str,
        vit_checkpoint_path: str,
        threshold: float = 0.90,
        device: str = 'cpu',
        input_mean = None,
        input_std = None
    ):
        
        self.device = device
        self.threshold = threshold
        
        # LOAD CNN MODEL
        self.cnn = CIFAR10CNN(num_classes=10).to(self.device)  
        input_mean = input_mean if input_mean is not None else (0.4914, 0.4822, 0.4465)
        input_std = input_std if input_std is not None else (0.2470, 0.2435, 0.2616)        
                                     
        
        checkpoint = torch.load(cnn_checkpoint_path, map_location = self.device)    # Load CNN checkpoint (.pt file)
        # Check if the checkpoint contains 'model_state_dict' key
        try:
                
            if 'model_state_dict' in checkpoint:
                self.cnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.cnn.load_state_dict(checkpoint)
                
        except FileNotFoundError as e:
            print(f"CNN model checkpoint not found at {cnn_checkpoint_path} : {e}")
        except RuntimeError as e:
            print(f"Error loading CNN model state dict: {e}")
        self.cnn.eval()
        print("CNN model loaded")
        
        
        # LOAD VIT MODEL
        self.vit = ViTForImageClassification.from_pretrained(vit_checkpoint_path).to(self.device)
        self.vit_processor = ViTImageProcessor.from_pretrained(vit_checkpoint_path)
        self.vit.eval()
        print("ViT model loaded")
        
    
        # SETUP PREPROCESSING
        # CNN (32x32):
        
        self.cnn_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( 
                mean = input_mean,
                std = input_std)
        ])
        
        self._warmup() # WARMUP TO REDUCE INITIAL LATENCY (CACHING)
        
        
    
    # WARMUP METHOD (TO REDUCE INITIAL LATENCY)
    def _warmup(
        self,
        num_runs: int = 5,
        input_sizes : dict = None 
    ):
        """ WARMUP MODELS TO REDUCE INITIAL LATENCY , IMPROVE CONSISTENCY IN INFERENCE LABELLING """
        if input_sizes is None:
            input_sizes = {
                'cnn': (1, 3, 32, 32),
                'vit': (1, 3, 224, 224)
            }
            
        dummy_cnn = torch.randn(*input_sizes['cnn']).to(self.device)
        dummy_vit = torch.randn(*input_sizes['vit']).to(self.device)
        
        with torch.no_grad(): # NO GRADIENTS NEEDED, FASTER
           for _ in range(num_runs):
                _ = self.cnn(dummy_cnn)
                _ = self.vit(pixel_values = dummy_vit).logits
        
        print("Models warmed up")
    
    
    
    
    def predict(
        self,
        image: Image.Image
    ) -> tuple:
        """ CNN + VIT ADAPTIVE INFERENCE , RETURNS PREDICTION, ROUTED MODEL, CONFIDENCE, METADATA """
        
        # METADATA FOR ANALYSIS
        stats = {
            'cnn_latency' : 0.0,
            'vit_latency' : 0.0,
            'total_latency' : 0.0,
            'routed_to' : None,
            'cnn_confidence' : 0.0,
            'cnn_prediction' : None,
        }
        
        # CNN INFERENCE (ALWAYS RUNNING)
    
        cnn_pred, cnn_confidence, cnn_latency = self._predict_cnn(image)
        stats['cnn_latency'] = cnn_latency
        stats['cnn_confidence'] = cnn_confidence
        stats['cnn_prediction'] = cnn_pred
        
        
        # ROUTING DECISION
        
        if cnn_confidence >= self.threshold:
            
            # HIGH CONFIDENCE
            stats['routed_to'] = 'CNN'
            stats['total_latency'] = cnn_latency
            return (
                cnn_pred,
                'CNN',
                cnn_confidence,
                cnn_latency,
                stats
            )
            
        else:
            # LOW CONFIDENCE - ROUTE TO VIT
        
            vit_pred, vit_confidence, vit_latency = self._predict_vit(image)
            
            stats['vit_latency'] = vit_latency
            stats['routed_to'] = 'ViT'
            stats['total_latency'] = cnn_latency + vit_latency
            
            return (
                vit_pred,
                'ViT',
                vit_confidence,
                stats['total_latency'],
                stats
            )
            
            
    # PREDICTION HELPERS
    
    # CNN PREDICTION HELPED
    def _predict_cnn(
        self,
        image: Image.Image
    ):
        """ PREDICTION USING CNN MODEL """
        tensor_cnn = self.cnn_transform(image)      # PIL to Tensor -> [3, 32, 32]
        batched = tensor_cnn.unsqueeze(0)           # BATCH DIMENSION [3, 32, 32] -> [1, 3, 32, 32]
        cnn_input = batched.to(self.device)         # MOVE TO CPU/GPU
        
        # INFERENCE
        start = time.perf_counter()                                             # START TIME (INF 1)
        with torch.no_grad():
            cnn_confidence , cnn_pred = self.cnn.get_confidence(cnn_input)
        cnn_latency = (time.perf_counter() - start) * 1000                      # END TIME (INF 2) -> LATENCY MS
        
        return cnn_pred.item(), cnn_confidence.item(), cnn_latency
  
    # VIT PREDICTION HELPER
    def _predict_vit(
        self,
        image: Image.Image
    ):
        """ PREDICTION USING VIT MODEL """
        vit_input = self.vit_processor(images = image, return_tensors = "pt")       # PREPROCESS FOR VIT
        vit_input = {k: v.to(self.device) for k, v in vit_input.items()}            # MOVE TO CPU/GPU
            
            # INFERENCE
        start = time.perf_counter()                                                 # START TIME (INF 1)
        with torch.no_grad():
            vit_outputs = self.vit(**vit_input)
            vit_logits = vit_outputs.logits
            vit_probs = F.softmax(vit_logits, dim = 1)
            vit_confidence, vit_pred = torch.max(vit_probs, dim = 1)
            vit_latency = (time.perf_counter() - start) * 1000                          # END TIME (INF 2) -> LATENCY MS
            
        return vit_pred.item(), vit_confidence.item(), vit_latency
    
    

# ENTRY (TESTING PURPOSES)
if __name__ == "__main__":
    # TEST ADAPTIVE CLASSIFIER
    
    # LOAD TEST DATASET
    test_dataset = datasets.CIFAR10(
        root = './data',
        train = False,
        download = True,    
    )


    # CREATE ADAPTIVE CLASSIFIER
    adaptive = AdaptiveClassifier(
        cnn_checkpoint_path = './results/checkpoints/cnn_best.pt',
        vit_checkpoint_path = './results/checkpoints/vit_finetuned_V2',
        threshold = 0.90,
        device = 'cpu'
    )
    
    # TEST ON IMAGES
    class_names = test_dataset.classes
        
    print("="*70)
    print(f"\nTESTING ADAPTIVE CLASSIFIER")
    print("="*70)

    # Configuration
    x = 10000  # Number of images to test (change to 5, 100, or 1000)

    # Tracking variables
    total_latency = 0.0
    cnn_count = 0
    vit_count = 0
    correct = 0
    cnn_latencies = []
    vit_latencies = []

    # Run predictions
    print(f"\nProcessing {x} test images...")
    for i in range(x):
        img, true_label = test_dataset[i]
        pred, model_used, confidence, latency, stats = adaptive.predict(img)
        
        # Track metrics
        total_latency += latency
        correct += (pred == true_label)
        cnn_count += (stats['routed_to'] == 'CNN')
        vit_count += (stats['routed_to'] == 'ViT')
        cnn_latencies.append(stats['cnn_latency'])
        if stats['routed_to'] == 'ViT':
            vit_latencies.append(stats['vit_latency'])
        
        # Progress indicator (every 100 images)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{x} images...")

    # Calculate metrics
    accuracy = 100 * correct / x
    avg_latency = total_latency / x
    avg_cnn_latency = sum(cnn_latencies) / len(cnn_latencies)
    avg_vit_latency = sum(vit_latencies) / len(vit_latencies) if vit_latencies else 0

    # Baselines ( ADJUST BASED ON ANALYSIS )
    vit_baseline_latency = 96.13  # ms
    vit_baseline_accuracy = 99.00  # %
    speedup = vit_baseline_latency / avg_latency

    # Print results
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print("="*70)

    print(f"\nOVERALL PERFORMANCE")
    print("-"*70)
    print(f"  Test Images:     {x}")
    print(f"  Correct:         {correct}")
    print(f"  Accuracy:        {accuracy:.2f}%")
    print(f"  Avg Latency:     {avg_latency:.2f} ms")

    print(f"\nROUTING STATISTICS")
    print("-"*70)
    print(f"  CNN Usage:       {cnn_count} images ({100*cnn_count/x:.1f}%)")
    print(f"  ViT Usage:       {vit_count} images ({100*vit_count/x:.1f}%)")

    print(f"\nLATENCY BREAKDOWN")
    print("-"*70)
    print(f"  Avg CNN Latency: {avg_cnn_latency:.2f} ms (always runs)")
    print(f"  Avg ViT Latency: {avg_vit_latency:.2f} ms (when routed)")
    print(f"  Avg Total:       {avg_latency:.2f} ms")

    print(f"\nBASELINE COMPARISON")
    print("-"*70)
    print(f"  ViT-only Accuracy: {vit_baseline_accuracy:.2f}%")
    print(f"  ViT-only Latency:  {vit_baseline_latency:.2f} ms")
    print(f"  Adaptive Accuracy: {accuracy:.2f}%")
    print(f"  Adaptive Latency:  {avg_latency:.2f} ms")

    print(f"\nPERFORMANCE GAINS")
    print("-"*70)
    print(f"  Speedup vs ViT:  {speedup:.2f}x")
    print(f"  Accuracy Retention: {accuracy/vit_baseline_accuracy:.2%}")

    print(f"\n{'='*70}")

        
        

        
        
        

    