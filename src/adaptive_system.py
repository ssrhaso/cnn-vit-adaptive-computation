import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import time
from PIL import Image



class AdaptiveClassifier:
    def __init__(
        self,
        cnn_model: torch.nn.Module,
        vit_checkpoint_path: str,
        threshold: float = 0.90,
        device: torch.device = torch.device("cpu")
    ):
        self.device = device
        self.threshold = threshold
        
        # LOAD CNN MODEL
        # LOAD VIT MODEL
        # SETUP PREPROCESSING TRANSFORMS FOR VIT
        # SET MODELS TO EVALUATION MODE
        
        pass
    
    
    def predict(
        self,
        image: Image.Image
    ) -> int:
        """ ADAPTIVE PREDICTION METHOD """
        
        # REROUTING LOGIC BASED ON CNN CONFIDENCE SCORES
        pass
    
    
    def _predict_cnn(
        self,
        image: Image.Image
    ):
        """ PREDICTION USING CNN MODEL """
        
        # PREPROCESS IMAGE FOR CNN (32x32)
        # FP
        # CONFIDENCE EXTRACTION (SOFTMAX)
        pass
    
    def _predict_vit(
        self,
        image: Image.Image
    ):
        """ PREDICTION USING VIT MODEL """
        
        # PREPROCESS IMAGE FOR VIT (224x224)
        # FP
        # CONFIDENCE EXTRACTION (SOFTMAX)
        pass
    
    