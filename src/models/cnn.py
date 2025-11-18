import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """SIMPLE CNN FOR CIFAR-10 CLASSIFICATION
    
    ~ 2.5m PARAMETERS
    TARGET ACC ~ 85% ON TEST SET
    TARGET INFERENCE ~ 10ms PER IMAGE (BATCH SIZE 1, CPU) 
    """
    
    def __init__(
        self,
        num_classes: int = 10
    ):
        super(CIFAR10CNN, self).__init__()
        
        # CONVOLUTIONAL LAYERS
        
        # BLOCK 1 (32x32) -> (16x16)
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 64 , kernel_size= 3, padding= 1)           # CONV LAYER (3 IN, 64 OUT, 3x3 KERNEL, PADDING 1)
        self.bn1 = nn.BatchNorm2d(64)                                                                   # NORMALISE ACTIVATION 
        self.conv2 = nn.Conv2d(in_channels= 64, out_channels= 64 , kernel_size= 3, padding= 1)          # CONV LAYER (64 IN, 64 OUT, 3x3 KERNEL, PADDING 1)
        self.bn2 = nn.BatchNorm2d(64)                                                                   # NORMALISE ACTIVATION
        self.pool1 = nn.MaxPool2d(kernel_size= 2, stride= 2)                                            # MAX POOLING (2x2 KERNEL, STRIDE 2)
        self.dropout1 = nn.Dropout(0.20)                                                                # DROPOUT (20%) - AVOID OVERFITTING
        
        # BLOCK 2 (16x16) -> (8x8)
        self.conv3 = nn.Conv2d(in_channels= 64, out_channels= 128 , kernel_size= 3, padding= 1)         # CONV LAYER (64 IN, 128 OUT, 3x3 KERNEL, PADDING 1)
        self.bn3 = nn.BatchNorm2d(128)                                                                  # NORMALISE ACTIVATION
        self.conv4 = nn.Conv2d(in_channels= 128, out_channels= 128 , kernel_size= 3, padding= 1)        # CONV LAYER (128 IN, 128 OUT, 3x3 KERNEL, PADDING 1)
        self.bn4 = nn.BatchNorm2d(128)                                                                  # NORMALISE ACTIVATION
        self.pool2 = nn.MaxPool2d(kernel_size= 2, stride= 2)                                            # MAX POOLING (2x2 KERNEL, STRIDE 2)
        self.dropout2 = nn.Dropout(0.30)                                                                # DROPOUT (30%) - AVOID OVERFITTING
        
        # BLOCK 3 (8x8) -> (4x4)
        self.conv5 = nn.Conv2d(in_channels= 128, out_channels= 256 , kernel_size= 3, padding= 1)        # CONV LAYER (128 IN, 256 OUT, 3x3 KERNEL, PADDING 1)
        self.bn5 = nn.BatchNorm2d(256)                                                                  # NORMALISE ACTIVATION
        self.conv6 = nn.Conv2d(in_channels= 256, out_channels= 256 , kernel_size= 3, padding= 1)        # CONV LAYER (256 IN, 256 OUT, 3x3 KERNEL, PADDING 1)
        self.bn6 = nn.BatchNorm2d(256)                                                                  # NORMALISE ACTIVATION
        self.pool3 = nn.MaxPool2d(kernel_size= 2, stride= 2)                                            # MAX POOLING (2x2 KERNEL, STRIDE 2)
        self.dropout3 = nn.Dropout(0.40)                                                                # DROPOUT (40%) - AVOID OVERFITTING

        # STOP AT 4x4 FEATURE MAPS IN ORDER TO KEEP MODEL LIGHTWEIGHT AND PREVENT OVERFITTING
        
        
        # CLASSIFIER (FULLY CONNECTED LAYERS), FLATTEN + FC LAYERS
        self.fc = nn.Linear(in_features = 256 * 4 * 4 , out_features = 512)                             # FULLY CONNECTED LAYER (256*4*4 IN, 512 OUT)
        self.bn_fc = nn.BatchNorm1d(512)                                                                # NORMALISE ACTIVATION
        self.dropout_fc = nn.Dropout(0.50)                                                              # DROPOUT (50%) - AVOID OVERFITTING
        self.out = nn.Linear(in_features = 512 , out_features = num_classes)                            # OUTPUT LAYER (512 IN, num_classes OUT)
        
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """ FORWARD PASS """
    
        # BLOCK 1
        x = F.relu(self.bn1(self.conv1(x)))        # CONV1 + BN + RELU
        x = F.relu(self.bn2(self.conv2(x)))        # CONV2 + BN + RELU
        x = self.pool1(x)                          # MAX POOLING
        x = self.dropout1(x)                       # DROPOUT
        
        # BLOCK 2
        x = F.relu(self.bn3(self.conv3(x)))        # CONV3 + BN + RELU
        x = F.relu(self.bn4(self.conv4(x)))        # CONV4 + BN + RELU
        x = self.pool2(x)                          # MAX POOLING
        x = self.dropout2(x)                       # DROPOUT
        
        # BLOCK 3
        x = F.relu(self.bn5(self.conv5(x)))        # CONV5 + BN + RELU
        x = F.relu(self.bn6(self.conv6(x)))        # CONV6 + BN + RELU
        x = self.pool3(x)                          # MAX POOLING
        x = self.dropout3(x)                       # DROPOUT
        
        # FLATTEN
        x = x.view(x.size(0), -1)                  # FLATTEN
        
        # CLASSIFIER
        x = F.relu(self.bn_fc(self.fc(x)))         # FULLY CONNECTED (FC) + BATCH NORMALIZATION (BN) + RELU (ACTIVATION)
        x = self.dropout_fc(x)                     # DROPOUT
        x = self.out(x)                            # OUTPUT LAYER
        
        return x        
    
    def get_confidence(
        self, 
        x: torch.Tensor
    ):
        """ GET PREDICTED CONFIDENCE (FOR ADAPTIVE GATING)"""
        logits = self.forward(x)                                        # FORWARD PASS
        probs = F.softmax(logits, dim = 1)                               # SOFTMAX TO GET PROBABILITIES
        confidence, predicted = torch.max(probs, dim = 1)                    # MAX PROBABILITY AS CONFIDENCE
        return confidence, predicted
        
        
        

# ENTRY
if __name__ == "__main__":
    model = CIFAR10CNN()
    
    # PARAM
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")  
    
    # TEST FORWARD PASS
    model.eval()
    test_input = torch.randn(1, 3, 32, 32)                          # Batch size 1, 3 channels, 32x32 image
    test_output = model(test_input)
    print(f"Test output shape: {test_output.shape}")                 # Should be (1, 10) for 10 classes
    
    # TEST CONFIDENCE
    confidence, predicted = model.get_confidence(test_input)
    print(f"Predicted class index: {predicted.item()}, Confidence: {confidence.item():.4f}")
    
    
        