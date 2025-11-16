import numpy as np

def softmax(logits):
    shifted = logits - np.max(logits)
    softmax = np.exp(shifted) / np.sum(np.exp(shifted))
    
    return softmax

logits = np.array([2000, 2001, 2002])
print(softmax(logits))
