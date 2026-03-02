import torch
import numpy as np

# Test the confidence threshold logic
def test_confidence_logic():
    print("Testing confidence threshold logic...")
    
    # Test case 1: High confidence real prediction (should remain REAL)
    probs1 = torch.tensor([0.95, 0.05])  # 95% real, 5% fake
    conf1, pred1 = torch.max(probs1, dim=0)
    if conf1.item() < 0.9:
        pred1 = torch.tensor(1)
    label1 = "FAKE" if pred1.item() == 1 else "REAL"
    print(f"Test 1 - High confidence real: {label1} (confidence: {conf1.item():.3f})")
    
    # Test case 2: Low confidence real prediction (should become FAKE)
    probs2 = torch.tensor([0.85, 0.15])  # 85% real, 15% fake
    conf2, pred2 = torch.max(probs2, dim=0)
    if conf2.item() < 0.9:
        pred2 = torch.tensor(1)
    label2 = "FAKE" if pred2.item() == 1 else "REAL"
    print(f"Test 2 - Low confidence real: {label2} (confidence: {conf2.item():.3f})")
    
    # Test case 3: High confidence fake prediction (should remain FAKE)
    probs3 = torch.tensor([0.05, 0.95])  # 5% real, 95% fake
    conf3, pred3 = torch.max(probs3, dim=0)
    if conf3.item() < 0.9:
        pred3 = torch.tensor(1)
    label3 = "FAKE" if pred3.item() == 1 else "REAL"
    print(f"Test 3 - High confidence fake: {label3} (confidence: {conf3.item():.3f})")
    
    # Test case 4: Low confidence fake prediction (should remain FAKE)
    probs4 = torch.tensor([0.15, 0.85])  # 15% real, 85% fake
    conf4, pred4 = torch.max(probs4, dim=0)
    if conf4.item() < 0.9:
        pred4 = torch.tensor(1)
    label4 = "FAKE" if pred4.item() == 1 else "REAL"
    print(f"Test 4 - Low confidence fake: {label4} (confidence: {conf4.item():.3f})")

if __name__ == "__main__":
    test_confidence_logic()
