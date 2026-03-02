import torch

# Test current logic exactly as implemented
probs = torch.tensor([0.85, 0.15])  # Should be FAKE (confidence < 0.9)
conf, pred = torch.max(probs, dim=0)
print(f'Before: pred={pred.item()}, conf={conf.item():.3f}')
if conf.item() < 0.9:
    pred = torch.tensor(1)
print(f'After: pred={pred.item()}, conf={conf.item():.3f}')
print(f'Result: {"FAKE" if pred.item() == 1 else "REAL"}')
