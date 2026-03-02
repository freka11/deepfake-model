import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import argparse

# Load your trained model
def load_model(model_path="models/best_model.pt"):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )
    
    # Try to load multi-dataset model first, then fallback to specified path
    model_paths = [
        "models/multi_dataset_model.ckpt",
        model_path
    ]
    
    for path in model_paths:
        try:
            if path.endswith('.ckpt'):
                # Load PyTorch Lightning checkpoint
                checkpoint = torch.load(path, map_location="cpu")
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Load regular PyTorch model
                model.load_state_dict(torch.load(path, map_location="cpu"))
            
            print(f"✅ Loaded model: {path}")
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
            continue
    
    model.eval()
    return model

# Preprocess and classify image
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = torch.max(probs).item()
        if confidence < 0.9:
            pred = 1  # Force to FAKE

    label = "FAKE" if pred == 1 else "REAL"
    print(f"\n🧠 Prediction: {label}")
    print(f"Real: {probs[0]:.3f} | Fake: {probs[1]:.3f} | Confidence: {confidence:.3f}")
# Run from terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file (.jpg/.png)")
    args = parser.parse_args()

    model = load_model()
    predict_image(args.image_path, model)
